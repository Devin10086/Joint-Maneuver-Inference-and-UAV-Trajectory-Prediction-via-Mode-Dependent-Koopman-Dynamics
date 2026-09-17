# -*- coding: utf-8 -*-
"""
flightbertpp_uav_regression_v3_paper_faithful.py

FlightBERT++ paper-faithful regression adaptation for AirSim UAV trajectory data.

Why this script exists:
    The official FlightBERT++ code is public, but it is designed for ADS-B style
    longitude/latitude/altitude/speed records and Gray-code classification.
    Your AirSim data is local Cartesian [x,y,z,vx,vy,vz], often including negative
    coordinates. Directly forcing it into the original Gray-code pipeline is not a
    clean comparison.

This V3 keeps the paper/source architecture as faithfully as practical while adapting only the representation head from Gray-code MBC to continuous regression:
    1. non-autoregressive multi-horizon prediction
    2. Transformer encoder for observed trajectory
    3. horizon-aware context generator
    4. differential-prompted decoder
    5. differential prediction paradigm

The original Gray-code/BCE output is intentionally replaced by continuous regression because the AirSim UAV benchmark uses signed local Cartesian states and is compared against continuous-regression baselines. The encoder/HACG/DPD structure follows the paper and official repository.

Input:
    DATA_ROOT/
      train/chains/*.csv
      val/chains/*.csv
      test/chains/*.csv

Required CSV columns:
    x,y,z,vx,vy,vz

Example:
    python flightbertpp_uav_regression_v3_paper_faithful.py \
      --data_root DATA/uav_koopman_v2_city_intent_env/processed \
      --epochs 80 \
      --out_dir DATA/uav_koopman_v2_city_intent_env/processed/flightbertpp_uav_regression_v2

Fast:
    python flightbertpp_uav_regression_v3_paper_faithful.py \
      --data_root DATA/uav_koopman_v2_city_intent_env/processed \
      --epochs 5 \
      --d_model 128 \
      --n_layers_enc 2 \
      --n_layers_dec 1 \
      --out_dir DATA/uav_koopman_v2_city_intent_env/processed/flightbertpp_uav_regression_v2_test
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
STATE_DIM = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# General utilities
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    pe = err[..., :3]
    ve = err[..., 3:]
    disp = np.linalg.norm(pe, axis=-1)
    return {
        "pos_rmse": float(np.sqrt(np.mean(pe ** 2))),
        "pos_mae": float(np.mean(np.abs(pe))),
        "vel_rmse": float(np.sqrt(np.mean(ve ** 2))),
        "vel_mae": float(np.mean(np.abs(ve))),
        "all_rmse": float(np.sqrt(np.mean(err ** 2))),
        "all_mae": float(np.mean(np.abs(err))),
        "ade": float(np.mean(disp)),
        "fde": float(np.mean(disp[:, -1])),
    }


# =============================================================================
# Data loading
# =============================================================================
def load_split_chains(root: Path, split: str) -> List[pd.DataFrame]:
    chain_dir = root / split / "chains"
    files = sorted(chain_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No csv files found in {chain_dir}")

    chains = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in STATE_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{f} missing required columns: {missing}")
        if len(df) < 3:
            continue
        df = df.copy()
        df["_chain_file"] = f.name
        chains.append(df)

    if not chains:
        raise RuntimeError(f"No valid chains loaded from {chain_dir}")
    return chains


@dataclass
class LocalFrame:
    origin: np.ndarray
    c: float
    s: float


def make_local_frame(last_state: np.ndarray, prev_state: Optional[np.ndarray] = None) -> LocalFrame:
    origin = last_state[:3].astype(np.float64).copy()
    vx, vy = float(last_state[3]), float(last_state[4])
    speed = math.sqrt(vx * vx + vy * vy)

    if speed < 1e-6 and prev_state is not None:
        dx = float(last_state[0] - prev_state[0])
        dy = float(last_state[1] - prev_state[1])
        speed = math.sqrt(dx * dx + dy * dy)
        if speed >= 1e-6:
            vx, vy = dx, dy

    if speed < 1e-6:
        c, s = 1.0, 0.0
    else:
        c, s = vx / speed, vy / speed

    return LocalFrame(origin=origin, c=float(c), s=float(s))


def global_to_local(states: np.ndarray, frame: LocalFrame) -> np.ndarray:
    x = states.astype(np.float64)
    pos = x[..., :3]
    vel = x[..., 3:]
    rel = pos - frame.origin

    lx = frame.c * rel[..., 0] + frame.s * rel[..., 1]
    ly = -frame.s * rel[..., 0] + frame.c * rel[..., 1]
    lz = rel[..., 2]

    lvx = frame.c * vel[..., 0] + frame.s * vel[..., 1]
    lvy = -frame.s * vel[..., 0] + frame.c * vel[..., 1]
    lvz = vel[..., 2]

    return np.stack([lx, ly, lz, lvx, lvy, lvz], axis=-1)


def batch_local_to_global(local_states: np.ndarray, origins: np.ndarray, cs: np.ndarray, ss: np.ndarray) -> np.ndarray:
    lx, ly, lz = local_states[..., 0], local_states[..., 1], local_states[..., 2]
    lvx, lvy, lvz = local_states[..., 3], local_states[..., 4], local_states[..., 5]

    c = cs[:, None]
    s = ss[:, None]
    origin = origins[:, None, :]

    gx = origin[..., 0] + c * lx - s * ly
    gy = origin[..., 1] + s * lx + c * ly
    gz = origin[..., 2] + lz

    gvx = c * lvx - s * lvy
    gvy = s * lvx + c * lvy
    gvz = lvz

    return np.stack([gx, gy, gz, gvx, gvy, gvz], axis=-1)


@dataclass
class StateNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float64)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float64)

    def to_json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_normalizer(train_chains: List[pd.DataFrame], input_len: int, horizon: int, stride: int) -> StateNormalizer:
    states = []
    for df in train_chains:
        arr = df[STATE_COLS].to_numpy(dtype=np.float64)
        max_start = len(df) - input_len - horizon
        if max_start < 0:
            continue
        for s0 in range(0, max_start + 1, stride):
            hist = arr[s0:s0 + input_len]
            fut = arr[s0 + input_len:s0 + input_len + horizon]
            frame = make_local_frame(hist[-1], hist[-2] if input_len > 1 else None)
            states.append(global_to_local(hist, frame))
            states.append(global_to_local(fut, frame))
    if not states:
        raise RuntimeError("No states collected for normalizer.")
    all_states = np.concatenate(states, axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0)
    std[std < 1e-8] = 1.0
    return StateNormalizer(mean=mean, std=std)


class UAVWindowDataset(Dataset):
    def __init__(self, chains, input_len, horizon, stride, norm: StateNormalizer):
        self.samples = []
        self.input_len = input_len
        self.horizon = horizon
        self.norm = norm

        for df in chains:
            arr = df[STATE_COLS].to_numpy(dtype=np.float64)
            max_start = len(df) - input_len - horizon
            if max_start < 0:
                continue

            for s0 in range(0, max_start + 1, stride):
                hist_g = arr[s0:s0 + input_len]
                fut_g = arr[s0 + input_len:s0 + input_len + horizon]
                frame = make_local_frame(hist_g[-1], hist_g[-2] if input_len > 1 else None)
                hist_l = global_to_local(hist_g, frame)
                fut_l = global_to_local(fut_g, frame)

                hist_std = norm.transform(hist_l)
                fut_std = norm.transform(fut_l)

                # Differential target:
                # d_1 = future_1 - last_history
                # d_k = future_k - future_{k-1}
                prev = np.concatenate([hist_std[-1:,:], fut_std[:-1,:]], axis=0)
                fut_diff = fut_std - prev

                # Differential prompt for decoder: observed standardized differences.
                hist_diff = hist_std[1:] - hist_std[:-1]

                row = df.iloc[s0 + input_len - 1]
                if "behavior" in df.columns:
                    beh_seq = [str(x).strip().lower() for x in df.iloc[s0:s0+input_len]["behavior"].tolist()]
                    dominant = pd.Series(beh_seq).mode().iloc[0] if len(beh_seq) else "unknown"
                else:
                    dominant = "unknown"

                self.samples.append({
                    "hist_std": hist_std.astype(np.float32),
                    "hist_diff": hist_diff.astype(np.float32),
                    "fut_std": fut_std.astype(np.float32),
                    "fut_diff": fut_diff.astype(np.float32),
                    "fut_global": fut_g.astype(np.float32),
                    "origin": frame.origin.astype(np.float32),
                    "c": np.float32(frame.c),
                    "s": np.float32(frame.s),
                    "scene_type": str(row.get("scene_type", "unknown")),
                    "dominant_behavior": dominant,
                    "chain_file": str(row.get("_chain_file", "")),
                })

        if not self.samples:
            raise RuntimeError(f"No samples created for input_len={input_len}, horizon={horizon}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["hist_std"]),
            torch.from_numpy(s["hist_diff"]),
            torch.from_numpy(s["fut_std"]),
            torch.from_numpy(s["fut_diff"]),
            torch.from_numpy(s["fut_global"]),
            torch.from_numpy(s["origin"]),
            torch.tensor(s["c"], dtype=torch.float32),
            torch.tensor(s["s"], dtype=torch.float32),
            {
                "scene_type": s["scene_type"],
                "dominant_behavior": s["dominant_behavior"],
                "chain_file": s["chain_file"],
            }
        )


# =============================================================================
# Model
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])



class ConvTokenEmbedding(nn.Module):
    """
    Same token-conv idea as the official FlightBERT++ EncTokenEmbedding/DecTokenEmbedding:
    Conv1d over the temporal dimension with kernel_size=3, then transpose back to [B,L,D].
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")
        if self.tokenConv.bias is not None:
            nn.init.zeros_(self.tokenConv.bias)

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FeatureAggregation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, max(d_model // 8, 8)),
            nn.GELU(),
            nn.Linear(max(d_model // 8, 8), 1),
        )

    def forward(self, x):
        # x: [B,L,D]
        energy = self.proj(x)
        w = torch.softmax(energy.squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class FlightBERTPPUAVRegressor(nn.Module):
    def __init__(
        self,
        input_len=20,
        horizon=10,
        state_dim=6,
        d_model=192,
        n_heads=4,
        n_layers_enc=4,
        n_layers_dec=2,
        dropout=0.1,
        embedding_type="conv",
        zero_head_init=False,
    ):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.state_dim = state_dim
        self.d_model = d_model

        # Paper/source-faithful trajectory point embedding:
        # official FlightBERT++ uses a Conv1d token embedding directly, without
        # an extra LayerNorm/GELU immediately after the embedding.
        if embedding_type == "conv":
            self.enc_embed = ConvTokenEmbedding(state_dim, d_model)
        elif embedding_type == "linear":
            # Retained only as an ablation option; not the recommended faithful setting.
            self.enc_embed = nn.Linear(state_dim, d_model)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        self.enc_pos = PositionalEncoding(d_model, dropout=dropout, max_len=input_len + horizon + 8)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)
        self.agg = FeatureAggregation(d_model)

        # Differential prompted decoder
        # Differential-prompt embedding mirrors the official Conv1d DPD input.
        if embedding_type == "conv":
            self.diff_embed = ConvTokenEmbedding(state_dim, d_model)
        else:
            self.diff_embed = nn.Linear(state_dim, d_model)

        # Horizon-aware Context Generator (HACG): horizon embedding + trajectory-level
        # context concatenation followed by one Linear projection, matching the official source.
        self.horizon_emb = nn.Embedding(horizon, d_model)
        self.horizon_linear = nn.Linear(d_model * 2, d_model)
        self.dec_pos = PositionalEncoding(d_model, dropout=dropout, max_len=input_len + horizon + 8)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers_dec)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, state_dim, bias=False),
        )

        # Optional ablation only. The official FlightBERT++ source does not zero-initialize
        # the final prediction head, so faithful runs should leave this disabled.
        if zero_head_init:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, hist_std, hist_diff):
        """
        hist_std:  [B,L,6]
        hist_diff: [B,L-1,6]
        output:
            pred_future_std [B,H,6]
            pred_diff       [B,H,6]
        """
        B = hist_std.size(0)
        device = hist_std.device

        enc = self.enc_embed(hist_std)
        enc = self.enc_pos(enc)
        memory = self.encoder(enc)

        context = self.agg(memory)  # [B,D]

        h_ids = torch.arange(self.horizon, device=device).unsqueeze(0).repeat(B, 1)
        h_emb = self.horizon_emb(h_ids)
        ctx = context.unsqueeze(1).repeat(1, self.horizon, 1)
        horizon_query = self.horizon_linear(torch.cat([ctx, h_emb], dim=-1))

        diff_prompt = self.diff_embed(hist_diff)
        dec_in = torch.cat([diff_prompt, horizon_query], dim=1)
        dec_in = self.dec_pos(dec_in)

        dec_out = self.decoder(tgt=dec_in, memory=memory)
        future_tokens = dec_out[:, -self.horizon:, :]
        pred_diff = self.head(future_tokens)

        # reconstruct standardized future by cumulative differentials
        last = hist_std[:, -1:, :]
        pred_future = torch.cumsum(torch.cat([last, pred_diff], dim=1), dim=1)[:, 1:, :]
        return pred_future, pred_diff


# =============================================================================
# Training / evaluation
# =============================================================================
def loss_fn(pred_future, pred_diff, true_future, true_diff, pos_weight=1.0, vel_weight=0.2, diff_weight=0.5, endpoint_weight=0.5):
    pos_loss = ((pred_future[..., :3] - true_future[..., :3]) ** 2).mean()
    vel_loss = ((pred_future[..., 3:] - true_future[..., 3:]) ** 2).mean()
    diff_loss = ((pred_diff - true_diff) ** 2).mean()
    end_loss = ((pred_future[:, -1, :3] - true_future[:, -1, :3]) ** 2).mean()
    return pos_weight * pos_loss + vel_weight * vel_loss + diff_weight * diff_loss + endpoint_weight * end_loss


def evaluate(model, ds, norm: StateNormalizer, batch_size, horizons_to_report, return_sample_df=True):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    y_true_all = []
    y_pred_all = []
    meta_rows = []

    with torch.no_grad():
        for hist, hdiff, fut, fdiff, fut_g, origin, c, s, meta in dl:
            hist = hist.to(DEVICE)
            hdiff = hdiff.to(DEVICE)
            pred_std, _ = model(hist, hdiff)
            pred_std = pred_std.detach().cpu().numpy()

            local_phys = norm.inverse(pred_std)
            pred_g = batch_local_to_global(local_phys, origin.numpy(), c.numpy(), s.numpy())
            true_g = fut_g.numpy().astype(np.float64)

            y_true_all.append(true_g)
            y_pred_all.append(pred_g)

            if return_sample_df:
                bs = pred_g.shape[0]
                for i in range(bs):
                    meta_rows.append({
                        "scene_type": meta["scene_type"][i],
                        "dominant_behavior": meta["dominant_behavior"][i],
                        "chain_file": meta["chain_file"][i],
                    })

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    rows = []
    sample_rows = []
    for H in horizons_to_report:
        met = compute_metrics(y_true[:, :H, :], y_pred[:, :H, :])
        rows.append({"horizon": H, **met})

        if return_sample_df:
            for i in range(y_true.shape[0]):
                im = compute_metrics(y_true[i:i+1, :H, :], y_pred[i:i+1, :H, :])
                sample_rows.append({
                    "method": "flightbertpp_uav_regression",
                    "horizon": H,
                    **meta_rows[i],
                    **im,
                })

    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def baseline_predictions(ds, method, norm, dt=0.1):
    y_true = []
    y_pred = []
    metas = []
    for s in ds.samples:
        fut_g = s["fut_global"].astype(np.float64)
        hist_local = norm.inverse(s["hist_std"].astype(np.float64))
        # Use global reconstruction for classical baselines.
        # The stored local frame is enough to convert history back to global.
        origins = s["origin"][None, :]
        cs = np.array([s["c"]], dtype=np.float64)
        ss = np.array([s["s"]], dtype=np.float64)
        hist_g = batch_local_to_global(hist_local[None, ...], origins, cs, ss)[0]

        last = hist_g[-1]
        prev = hist_g[-2] if hist_g.shape[0] > 1 else hist_g[-1]
        H = ds.horizon

        if method == "constant_velocity":
            pos = last[:3].copy()
            vel = last[3:].copy()
            pred = []
            for _ in range(H):
                pos = pos + dt * vel
                pred.append(np.concatenate([pos, vel]))
            pred = np.stack(pred)
        elif method == "constant_acceleration":
            pos = last[:3].copy()
            vel = last[3:].copy()
            acc = (last[3:] - prev[3:]) / max(dt, 1e-8)
            pred = []
            for _ in range(H):
                pos = pos + dt * vel + 0.5 * dt * dt * acc
                vel = vel + dt * acc
                pred.append(np.concatenate([pos, vel]))
            pred = np.stack(pred)
        else:
            raise ValueError(method)

        y_true.append(fut_g)
        y_pred.append(pred)
        metas.append(s)

    return np.stack(y_true), np.stack(y_pred), metas



def choose_score(row, metric="combo", ade_weight=0.5, fde_weight=0.25):
    if metric == "pos_rmse":
        return float(row["pos_rmse"])
    if metric == "ade":
        return float(row["ade"])
    if metric == "fde":
        return float(row["fde"])
    if metric == "combo":
        return float(row["pos_rmse"]) + ade_weight * float(row["ade"]) + fde_weight * float(row["fde"])
    raise ValueError(metric)


def eval_baseline(ds, method, norm, horizons, dt):
    y_true, y_pred, metas = baseline_predictions(ds, method, norm, dt)
    rows = []
    sample_rows = []
    for H in horizons:
        met = compute_metrics(y_true[:, :H, :], y_pred[:, :H, :])
        rows.append({"method": method, "horizon": H, **met})
        for i, s in enumerate(metas):
            im = compute_metrics(y_true[i:i+1, :H, :], y_pred[i:i+1, :H, :])
            sample_rows.append({
                "method": method,
                "horizon": H,
                "scene_type": s["scene_type"],
                "dominant_behavior": s["dominant_behavior"],
                "chain_file": s["chain_file"],
                **im,
            })
    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def main():
    ap = argparse.ArgumentParser(description="FlightBERT++-style regression adaptation for UAV data.")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--input_len", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--report_horizons", type=str, default="1,3,5,10")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers_enc", type=int, default=4)
    ap.add_argument("--n_layers_dec", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--embedding_type", type=str, default="conv", choices=["conv", "linear"])
    ap.add_argument("--zero_head_init", action="store_true")
    ap.add_argument("--pos_weight", type=float, default=1.0)
    ap.add_argument("--vel_weight", type=float, default=0.2)
    ap.add_argument("--diff_weight", type=float, default=0.5)
    ap.add_argument("--endpoint_weight", type=float, default=0.5)
    ap.add_argument("--val_metric", type=str, default="combo", choices=["pos_rmse", "ade", "fde", "combo"])
    ap.add_argument("--combo_ade_weight", type=float, default=0.5)
    ap.add_argument("--combo_fde_weight", type=float, default=0.25)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_parallel", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "flightbertpp_uav_regression_v2"
    ensure_dir(out_dir)

    report_horizons = parse_int_list(args.report_horizons)
    report_horizons = [h for h in report_horizons if h <= args.horizon]

    train_chains = load_split_chains(root, "train")
    val_chains = load_split_chains(root, "val")
    test_chains = load_split_chains(root, "test")

    norm = fit_normalizer(train_chains, args.input_len, args.horizon, args.stride)

    train_ds = UAVWindowDataset(train_chains, args.input_len, args.horizon, args.stride, norm)
    val_ds = UAVWindowDataset(val_chains, args.input_len, args.horizon, args.stride, norm)
    test_ds = UAVWindowDataset(test_chains, args.input_len, args.horizon, args.stride, norm)

    print(f"samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"device: {DEVICE}, cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    model = FlightBERTPPUAVRegressor(
        input_len=args.input_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers_enc=args.n_layers_enc,
        n_layers_dec=args.n_layers_dec,
        dropout=args.dropout,
        embedding_type=args.embedding_type,
        zero_head_init=args.zero_head_init,
    ).to(DEVICE)

    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    best_path = out_dir / "flightbertpp_uav_regression_best.pt"
    train_log = []
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for hist, hdiff, fut, fdiff, _fut_g, _origin, _c, _s, _meta in train_dl:
            hist = hist.to(DEVICE, non_blocking=True)
            hdiff = hdiff.to(DEVICE, non_blocking=True)
            fut = fut.to(DEVICE, non_blocking=True)
            fdiff = fdiff.to(DEVICE, non_blocking=True)

            pred_fut, pred_diff = model(hist, hdiff)
            loss = loss_fn(
                pred_fut, pred_diff, fut, fdiff,
                pos_weight=args.pos_weight,
                vel_weight=args.vel_weight,
                diff_weight=args.diff_weight,
                endpoint_weight=args.endpoint_weight,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total += float(loss.detach().cpu()) * hist.size(0)
            n += hist.size(0)

        val_df, _ = evaluate(model, val_ds, norm, args.batch_size, [args.horizon], return_sample_df=False)
        val_metric = choose_score(
            val_df.iloc[0],
            metric=args.val_metric,
            ade_weight=args.combo_ade_weight,
            fde_weight=args.combo_fde_weight,
        )
        row = {
            "epoch": ep,
            "train_loss": total / max(n, 1),
            "val_score": val_metric,
            "val_pos_rmse": float(val_df.iloc[0]["pos_rmse"]),
            "val_ade": float(val_df.iloc[0]["ade"]),
            "val_fde": float(val_df.iloc[0]["fde"]),
        }
        train_log.append(row)

        print(f"epoch={ep:03d} train_loss={row['train_loss']:.6f} val_score={row['val_score']:.6f} val_pos_rmse={row['val_pos_rmse']:.6f} val_ade={row['val_ade']:.6f} val_fde={row['val_fde']:.6f}")

        if val_metric < best_score:
            best_score = val_metric
            bad = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_path)
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print(f"early stop at epoch {ep}")
                break

    # Reload best
    state = torch.load(best_path, map_location=DEVICE)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    test_df, sample_df = evaluate(model, test_ds, norm, args.batch_size, report_horizons, return_sample_df=True)
    test_df.insert(0, "method", "flightbertpp_uav_regression")

    # Baselines for sanity
    base_rows = []
    base_samples = []
    for base in ["constant_velocity", "constant_acceleration"]:
        bdf, bsdf = eval_baseline(test_ds, base, norm, report_horizons, args.dt)
        base_rows.append(bdf)
        base_samples.append(bsdf)

    compare_df = pd.concat(base_rows + [test_df], ignore_index=True)
    all_sample_df = pd.concat(base_samples + [sample_df], ignore_index=True)

    compare_df.to_csv(out_dir / "compare_horizon_metrics_test.csv", index=False)
    all_sample_df.to_csv(out_dir / "sample_metrics_test.csv", index=False)
    pd.DataFrame(train_log).to_csv(out_dir / "training_log.csv", index=False)

    # Grouped metrics
    def group_sample_metrics(df, col):
        rows = []
        metric_cols = ["pos_rmse", "pos_mae", "vel_rmse", "vel_mae", "all_rmse", "all_mae", "ade", "fde"]
        for key, g in df.groupby(["method", "horizon", col], dropna=False):
            m, H, v = key
            row = {"method": m, "horizon": H, col: v, "num_samples": len(g)}
            for c in metric_cols:
                row[c] = float(g[c].mean())
            rows.append(row)
        return pd.DataFrame(rows)

    group_sample_metrics(all_sample_df, "scene_type").to_csv(out_dir / "grouped_metrics_by_scene_test.csv", index=False)
    group_sample_metrics(all_sample_df, "dominant_behavior").to_csv(out_dir / "grouped_metrics_by_behavior_test.csv", index=False)

    report = {
        "version": "flightbertpp_uav_regression_v3_paper_faithful",
        "data_root": str(root),
        "out_dir": str(out_dir),
        "input_len": args.input_len,
        "horizon": args.horizon,
        "report_horizons": report_horizons,
        "stride": args.stride,
        "dt": args.dt,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers_enc": args.n_layers_enc,
        "n_layers_dec": args.n_layers_dec,
        "dropout": args.dropout,
        "embedding_type": args.embedding_type,
        "zero_head_init": bool(args.zero_head_init),
        "pos_weight": args.pos_weight,
        "vel_weight": args.vel_weight,
        "diff_weight": args.diff_weight,
        "endpoint_weight": args.endpoint_weight,
        "val_metric": args.val_metric,
        "combo_ade_weight": args.combo_ade_weight,
        "combo_fde_weight": args.combo_fde_weight,
        "seed": args.seed,
        "normalizer": norm.to_json(),
        "notes": [
            "This is a paper/source-faithful FlightBERT++ structural adaptation with a continuous-regression output head instead of the original Gray-code MBC head.",
            "It preserves the paper/source trajectory encoder, Conv1d token embedding, attention sequence aggregation, HACG, differential-prompted decoder, and non-autoregressive differential prediction structure.",
            "It uses local-frame standardized UAV state [x,y,z,vx,vy,vz].",
            "Gray-code/BCE is not used because this benchmark operates on signed continuous local Cartesian UAV states; this is the only major representation-level adaptation.",
            "The recommended faithful setting uses conv embedding, 4 encoder layers, 2 decoder layers, no zero-head initialization, and post-norm Transformer layers as in the official source.",
            "No CV skeleton is used inside the model.",
        ],
    }
    with open(out_dir / "report_flightbertpp_uav_regression_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== FlightBERT++ UAV Regression V3 Paper-Faithful finished ===")
    print(compare_df.to_string(index=False))
    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
