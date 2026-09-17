# -*- coding: utf-8 -*-
"""
mtr_uav_lite_v3_paper_faithful.py

MTR-UAV V3: paper-faithful continuous-state adaptation of
"Motion Transformer with Global Intention Localization and Local Movement Refinement"
(NeurIPS 2022) for a single AirSim UAV.

What is kept from MTR:
    1. Agent-centric local coordinates.
    2. K-means intention points.
    3. Motion query pairs:
         - static intention query for global intention localization;
         - dynamic searching query updated from the endpoint predicted by each decoder layer.
    4. Stacked decoder layers with iterative trajectory refinement.
    5. Per-layer probabilistic trajectory heads and classification heads.
    6. Hard assignment to the intention point nearest the ground-truth endpoint.
    7. Gaussian negative log-likelihood + velocity regression + mode classification.
    8. NMS over candidate endpoints before multimodal evaluation.

Necessary single-UAV adaptations:
    - The original road-map and multi-agent encoders are unavailable in this baseline.
      The decoder therefore cross-attends to the encoded UAV history only.
    - The original 2-D correlated Gaussian head is extended to a diagonal 3-D
      Gaussian for NED position [x,y,z].
    - Inputs remain the common continuous state [x,y,z,vx,vy,vz] so the model is
      comparable with TopoKoop, FlightBERT++-UAV-Reg, and PECNet-UAV.

Data layout:
    DATA_ROOT/
      train/chains/*.csv
      val/chains/*.csv
      test/chains/*.csv

Required CSV columns:
    x,y,z,vx,vy,vz

Formal comparison uses:
    method = mtr_uav_paper_top1

The oracle method uses the ground-truth future only to select the best candidate and
must be reported as a multimodal upper bound, not as a deployable predictor:
    method = mtr_uav_paper_oracle_minfde
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
STATE_DIM = 6
POS_DIM = 3
VEL_DIM = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Metrics in global physical coordinates."""
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


def score_from_metrics(
    metrics: Dict[str, float],
    metric: str = "pos_rmse",
    ade_weight: float = 0.5,
    fde_weight: float = 0.25,
) -> float:
    if metric == "pos_rmse":
        return float(metrics["pos_rmse"])
    if metric == "ade":
        return float(metrics["ade"])
    if metric == "fde":
        return float(metrics["fde"])
    if metric == "combo":
        return (
            float(metrics["pos_rmse"])
            + ade_weight * float(metrics["ade"])
            + fde_weight * float(metrics["fde"])
        )
    raise ValueError(f"Unsupported metric: {metric}")


# =============================================================================
# Data and local coordinate frame
# =============================================================================
def load_split_chains(root: Path, split: str) -> List[pd.DataFrame]:
    chain_dir = root / split / "chains"
    files = sorted(chain_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No csv files found in {chain_dir}")

    chains: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        missing = [c for c in STATE_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{file_path} missing required columns: {missing}")
        if len(df) < 3:
            continue
        df = df.copy()
        df[STATE_COLS] = df[STATE_COLS].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=STATE_COLS).reset_index(drop=True)
        if len(df) < 3:
            continue
        df["_chain_file"] = file_path.name
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
    speed = math.hypot(vx, vy)

    if speed < 1e-6 and prev_state is not None:
        dx = float(last_state[0] - prev_state[0])
        dy = float(last_state[1] - prev_state[1])
        speed = math.hypot(dx, dy)
        if speed >= 1e-6:
            vx, vy = dx, dy

    if speed < 1e-6:
        c, s = 1.0, 0.0
    else:
        c, s = vx / speed, vy / speed
    return LocalFrame(origin=origin, c=float(c), s=float(s))


def global_to_local(states: np.ndarray, frame: LocalFrame) -> np.ndarray:
    values = states.astype(np.float64)
    pos = values[..., :3]
    vel = values[..., 3:]
    rel = pos - frame.origin

    lx = frame.c * rel[..., 0] + frame.s * rel[..., 1]
    ly = -frame.s * rel[..., 0] + frame.c * rel[..., 1]
    lz = rel[..., 2]

    lvx = frame.c * vel[..., 0] + frame.s * vel[..., 1]
    lvy = -frame.s * vel[..., 0] + frame.c * vel[..., 1]
    lvz = vel[..., 2]
    return np.stack([lx, ly, lz, lvx, lvy, lvz], axis=-1)


def batch_local_to_global(
    local_states: np.ndarray,
    origins: np.ndarray,
    cs: np.ndarray,
    ss: np.ndarray,
) -> np.ndarray:
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

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float64)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float64)

    def pos_to_std(self, positions: np.ndarray) -> np.ndarray:
        return ((positions - self.mean[:3]) / self.std[:3]).astype(np.float64)

    def pos_from_std(self, positions_std: np.ndarray) -> np.ndarray:
        return (positions_std * self.std[:3] + self.mean[:3]).astype(np.float64)

    def to_json(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_normalizer(
    train_chains: Sequence[pd.DataFrame],
    input_len: int,
    horizon: int,
    stride: int,
) -> StateNormalizer:
    states: List[np.ndarray] = []
    for df in train_chains:
        arr = df[STATE_COLS].to_numpy(dtype=np.float64)
        max_start = len(arr) - input_len - horizon
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            hist = arr[start:start + input_len]
            fut = arr[start + input_len:start + input_len + horizon]
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


class UAVMTRPaperDataset(Dataset):
    def __init__(
        self,
        chains: Sequence[pd.DataFrame],
        input_len: int,
        horizon: int,
        stride: int,
        norm: StateNormalizer,
    ):
        self.samples: List[Dict[str, object]] = []
        self.input_len = input_len
        self.horizon = horizon
        self.norm = norm

        for df in chains:
            arr = df[STATE_COLS].to_numpy(dtype=np.float64)
            max_start = len(arr) - input_len - horizon
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, stride):
                hist_g = arr[start:start + input_len]
                fut_g = arr[start + input_len:start + input_len + horizon]
                frame = make_local_frame(hist_g[-1], hist_g[-2] if input_len > 1 else None)
                hist_l = global_to_local(hist_g, frame)
                fut_l = global_to_local(fut_g, frame)
                hist_std = norm.transform(hist_l)
                fut_std = norm.transform(fut_l)

                row = df.iloc[start + input_len - 1]
                if "behavior" in df.columns:
                    seq = (
                        df.iloc[start:start + input_len]["behavior"]
                        .fillna("unknown")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .tolist()
                    )
                    dominant = pd.Series(seq).mode().iloc[0] if seq else "unknown"
                else:
                    dominant = "unknown"

                self.samples.append({
                    "hist_std": hist_std.astype(np.float32),
                    "fut_std": fut_std.astype(np.float32),
                    "fut_local": fut_l.astype(np.float32),
                    "fut_global": fut_g.astype(np.float32),
                    "origin": frame.origin.astype(np.float32),
                    "c": np.float32(frame.c),
                    "s": np.float32(frame.s),
                    "scene_type": str(row.get("scene_type", "unknown")),
                    "dominant_behavior": dominant,
                    "chain_file": str(row.get("_chain_file", "")),
                })

        if not self.samples:
            raise RuntimeError(
                f"No samples created for input_len={input_len}, horizon={horizon}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample["hist_std"]),
            torch.from_numpy(sample["fut_std"]),
            torch.from_numpy(sample["fut_local"]),
            torch.from_numpy(sample["fut_global"]),
            torch.from_numpy(sample["origin"]),
            torch.tensor(sample["c"], dtype=torch.float32),
            torch.tensor(sample["s"], dtype=torch.float32),
            {
                "scene_type": sample["scene_type"],
                "dominant_behavior": sample["dominant_behavior"],
                "chain_file": sample["chain_file"],
            },
        )


def collect_train_endpoints_local(ds: UAVMTRPaperDataset) -> np.ndarray:
    return np.stack([s["fut_local"][-1, :3] for s in ds.samples]).astype(np.float64)


# =============================================================================
# K-means intention points
# =============================================================================
def kmeans_plus_plus_init(
    values: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = values.shape[0]
    if n < k:
        raise ValueError(f"num_samples={n} is smaller than k={k}")
    centers = np.empty((k, values.shape[1]), dtype=np.float64)
    idx = int(rng.integers(0, n))
    centers[0] = values[idx]
    dist_sq = ((values - centers[0]) ** 2).sum(axis=1)

    for i in range(1, k):
        total = float(dist_sq.sum())
        if total <= 1e-12:
            idx = int(rng.integers(0, n))
        else:
            idx = int(rng.choice(n, p=dist_sq / total))
        centers[i] = values[idx]
        new_dist_sq = ((values - centers[i]) ** 2).sum(axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    return centers


def simple_kmeans(
    values: np.ndarray,
    k: int,
    seed: int,
    iterations: int = 100,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus_init(values, k, rng)
    for _ in range(iterations):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for index in range(k):
            mask = labels == index
            if mask.any():
                new_centers[index] = values[mask].mean(axis=0)
            else:
                new_centers[index] = values[int(rng.integers(0, len(values)))]
        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        if shift < 1e-6:
            break
    return centers.astype(np.float32)


# =============================================================================
# Network modules
# =============================================================================
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length, :]


def sine_position_embedding_3d(coords: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal embedding for arbitrary [...,3] coordinates."""
    if coords.shape[-1] != 3:
        raise ValueError(f"Expected [...,3] coordinates, got {coords.shape}")
    num_freq = max(1, math.ceil(d_model / (2 * 3)))
    freq = torch.exp(
        torch.arange(num_freq, device=coords.device, dtype=coords.dtype)
        * (-math.log(10000.0) / max(num_freq - 1, 1))
    )
    scaled = coords.unsqueeze(-1) / freq
    emb = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    emb = emb.flatten(start_dim=-2)
    if emb.shape[-1] < d_model:
        emb = F.pad(emb, (0, d_model - emb.shape[-1]))
    elif emb.shape[-1] > d_model:
        emb = emb[..., :d_model]
    return emb


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
        final_activation: bool = False,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or final_activation:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class MotionQueryDecoderLayer(nn.Module):
    """
    MTR-style motion-query-pair layer.

    Static intention query is added to self-attention queries/keys.
    Dynamic searching query is added to cross-attention queries and is updated
    from the endpoint predicted by the preceding layer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query_content: torch.Tensor,
        static_query: torch.Tensor,
        dynamic_query: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        # Mode-to-mode propagation under static intention positions.
        q = query_content + static_query
        self_out, _ = self.self_attn(q, q, query_content, need_weights=False)
        query_content = self.norm1(query_content + self.dropout1(self_out))

        # Search trajectory-specific temporal evidence using the dynamic endpoint query.
        q = query_content + dynamic_query
        k = memory + memory_pos
        cross_out, _ = self.cross_attn(q, k, memory, need_weights=False)
        query_content = self.norm2(query_content + self.dropout2(cross_out))

        ff = self.linear2(self.dropout(F.relu(self.linear1(query_content))))
        query_content = self.norm3(query_content + self.dropout3(ff))
        return query_content


class MTRUAVPaperFaithful(nn.Module):
    def __init__(
        self,
        intention_points_std: np.ndarray,
        input_len: int = 20,
        horizon: int = 10,
        state_dim: int = 6,
        d_model: int = 192,
        n_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        min_log_std: float = -5.0,
        max_log_std: float = 3.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.input_len = input_len
        self.horizon = horizon
        self.state_dim = state_dim
        self.d_model = d_model
        self.num_queries = int(intention_points_std.shape[0])
        self.num_decoder_layers = num_decoder_layers
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.register_buffer(
            "intention_points_std",
            torch.tensor(intention_points_std, dtype=torch.float32),
        )

        self.hist_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.temporal_pos = TemporalPositionalEncoding(d_model, max_len=input_len + 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.center_pool_score = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, d_model // 4), 1),
        )
        self.center_proj = MLP(d_model, [d_model], d_model)

        self.intention_query_mlp = MLP(d_model, [d_model], d_model)
        self.searching_query_mlp = MLP(d_model, [d_model], d_model)

        self.decoder_layers = nn.ModuleList([
            MotionQueryDecoderLayer(d_model, n_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.query_fusion_layers = nn.ModuleList([
            MLP(d_model * 2, [d_model, d_model], d_model)
            for _ in range(num_decoder_layers)
        ])

        # 3-D diagonal Gaussian: mean(3), log_std(3), velocity(3) per step.
        output_dim = horizon * 9
        self.motion_reg_heads = nn.ModuleList([
            MLP(d_model, [d_model, d_model], output_dim)
            for _ in range(num_decoder_layers)
        ])
        self.motion_cls_heads = nn.ModuleList([
            MLP(d_model, [d_model, d_model], 1)
            for _ in range(num_decoder_layers)
        ])

    def pool_center(self, memory: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.center_pool_score(memory).squeeze(-1), dim=1)
        return (memory * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, hist_std: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        batch_size, history_len, _ = hist_std.shape
        memory_pos = self.temporal_pos(history_len).to(hist_std.dtype)
        memory = self.encoder(self.hist_embed(hist_std) + memory_pos)
        center_feature = self.center_proj(self.pool_center(memory))

        points = self.intention_points_std
        static_query_single = self.intention_query_mlp(
            sine_position_embedding_3d(points, self.d_model)
        )
        static_query = static_query_single.unsqueeze(0).expand(batch_size, -1, -1)

        dynamic_center = points.unsqueeze(0).expand(batch_size, -1, -1)
        query_content = torch.zeros(
            batch_size,
            self.num_queries,
            self.d_model,
            device=hist_std.device,
            dtype=hist_std.dtype,
        )
        center_repeated = center_feature.unsqueeze(1).expand(-1, self.num_queries, -1)

        outputs: List[Dict[str, torch.Tensor]] = []
        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            dynamic_query = self.searching_query_mlp(
                sine_position_embedding_3d(dynamic_center, self.d_model)
            )
            query_content = decoder_layer(
                query_content=query_content,
                static_query=static_query,
                dynamic_query=dynamic_query,
                memory=memory,
                memory_pos=memory_pos.expand(batch_size, -1, -1),
            )
            query_content = self.query_fusion_layers[layer_idx](
                torch.cat([query_content, center_repeated], dim=-1)
            )

            raw = self.motion_reg_heads[layer_idx](query_content).view(
                batch_size, self.num_queries, self.horizon, 9
            )
            mean_pos = raw[..., 0:3]
            log_std = raw[..., 3:6].clamp(self.min_log_std, self.max_log_std)
            pred_vel = raw[..., 6:9]
            scores = self.motion_cls_heads[layer_idx](query_content).squeeze(-1)

            outputs.append({
                "scores": scores,
                "mean_pos": mean_pos,
                "log_std": log_std,
                "pred_vel": pred_vel,
            })
            dynamic_center = mean_pos[:, :, -1, :]

        return outputs


# =============================================================================
# Losses
# =============================================================================
def assign_by_intention_points_local(
    gt_future_local: torch.Tensor,
    intention_points_local: torch.Tensor,
) -> torch.Tensor:
    gt_endpoint = gt_future_local[:, -1, :3]
    distances = torch.sum(
        (gt_endpoint[:, None, :] - intention_points_local[None, :, :]) ** 2,
        dim=-1,
    )
    return distances.argmin(dim=-1)


def diagonal_gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    inv_var = torch.exp(-2.0 * log_std)
    nll = 0.5 * (target - mean) ** 2 * inv_var + log_std + 0.5 * math.log(2.0 * math.pi)
    return nll.sum(dim=-1).mean()


def mtr_paper_loss(
    layer_outputs: Sequence[Dict[str, torch.Tensor]],
    fut_std: torch.Tensor,
    fut_local: torch.Tensor,
    intention_points_local: torch.Tensor,
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    vel_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    positive_idx = assign_by_intention_points_local(
        fut_local, intention_points_local
    )
    batch_idx = torch.arange(fut_std.shape[0], device=fut_std.device)

    total = torch.zeros((), device=fut_std.device)
    reg_acc = 0.0
    vel_acc = 0.0
    cls_acc = 0.0
    acc_acc = 0.0

    for output in layer_outputs:
        mean = output["mean_pos"][batch_idx, positive_idx]
        log_std = output["log_std"][batch_idx, positive_idx]
        pred_vel = output["pred_vel"][batch_idx, positive_idx]
        scores = output["scores"]

        reg_loss = diagonal_gaussian_nll(mean, log_std, fut_std[..., :3])
        velocity_loss = F.l1_loss(pred_vel, fut_std[..., 3:])
        classification_loss = F.cross_entropy(scores, positive_idx)
        layer_loss = (
            reg_weight * reg_loss
            + vel_weight * velocity_loss
            + cls_weight * classification_loss
        )
        total = total + layer_loss
        reg_acc += float(reg_loss.detach().cpu())
        vel_acc += float(velocity_loss.detach().cpu())
        cls_acc += float(classification_loss.detach().cpu())
        acc_acc += float((scores.argmax(dim=-1) == positive_idx).float().mean().detach().cpu())

    count = max(len(layer_outputs), 1)
    total = total / count
    return total, {
        "reg_nll": reg_acc / count,
        "vel_l1": vel_acc / count,
        "cls_ce": cls_acc / count,
        "assigned_mode_acc": acc_acc / count,
    }


# =============================================================================
# NMS and evaluation
# =============================================================================
def endpoint_nms_single(
    endpoints: np.ndarray,
    scores: np.ndarray,
    num_output_modes: int,
    distance_threshold: float,
) -> np.ndarray:
    """Score-ordered endpoint NMS, with deterministic fill if suppression is strong."""
    order = np.argsort(-scores)
    selected: List[int] = []
    for idx in order:
        if not selected:
            selected.append(int(idx))
        else:
            distance = np.linalg.norm(endpoints[idx] - endpoints[selected], axis=-1)
            if np.all(distance >= distance_threshold):
                selected.append(int(idx))
        if len(selected) >= num_output_modes:
            break

    if len(selected) < num_output_modes:
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= num_output_modes:
                break
    return np.asarray(selected, dtype=np.int64)


def select_modes_with_nms(
    pred_local: np.ndarray,
    scores: np.ndarray,
    num_output_modes: int,
    distance_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        pred_local: [B,K,H,6] physical local state.
        scores: [B,K] logits.
    Returns:
        selected trajectories [B,M,H,6], selected probabilities [B,M], indices [B,M].
    """
    score_shift = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(score_shift)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    selected_trajs = []
    selected_probs = []
    selected_indices = []
    for b in range(pred_local.shape[0]):
        idx = endpoint_nms_single(
            endpoints=pred_local[b, :, -1, :3],
            scores=probs[b],
            num_output_modes=min(num_output_modes, pred_local.shape[1]),
            distance_threshold=distance_threshold,
        )
        selected_trajs.append(pred_local[b, idx])
        selected_probs.append(probs[b, idx])
        selected_indices.append(idx)
    return (
        np.stack(selected_trajs),
        np.stack(selected_probs),
        np.stack(selected_indices),
    )


def evaluate(
    model: MTRUAVPaperFaithful,
    ds: UAVMTRPaperDataset,
    norm: StateNormalizer,
    batch_size: int,
    horizons_to_report: Sequence[int],
    num_output_modes: int,
    nms_dist_thresh: float,
    num_workers: int = 0,
    save_prediction_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true_all: List[np.ndarray] = []
    y_top_all: List[np.ndarray] = []
    y_oracle_all: List[np.ndarray] = []
    selected_prob_all: List[np.ndarray] = []
    selected_idx_all: List[np.ndarray] = []
    meta_rows: List[Dict[str, str]] = []

    with torch.no_grad():
        for hist, _fut_std, _fut_local, fut_g, origin, c, s, meta in loader:
            hist = hist.to(DEVICE, non_blocking=True)
            output = model(hist)[-1]

            mean_std = output["mean_pos"].detach().cpu().numpy()
            vel_std = output["pred_vel"].detach().cpu().numpy()
            score_logits = output["scores"].detach().cpu().numpy()

            full_std = np.concatenate([mean_std, vel_std], axis=-1)
            # StateNormalizer broadcasts over [B,K,H,6].
            full_local = norm.inverse(full_std)
            selected_local, selected_probs, selected_indices = select_modes_with_nms(
                pred_local=full_local,
                scores=score_logits,
                num_output_modes=num_output_modes,
                distance_threshold=nms_dist_thresh,
            )

            origin_np = origin.numpy()
            c_np = c.numpy()
            s_np = s.numpy()
            batch_size_actual, modes, horizon, state_dim = selected_local.shape
            flat_local = selected_local.reshape(batch_size_actual * modes, horizon, state_dim)
            flat_origins = np.repeat(origin_np, modes, axis=0)
            flat_c = np.repeat(c_np, modes, axis=0)
            flat_s = np.repeat(s_np, modes, axis=0)
            selected_global = batch_local_to_global(
                flat_local, flat_origins, flat_c, flat_s
            ).reshape(batch_size_actual, modes, horizon, state_dim)

            true_global = fut_g.numpy().astype(np.float64)
            top_global = selected_global[:, 0]
            fde = np.linalg.norm(
                selected_global[:, :, -1, :3]
                - true_global[:, None, -1, :3],
                axis=-1,
            )
            oracle_idx = fde.argmin(axis=1)
            oracle_global = selected_global[np.arange(batch_size_actual), oracle_idx]

            y_true_all.append(true_global)
            y_top_all.append(top_global)
            y_oracle_all.append(oracle_global)
            selected_prob_all.append(selected_probs)
            selected_idx_all.append(selected_indices)

            for i in range(batch_size_actual):
                meta_rows.append({
                    "scene_type": meta["scene_type"][i],
                    "dominant_behavior": meta["dominant_behavior"][i],
                    "chain_file": meta["chain_file"][i],
                })

    y_true = np.concatenate(y_true_all, axis=0)
    y_top = np.concatenate(y_top_all, axis=0)
    y_oracle = np.concatenate(y_oracle_all, axis=0)
    selected_probs = np.concatenate(selected_prob_all, axis=0)
    selected_indices = np.concatenate(selected_idx_all, axis=0)

    compare_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for horizon in horizons_to_report:
        top_metrics = compute_metrics(y_true[:, :horizon], y_top[:, :horizon])
        compare_rows.append({
            "method": "mtr_uav_paper_top1",
            "horizon": horizon,
            **top_metrics,
        })
        oracle_metrics = compute_metrics(y_true[:, :horizon], y_oracle[:, :horizon])
        compare_rows.append({
            "method": "mtr_uav_paper_oracle_minfde",
            "horizon": horizon,
            **oracle_metrics,
        })

        for i in range(y_true.shape[0]):
            metrics = compute_metrics(
                y_true[i:i + 1, :horizon],
                y_top[i:i + 1, :horizon],
            )
            sample_rows.append({
                "method": "mtr_uav_paper_top1",
                "horizon": horizon,
                **meta_rows[i],
                **metrics,
            })

    if save_prediction_path is not None:
        ensure_dir(save_prediction_path.parent)
        np.savez_compressed(
            save_prediction_path,
            y_true_global=y_true,
            y_top1_global=y_top,
            y_oracle_global=y_oracle,
            selected_probs=selected_probs,
            selected_query_indices=selected_indices,
        )

    mode_report = {
        "num_test_samples": int(y_true.shape[0]),
        "num_output_modes": int(selected_probs.shape[1]),
        "mean_top1_probability": float(selected_probs[:, 0].mean()),
        "selected_query_histogram": np.bincount(
            selected_indices.reshape(-1), minlength=model.num_queries
        ).tolist(),
    }
    return pd.DataFrame(compare_rows), pd.DataFrame(sample_rows), mode_report


# =============================================================================
# Classical baselines
# =============================================================================
def baseline_predictions(
    ds: UAVMTRPaperDataset,
    method: str,
    norm: StateNormalizer,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    y_true = []
    y_pred = []
    metas = []
    for sample in ds.samples:
        future_global = sample["fut_global"].astype(np.float64)
        history_local = norm.inverse(sample["hist_std"].astype(np.float64))
        history_global = batch_local_to_global(
            history_local[None],
            sample["origin"][None],
            np.asarray([sample["c"]], dtype=np.float64),
            np.asarray([sample["s"]], dtype=np.float64),
        )[0]
        last = history_global[-1]
        prev = history_global[-2] if len(history_global) > 1 else last
        pos = last[:3].copy()
        vel = last[3:].copy()
        trajectory = []

        if method == "constant_velocity":
            for _ in range(ds.horizon):
                pos = pos + dt * vel
                trajectory.append(np.concatenate([pos, vel]))
        elif method == "constant_acceleration":
            acceleration = (last[3:] - prev[3:]) / max(dt, 1e-8)
            for _ in range(ds.horizon):
                pos = pos + dt * vel + 0.5 * dt * dt * acceleration
                vel = vel + dt * acceleration
                trajectory.append(np.concatenate([pos, vel]))
        else:
            raise ValueError(method)

        y_true.append(future_global)
        y_pred.append(np.stack(trajectory))
        metas.append(sample)
    return np.stack(y_true), np.stack(y_pred), metas


def eval_baseline(
    ds: UAVMTRPaperDataset,
    method: str,
    norm: StateNormalizer,
    horizons: Sequence[int],
    dt: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true, y_pred, metas = baseline_predictions(ds, method, norm, dt)
    rows = []
    sample_rows = []
    for horizon in horizons:
        rows.append({
            "method": method,
            "horizon": horizon,
            **compute_metrics(y_true[:, :horizon], y_pred[:, :horizon]),
        })
        for i, sample in enumerate(metas):
            sample_rows.append({
                "method": method,
                "horizon": horizon,
                "scene_type": sample["scene_type"],
                "dominant_behavior": sample["dominant_behavior"],
                "chain_file": sample["chain_file"],
                **compute_metrics(
                    y_true[i:i + 1, :horizon],
                    y_pred[i:i + 1, :horizon],
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def group_sample_metrics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    metrics = [
        "pos_rmse", "pos_mae", "vel_rmse", "vel_mae",
        "all_rmse", "all_mae", "ade", "fde",
    ]
    rows = []
    for (method, horizon, value), group in df.groupby(
        ["method", "horizon", column], dropna=False
    ):
        row: Dict[str, object] = {
            "method": method,
            "horizon": horizon,
            column: value,
            "num_samples": len(group),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-faithful MTR adaptation for single-UAV trajectory prediction."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--input_len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--report_horizons", type=str, default="1,3,5,10")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.1)

    parser.add_argument("--num_queries", type=int, default=64)
    parser.add_argument("--num_output_modes", type=int, default=6)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--nms_dist_thresh", type=float, default=0.5)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_log_std", type=float, default=-5.0)
    parser.add_argument("--max_log_std", type=float, default=3.0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_milestones", type=str, default="20,22,24,26,28")
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--val_metric",
        type=str,
        default="pos_rmse",
        choices=["pos_rmse", "ade", "fde", "combo"],
    )
    parser.add_argument("--combo_ade_weight", type=float, default=0.5)
    parser.add_argument("--combo_fde_weight", type=float, default=0.25)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--vel_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1000.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_parallel", action="store_true")
    args = parser.parse_args()

    if args.num_output_modes > args.num_queries:
        raise ValueError("num_output_modes cannot exceed num_queries")

    set_seed(args.seed)
    root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "mtr_uav_v3_paper_faithful"
    ensure_dir(out_dir)

    report_horizons = [
        horizon for horizon in parse_int_list(args.report_horizons)
        if 1 <= horizon <= args.horizon
    ]
    if not report_horizons:
        raise ValueError("No valid report horizons")

    train_chains = load_split_chains(root, "train")
    val_chains = load_split_chains(root, "val")
    test_chains = load_split_chains(root, "test")
    norm = fit_normalizer(
        train_chains,
        input_len=args.input_len,
        horizon=args.horizon,
        stride=args.stride,
    )

    train_ds = UAVMTRPaperDataset(
        train_chains, args.input_len, args.horizon, args.stride, norm
    )
    val_ds = UAVMTRPaperDataset(
        val_chains, args.input_len, args.horizon, args.stride, norm
    )
    test_ds = UAVMTRPaperDataset(
        test_chains, args.input_len, args.horizon, args.stride, norm
    )

    endpoints_local = collect_train_endpoints_local(train_ds)
    intention_points_local = simple_kmeans(
        endpoints_local,
        k=args.num_queries,
        seed=args.seed,
        iterations=args.kmeans_iters,
    )
    intention_points_std = norm.pos_to_std(intention_points_local).astype(np.float32)
    np.save(out_dir / "intention_points_local.npy", intention_points_local)
    np.save(out_dir / "intention_points_std.npy", intention_points_std)

    print(f"samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"device: {DEVICE}, cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"num_queries={args.num_queries}, num_output_modes={args.num_output_modes}")
    print(f"intention_points_local range min={intention_points_local.min(axis=0)} max={intention_points_local.max(axis=0)}")

    model = MTRUAVPaperFaithful(
        intention_points_std=intention_points_std,
        input_len=args.input_len,
        horizon=args.horizon,
        state_dim=STATE_DIM,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
    ).to(DEVICE)

    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    def model_module() -> MTRUAVPaperFaithful:
        return model.module if isinstance(model, nn.DataParallel) else model

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    milestones = parse_int_list(args.lr_milestones)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_gamma
    )

    intention_points_local_t = torch.tensor(
        intention_points_local, dtype=torch.float32, device=DEVICE
    )
    best_score = float("inf")
    best_path = out_dir / "mtr_uav_paper_best.pt"
    train_log: List[Dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        part_sums = {
            "reg_nll": 0.0,
            "vel_l1": 0.0,
            "cls_ce": 0.0,
            "assigned_mode_acc": 0.0,
        }

        for hist, fut_std, fut_local, _fut_g, _origin, _c, _s, _meta in train_loader:
            hist = hist.to(DEVICE, non_blocking=True)
            fut_std = fut_std.to(DEVICE, non_blocking=True)
            fut_local = fut_local.to(DEVICE, non_blocking=True)

            outputs = model(hist)
            loss, parts = mtr_paper_loss(
                outputs,
                fut_std=fut_std,
                fut_local=fut_local,
                intention_points_local=intention_points_local_t,
                cls_weight=args.cls_weight,
                reg_weight=args.reg_weight,
                vel_weight=args.vel_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_n = hist.shape[0]
            total_samples += batch_n
            total_loss += float(loss.detach().cpu()) * batch_n
            for key, value in parts.items():
                part_sums[key] += value * batch_n

        # Top-1 validation is the deployable selection criterion.
        val_df, _, _ = evaluate(
            model_module(),
            val_ds,
            norm,
            batch_size=args.batch_size,
            horizons_to_report=[args.horizon],
            num_output_modes=args.num_output_modes,
            nms_dist_thresh=args.nms_dist_thresh,
            num_workers=args.num_workers,
        )
        val_top = val_df[val_df["method"] == "mtr_uav_paper_top1"].iloc[0]
        val_oracle = val_df[
            val_df["method"] == "mtr_uav_paper_oracle_minfde"
        ].iloc[0]
        val_score = score_from_metrics(
            val_top.to_dict(),
            metric=args.val_metric,
            ade_weight=args.combo_ade_weight,
            fde_weight=args.combo_fde_weight,
        )

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(total_samples, 1),
            "val_score": float(val_score),
            "val_pos_rmse": float(val_top["pos_rmse"]),
            "val_ade": float(val_top["ade"]),
            "val_fde": float(val_top["fde"]),
            "val_oracle_fde": float(val_oracle["fde"]),
            **{
                key: value / max(total_samples, 1)
                for key, value in part_sums.items()
            },
        }
        train_log.append(row)
        print(
            f"epoch={epoch:03d} lr={row['lr']:.2e} "
            f"loss={row['train_loss']:.6f} reg_nll={row['reg_nll']:.6f} "
            f"vel_l1={row['vel_l1']:.6f} cls_ce={row['cls_ce']:.6f} "
            f"assign_acc={row['assigned_mode_acc']:.4f} "
            f"val_pos_rmse={row['val_pos_rmse']:.6f} "
            f"val_ade={row['val_ade']:.6f} val_fde={row['val_fde']:.6f} "
            f"val_oracle_fde={row['val_oracle_fde']:.6f}"
        )

        if val_score < best_score:
            best_score = float(val_score)
            bad_epochs = 0
            torch.save(model_module().state_dict(), best_path)
        else:
            bad_epochs += 1

        scheduler.step()
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    if not best_path.exists():
        raise RuntimeError("Best checkpoint was not created")
    state = torch.load(best_path, map_location=DEVICE)
    model_module().load_state_dict(state)

    mtr_df, mtr_sample_df, mode_report = evaluate(
        model_module(),
        test_ds,
        norm,
        batch_size=args.batch_size,
        horizons_to_report=report_horizons,
        num_output_modes=args.num_output_modes,
        nms_dist_thresh=args.nms_dist_thresh,
        num_workers=args.num_workers,
        save_prediction_path=out_dir / "test_predictions_global.npz",
    )

    baseline_rows = []
    baseline_samples = []
    for method in ["constant_velocity", "constant_acceleration"]:
        frame, sample_frame = eval_baseline(
            test_ds, method, norm, report_horizons, args.dt
        )
        baseline_rows.append(frame)
        baseline_samples.append(sample_frame)

    compare_df = pd.concat([*baseline_rows, mtr_df], ignore_index=True)
    sample_df = pd.concat([*baseline_samples, mtr_sample_df], ignore_index=True)
    compare_df.to_csv(out_dir / "compare_horizon_metrics_test.csv", index=False)
    sample_df.to_csv(out_dir / "sample_metrics_test.csv", index=False)
    pd.DataFrame(train_log).to_csv(out_dir / "training_log.csv", index=False)
    group_sample_metrics(sample_df, "scene_type").to_csv(
        out_dir / "grouped_metrics_by_scene_test.csv", index=False
    )
    group_sample_metrics(sample_df, "dominant_behavior").to_csv(
        out_dir / "grouped_metrics_by_behavior_test.csv", index=False
    )

    num_parameters = int(sum(p.numel() for p in model_module().parameters()))
    report = {
        "version": "mtr_uav_lite_v3_paper_faithful",
        "data_root": str(root),
        "out_dir": str(out_dir),
        "input_len": args.input_len,
        "horizon": args.horizon,
        "report_horizons": report_horizons,
        "stride": args.stride,
        "dt": args.dt,
        "num_queries": args.num_queries,
        "num_output_modes": args.num_output_modes,
        "nms_dist_thresh_m": args.nms_dist_thresh,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_milestones": milestones,
        "lr_gamma": args.lr_gamma,
        "patience": args.patience,
        "loss_weights": {
            "classification": args.cls_weight,
            "gaussian_regression": args.reg_weight,
            "velocity": args.vel_weight,
        },
        "seed": args.seed,
        "num_parameters": num_parameters,
        "normalizer": norm.to_json(),
        "test_mode_report": mode_report,
        "formal_comparison_method": "mtr_uav_paper_top1",
        "oracle_upper_bound_method": "mtr_uav_paper_oracle_minfde",
        "notes": [
            "Static intention queries are sinusoidal embeddings of K-means endpoint priors.",
            "Dynamic searching queries are updated from each decoder layer's predicted endpoint.",
            "Every decoder layer has a probabilistic trajectory head and a classification head.",
            "Training uses nearest-intention hard assignment, diagonal 3-D Gaussian NLL, velocity L1, and classification CE.",
            "The original map and multi-agent modules are omitted because each chain contains one UAV and no synchronized map polyline tokens.",
            "The original 2-D correlated GMM is adapted to a diagonal 3-D Gaussian for NED position.",
            "Only mtr_uav_paper_top1 is deployable; oracle_minfde uses ground-truth future for candidate selection.",
        ],
    }
    with open(
        out_dir / "report_mtr_uav_lite_v3_paper_faithful.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n=== MTR-UAV V3 paper-faithful finished ===")
    print(compare_df.to_string(index=False))
    print(f"\nParameters: {num_parameters:,}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
