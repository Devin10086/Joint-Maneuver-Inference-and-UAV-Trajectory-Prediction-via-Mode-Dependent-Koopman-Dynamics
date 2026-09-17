#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pecnet_uav_lite_v2_paper_faithful.py

PECNet-UAV V2: paper-faithful single-UAV adaptation of PECNet
("It Is Not the Journey but the Destination", ECCV 2020).

Kept from the original PECNet design:
  1. MLP past-trajectory encoder.
  2. MLP destination encoder.
  3. CVAE posterior q(z | past, destination) during training.
  4. Standard Gaussian latent sampling at inference (configurable sigma).
  5. Endpoint decoder conditioned on past features and z.
  6. Trajectory predictor conditioned on past features and the generated endpoint.
  7. Predictor outputs intermediate future positions; generated endpoint is appended.
  8. MSE endpoint reconstruction + KL divergence + MSE intermediate-trajectory loss.
  9. Best-of-N oracle endpoint evaluation, reported separately from deployable prediction.

UAV-specific adaptations required for a fair comparison:
  - Single UAV, so no cross-agent non-local social pooling is used.
  - 3D endpoint and 3D trajectory positions replace pedestrian 2D positions.
  - Input can use either position-only history (pos3) or the common UAV state
    [x,y,z,vx,vy,vz] (state6). The formal fair-comparison runner uses state6.
  - The same local-frame normalization and train/val/test window protocol as the
    other UAV baselines is retained.
  - The model predicts positions as PECNet does; velocities are reconstructed by
    finite differences only for common-format reporting and are not training targets.

Data layout:
    DATA_ROOT/train/chains/*.csv
    DATA_ROOT/val/chains/*.csv
    DATA_ROOT/test/chains/*.csv
Required columns:
    x,y,z,vx,vy,vz

Primary deployable method:
    pecnet_uav_paper_z0
        Uses z=0, the center of the inference latent distribution. No test future is used.

Upper-bound method (not deployable):
    pecnet_uav_paper_oracle_minfde
        Samples N endpoints and uses the true future endpoint to choose the best sample.
"""

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
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
    metric: str,
    ade_weight: float,
    fde_weight: float,
) -> float:
    if metric == "pos_rmse":
        return float(metrics["pos_rmse"])
    if metric == "ade":
        return float(metrics["ade"])
    if metric == "fde":
        return float(metrics["fde"])
    if metric == "combo":
        return float(
            metrics["pos_rmse"]
            + ade_weight * metrics["ade"]
            + fde_weight * metrics["fde"]
        )
    raise ValueError(f"Unsupported metric: {metric}")


# =============================================================================
# Data and local frame
# =============================================================================
def load_split_chains(root: Path, split: str) -> List[pd.DataFrame]:
    chain_dir = root / split / "chains"
    files = sorted(chain_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {chain_dir}")

    chains: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        missing = [c for c in STATE_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{file_path} missing required columns: {missing}")
        if len(df) < 3:
            continue
        df = df.copy()
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
    x = np.asarray(states, dtype=np.float64)
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


def batch_local_to_global(
    local_states: np.ndarray,
    origins: np.ndarray,
    cs: np.ndarray,
    ss: np.ndarray,
) -> np.ndarray:
    local_states = np.asarray(local_states, dtype=np.float64)
    lx, ly, lz = local_states[..., 0], local_states[..., 1], local_states[..., 2]
    lvx, lvy, lvz = local_states[..., 3], local_states[..., 4], local_states[..., 5]

    c = np.asarray(cs, dtype=np.float64)[:, None]
    s = np.asarray(ss, dtype=np.float64)[:, None]
    origin = np.asarray(origins, dtype=np.float64)[:, None, :]

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
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float64)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) * self.std + self.mean).astype(np.float64)

    def inverse_positions(self, pos_std: np.ndarray) -> np.ndarray:
        return (
            np.asarray(pos_std, dtype=np.float64) * self.std[:3]
            + self.mean[:3]
        ).astype(np.float64)

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
        max_start = len(df) - input_len - horizon
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            hist = arr[start:start + input_len]
            future = arr[start + input_len:start + input_len + horizon]
            frame = make_local_frame(hist[-1], hist[-2] if input_len > 1 else None)
            states.append(global_to_local(hist, frame))
            states.append(global_to_local(future, frame))

    if not states:
        raise RuntimeError("No states collected for normalizer")

    all_states = np.concatenate(states, axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0)
    std[std < 1e-8] = 1.0
    return StateNormalizer(mean=mean, std=std)


class UAVPECNetPaperDataset(Dataset):
    def __init__(
        self,
        chains: Sequence[pd.DataFrame],
        input_len: int,
        horizon: int,
        stride: int,
        norm: StateNormalizer,
        past_feature_mode: str,
    ) -> None:
        if past_feature_mode not in {"state6", "pos3"}:
            raise ValueError(f"Unsupported past_feature_mode={past_feature_mode}")
        self.samples: List[Dict[str, object]] = []
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.norm = norm
        self.past_feature_mode = past_feature_mode

        for df in chains:
            arr = df[STATE_COLS].to_numpy(dtype=np.float64)
            max_start = len(df) - input_len - horizon
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, stride):
                hist_global = arr[start:start + input_len]
                future_global = arr[start + input_len:start + input_len + horizon]
                frame = make_local_frame(
                    hist_global[-1],
                    hist_global[-2] if input_len > 1 else None,
                )
                hist_local = global_to_local(hist_global, frame)
                future_local = global_to_local(future_global, frame)
                hist_std = norm.transform(hist_local)
                future_std = norm.transform(future_local)

                if past_feature_mode == "state6":
                    past_input = hist_std
                else:
                    past_input = hist_std[:, :3]

                # PECNet's initial_pos is the current point after shifting the first
                # observation to the origin. In the common last-point local frame,
                # use the equivalent observed span vector in standardized units.
                history_span_std = (
                    (hist_local[-1, :3] - hist_local[0, :3])
                    / norm.std[:3]
                )

                row = df.iloc[start + input_len - 1]
                if "behavior" in df.columns:
                    seq = [
                        str(x).strip().lower()
                        for x in df.iloc[start:start + input_len]["behavior"].tolist()
                    ]
                    dominant = pd.Series(seq).mode().iloc[0] if seq else "unknown"
                else:
                    dominant = "unknown"

                self.samples.append({
                    "past_input": past_input.astype(np.float32),
                    "hist_local": hist_local.astype(np.float32),
                    "future_pos_std": future_std[:, :3].astype(np.float32),
                    "endpoint_std": future_std[-1, :3].astype(np.float32),
                    "history_span_std": history_span_std.astype(np.float32),
                    "future_global": future_global.astype(np.float32),
                    "origin": frame.origin.astype(np.float32),
                    "c": np.float32(frame.c),
                    "s": np.float32(frame.s),
                    "scene_type": str(row.get("scene_type", "unknown")),
                    "dominant_behavior": str(dominant),
                    "chain_file": str(row.get("_chain_file", "")),
                })

        if not self.samples:
            raise RuntimeError(
                f"No samples created for input_len={input_len}, horizon={horizon}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return (
            torch.from_numpy(sample["past_input"]),
            torch.from_numpy(sample["future_pos_std"]),
            torch.from_numpy(sample["endpoint_std"]),
            torch.from_numpy(sample["history_span_std"]),
            torch.from_numpy(sample["hist_local"]),
            torch.from_numpy(sample["future_global"]),
            torch.from_numpy(sample["origin"]),
            torch.tensor(sample["c"], dtype=torch.float32),
            torch.tensor(sample["s"], dtype=torch.float32),
            {
                "scene_type": sample["scene_type"],
                "dominant_behavior": sample["dominant_behavior"],
                "chain_file": sample["chain_file"],
            },
        )


# =============================================================================
# Paper-faithful PECNet modules
# =============================================================================
class PaperMLP(nn.Module):
    """Original PECNet-style MLP: Linear + ReLU between layers, no LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float = -1.0,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(x) for x in hidden_sizes], int(output_dim)]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout >= 0.0:
                    p = min(0.1, self.dropout / 3.0) if i == 1 else self.dropout
                    x = F.dropout(x, p=p, training=self.training)
        return x


class PECNetUAVPaper(nn.Module):
    def __init__(
        self,
        past_input_dim: int,
        horizon: int,
        fdim: int,
        zdim: int,
        enc_past_sizes: Sequence[int],
        enc_dest_sizes: Sequence[int],
        enc_latent_sizes: Sequence[int],
        dec_sizes: Sequence[int],
        predictor_sizes: Sequence[int],
        dropout: float = -1.0,
    ) -> None:
        super().__init__()
        if horizon < 2:
            raise ValueError("PECNet predictor requires horizon >= 2")
        self.horizon = int(horizon)
        self.fdim = int(fdim)
        self.zdim = int(zdim)

        self.encoder_past = PaperMLP(
            input_dim=past_input_dim,
            output_dim=fdim,
            hidden_sizes=enc_past_sizes,
            dropout=dropout,
        )
        self.encoder_dest = PaperMLP(
            input_dim=POS_DIM,
            output_dim=fdim,
            hidden_sizes=enc_dest_sizes,
            dropout=dropout,
        )
        self.encoder_latent = PaperMLP(
            input_dim=2 * fdim,
            output_dim=2 * zdim,
            hidden_sizes=enc_latent_sizes,
            dropout=dropout,
        )
        self.decoder = PaperMLP(
            input_dim=fdim + zdim,
            output_dim=POS_DIM,
            hidden_sizes=dec_sizes,
            dropout=dropout,
        )
        self.predictor = PaperMLP(
            input_dim=2 * fdim + POS_DIM,
            output_dim=POS_DIM * (horizon - 1),
            hidden_sizes=predictor_sizes,
            dropout=dropout,
        )

    @staticmethod
    def _flatten_past(past_input: torch.Tensor) -> torch.Tensor:
        return past_input.reshape(past_input.shape[0], -1)

    def encode_past(self, past_input: torch.Tensor) -> torch.Tensor:
        return self.encoder_past(self._flatten_past(past_input))

    def posterior(
        self,
        past_features: torch.Tensor,
        endpoint_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dest_features = self.encoder_dest(endpoint_gt)
        latent = self.encoder_latent(torch.cat([past_features, dest_features], dim=-1))
        mu, logvar = latent.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-12.0, max=8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_endpoint(
        self,
        past_features: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(torch.cat([past_features, z], dim=-1))

    def predict_intermediate(
        self,
        past_features: torch.Tensor,
        generated_endpoint: torch.Tensor,
        history_span_std: torch.Tensor,
    ) -> torch.Tensor:
        dest_features = self.encoder_dest(generated_endpoint)
        features = torch.cat(
            [past_features, dest_features, history_span_std],
            dim=-1,
        )
        return self.predictor(features).reshape(-1, self.horizon - 1, POS_DIM)

    def trajectory_from_endpoint(
        self,
        past_input: torch.Tensor,
        generated_endpoint: torch.Tensor,
        history_span_std: torch.Tensor,
    ) -> torch.Tensor:
        past_features = self.encode_past(past_input)
        intermediate = self.predict_intermediate(
            past_features,
            generated_endpoint,
            history_span_std,
        )
        return torch.cat([intermediate, generated_endpoint[:, None, :]], dim=1)

    def forward(
        self,
        past_input: torch.Tensor,
        endpoint_gt: torch.Tensor,
        history_span_std: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_train(past_input, endpoint_gt, history_span_std)

    def forward_train(
        self,
        past_input: torch.Tensor,
        endpoint_gt: torch.Tensor,
        history_span_std: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        past_features = self.encode_past(past_input)
        mu, logvar = self.posterior(past_features, endpoint_gt)
        z = self.reparameterize(mu, logvar)
        generated_endpoint = self.decode_endpoint(past_features, z)
        intermediate = self.predict_intermediate(
            past_features,
            generated_endpoint,
            history_span_std,
        )
        trajectory = torch.cat(
            [intermediate, generated_endpoint[:, None, :]],
            dim=1,
        )
        return {
            "endpoint": generated_endpoint,
            "intermediate": intermediate,
            "trajectory": trajectory,
            "mu": mu,
            "logvar": logvar,
        }

    def endpoint_from_z(
        self,
        past_input: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        past_features = self.encode_past(past_input)
        return self.decode_endpoint(past_features, z)

    def z0_prediction(
        self,
        past_input: torch.Tensor,
        history_span_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_features = self.encode_past(past_input)
        z = torch.zeros(
            past_input.shape[0],
            self.zdim,
            dtype=past_input.dtype,
            device=past_input.device,
        )
        endpoint = self.decode_endpoint(past_features, z)
        intermediate = self.predict_intermediate(
            past_features,
            endpoint,
            history_span_std,
        )
        trajectory = torch.cat([intermediate, endpoint[:, None, :]], dim=1)
        return trajectory, endpoint

    def sample_endpoints(
        self,
        past_input: torch.Tensor,
        num_samples: int,
        sigma: float,
        truncation: float = 0.0,
    ) -> torch.Tensor:
        past_features = self.encode_past(past_input)
        batch_size = past_input.shape[0]
        z = torch.randn(
            batch_size,
            int(num_samples),
            self.zdim,
            dtype=past_input.dtype,
            device=past_input.device,
        ) * float(sigma)
        if truncation > 0.0:
            bound = float(truncation) * float(sigma)
            z = torch.clamp(z, min=-bound, max=bound)
        past_rep = past_features[:, None, :].expand(
            batch_size, int(num_samples), self.fdim
        )
        endpoint = self.decoder(
            torch.cat([past_rep, z], dim=-1).reshape(-1, self.fdim + self.zdim)
        )
        return endpoint.reshape(batch_size, int(num_samples), POS_DIM)


# =============================================================================
# Loss and physical reconstruction
# =============================================================================
def pecnet_paper_loss(
    output: Dict[str, torch.Tensor],
    future_pos_std: torch.Tensor,
    endpoint_gt: torch.Tensor,
    kld_reg: float,
    adl_reg: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    endpoint_recon = F.mse_loss(output["endpoint"], endpoint_gt)
    intermediate_gt = future_pos_std[:, :-1, :]
    trajectory_loss = F.mse_loss(output["intermediate"], intermediate_gt)
    mu, logvar = output["mu"], output["logvar"]
    kld = -0.5 * torch.mean(
        torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )
    total = endpoint_recon + float(kld_reg) * kld + float(adl_reg) * trajectory_loss
    return total, {
        "endpoint_reconstruction": float(endpoint_recon.detach().cpu()),
        "kld": float(kld.detach().cpu()),
        "intermediate_trajectory": float(trajectory_loss.detach().cpu()),
    }


def positions_std_to_global_state(
    pos_std: np.ndarray,
    hist_local: np.ndarray,
    origins: np.ndarray,
    cs: np.ndarray,
    ss: np.ndarray,
    norm: StateNormalizer,
    dt: float,
) -> np.ndarray:
    pos_local = norm.inverse_positions(pos_std)
    last_pos = np.asarray(hist_local, dtype=np.float64)[:, -1, :3]
    vel_local = np.zeros_like(pos_local)
    vel_local[:, 0, :] = (pos_local[:, 0, :] - last_pos) / max(float(dt), 1e-8)
    if pos_local.shape[1] > 1:
        vel_local[:, 1:, :] = np.diff(pos_local, axis=1) / max(float(dt), 1e-8)
    local_state = np.concatenate([pos_local, vel_local], axis=-1)
    return batch_local_to_global(local_state, origins, cs, ss)


def endpoints_std_to_global(
    endpoint_std: np.ndarray,
    origins: np.ndarray,
    cs: np.ndarray,
    ss: np.ndarray,
    norm: StateNormalizer,
) -> np.ndarray:
    endpoint_local = norm.inverse_positions(endpoint_std)
    batch_size = endpoint_local.shape[0]
    samples = endpoint_local.shape[1]
    flat_local = endpoint_local.reshape(batch_size * samples, 1, POS_DIM)
    local_state = np.zeros((batch_size * samples, 1, STATE_DIM), dtype=np.float64)
    local_state[..., :3] = flat_local
    rep_origins = np.repeat(origins, samples, axis=0)
    rep_cs = np.repeat(cs, samples, axis=0)
    rep_ss = np.repeat(ss, samples, axis=0)
    global_state = batch_local_to_global(local_state, rep_origins, rep_cs, rep_ss)
    return global_state[:, 0, :3].reshape(batch_size, samples, POS_DIM)


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(
    model: PECNetUAVPaper,
    dataset: UAVPECNetPaperDataset,
    norm: StateNormalizer,
    batch_size: int,
    horizons_to_report: Sequence[int],
    num_samples: int,
    sigma: float,
    truncation: float,
    dt: float,
    method_prefix: str = "pecnet_uav_paper",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    y_true_all: List[np.ndarray] = []
    y_z0_all: List[np.ndarray] = []
    y_oracle_all: List[np.ndarray] = []
    meta_rows: List[Dict[str, str]] = []

    with torch.no_grad():
        for (
            past_input,
            _future_pos_std,
            _endpoint_gt,
            history_span_std,
            hist_local,
            future_global,
            origin,
            c,
            s,
            meta,
        ) in loader:
            past_input = past_input.to(DEVICE, non_blocking=True)
            history_span_std = history_span_std.to(DEVICE, non_blocking=True)

            z0_traj_std, _ = model.z0_prediction(past_input, history_span_std)
            endpoint_samples_std = model.sample_endpoints(
                past_input,
                num_samples=num_samples,
                sigma=sigma,
                truncation=truncation,
            )

            origin_np = origin.numpy().astype(np.float64)
            c_np = c.numpy().astype(np.float64)
            s_np = s.numpy().astype(np.float64)
            hist_local_np = hist_local.numpy().astype(np.float64)
            true_global = future_global.numpy().astype(np.float64)

            z0_global = positions_std_to_global_state(
                z0_traj_std.cpu().numpy(),
                hist_local_np,
                origin_np,
                c_np,
                s_np,
                norm,
                dt,
            )

            endpoint_samples_np = endpoint_samples_std.cpu().numpy()
            endpoint_samples_global = endpoints_std_to_global(
                endpoint_samples_np,
                origin_np,
                c_np,
                s_np,
                norm,
            )
            endpoint_error = np.linalg.norm(
                endpoint_samples_global - true_global[:, None, -1, :3],
                axis=-1,
            )
            best_idx = endpoint_error.argmin(axis=1)
            best_endpoint_std = endpoint_samples_std[
                torch.arange(endpoint_samples_std.shape[0], device=DEVICE),
                torch.from_numpy(best_idx).to(DEVICE),
            ]
            oracle_traj_std = model.trajectory_from_endpoint(
                past_input,
                best_endpoint_std,
                history_span_std,
            )
            oracle_global = positions_std_to_global_state(
                oracle_traj_std.cpu().numpy(),
                hist_local_np,
                origin_np,
                c_np,
                s_np,
                norm,
                dt,
            )

            y_true_all.append(true_global)
            y_z0_all.append(z0_global)
            y_oracle_all.append(oracle_global)

            for i in range(true_global.shape[0]):
                meta_rows.append({
                    "scene_type": meta["scene_type"][i],
                    "dominant_behavior": meta["dominant_behavior"][i],
                    "chain_file": meta["chain_file"][i],
                })

    y_true = np.concatenate(y_true_all, axis=0)
    y_z0 = np.concatenate(y_z0_all, axis=0)
    y_oracle = np.concatenate(y_oracle_all, axis=0)

    compare_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    method_predictions = {
        f"{method_prefix}_z0": y_z0,
        f"{method_prefix}_oracle_minfde": y_oracle,
    }

    for horizon in horizons_to_report:
        for method, prediction in method_predictions.items():
            metrics = compute_metrics(y_true[:, :horizon], prediction[:, :horizon])
            compare_rows.append({"method": method, "horizon": int(horizon), **metrics})

        # Per-sample output only for the deployable z=0 method.
        for i in range(y_true.shape[0]):
            metrics = compute_metrics(
                y_true[i:i + 1, :horizon],
                y_z0[i:i + 1, :horizon],
            )
            sample_rows.append({
                "method": f"{method_prefix}_z0",
                "horizon": int(horizon),
                **meta_rows[i],
                **metrics,
            })

    arrays = {"y_true": y_true, "y_z0": y_z0, "y_oracle": y_oracle}
    return pd.DataFrame(compare_rows), pd.DataFrame(sample_rows), arrays


def baseline_predictions(
    dataset: UAVPECNetPaperDataset,
    method: str,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    metas: List[Dict[str, object]] = []

    for sample in dataset.samples:
        future_global = np.asarray(sample["future_global"], dtype=np.float64)
        hist_local = np.asarray(sample["hist_local"], dtype=np.float64)
        origins = np.asarray(sample["origin"], dtype=np.float64)[None]
        cs = np.asarray([sample["c"]], dtype=np.float64)
        ss = np.asarray([sample["s"]], dtype=np.float64)
        hist_global = batch_local_to_global(hist_local[None], origins, cs, ss)[0]

        last = hist_global[-1]
        prev = hist_global[-2] if len(hist_global) > 1 else last
        pos = last[:3].copy()
        vel = last[3:].copy()
        pred: List[np.ndarray] = []

        if method == "constant_velocity":
            for _ in range(dataset.horizon):
                pos = pos + float(dt) * vel
                pred.append(np.concatenate([pos, vel]))
        elif method == "constant_acceleration":
            acc = (last[3:] - prev[3:]) / max(float(dt), 1e-8)
            for _ in range(dataset.horizon):
                pos = pos + float(dt) * vel + 0.5 * float(dt) ** 2 * acc
                vel = vel + float(dt) * acc
                pred.append(np.concatenate([pos, vel]))
        else:
            raise ValueError(method)

        y_true.append(future_global)
        y_pred.append(np.stack(pred))
        metas.append(sample)

    return np.stack(y_true), np.stack(y_pred), metas


def eval_baseline(
    dataset: UAVPECNetPaperDataset,
    method: str,
    horizons: Sequence[int],
    dt: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true, y_pred, metas = baseline_predictions(dataset, method, dt)
    rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for horizon in horizons:
        metrics = compute_metrics(y_true[:, :horizon], y_pred[:, :horizon])
        rows.append({"method": method, "horizon": int(horizon), **metrics})
        for i, sample in enumerate(metas):
            im = compute_metrics(
                y_true[i:i + 1, :horizon],
                y_pred[i:i + 1, :horizon],
            )
            sample_rows.append({
                "method": method,
                "horizon": int(horizon),
                "scene_type": sample["scene_type"],
                "dominant_behavior": sample["dominant_behavior"],
                "chain_file": sample["chain_file"],
                **im,
            })
    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def group_sample_metrics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    metric_cols = [
        "pos_rmse",
        "pos_mae",
        "vel_rmse",
        "vel_mae",
        "all_rmse",
        "all_mae",
        "ade",
        "fde",
    ]
    rows: List[Dict[str, object]] = []
    for (method, horizon, value), group in df.groupby(
        ["method", "horizon", column], dropna=False
    ):
        row: Dict[str, object] = {
            "method": method,
            "horizon": horizon,
            column: value,
            "num_samples": len(group),
        }
        for metric in metric_cols:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-faithful single-UAV PECNet adaptation"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--input_len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--report_horizons", default="1,3,5,10")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.1)

    parser.add_argument(
        "--past_feature_mode",
        choices=["state6", "pos3"],
        default="state6",
        help="state6 is the fair UAV comparison setting; pos3 is closest to original PECNet input.",
    )
    parser.add_argument("--fdim", type=int, default=16)
    parser.add_argument("--zdim", type=int, default=16)
    parser.add_argument("--enc_past_sizes", default="512,256")
    parser.add_argument("--enc_dest_sizes", default="8,16")
    parser.add_argument("--enc_latent_sizes", default="8,50")
    parser.add_argument("--dec_sizes", default="1024,512,1024")
    parser.add_argument("--predictor_sizes", default="1024,512,256")
    parser.add_argument("--dropout", type=float, default=-1.0)

    parser.add_argument("--epochs", type=int, default=650)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--val_metric",
        choices=["pos_rmse", "ade", "fde", "combo"],
        default="pos_rmse",
    )
    parser.add_argument("--combo_ade_weight", type=float, default=0.5)
    parser.add_argument("--combo_fde_weight", type=float, default=0.25)
    parser.add_argument("--kld_reg", type=float, default=1.0)
    parser.add_argument("--adl_reg", type=float, default=1.0)

    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=1.3)
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.0,
        help="0 disables clipping, matching the public PECNet code; positive value clips at truncation*sigma.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_parallel", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(args.data_root)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else root / "pecnet_uav_lite_v2_paper_faithful"
    )
    ensure_dir(out_dir)

    report_horizons = [
        h for h in parse_int_list(args.report_horizons) if h <= args.horizon
    ]
    if not report_horizons:
        raise ValueError("No valid report horizons")

    train_chains = load_split_chains(root, "train")
    val_chains = load_split_chains(root, "val")
    test_chains = load_split_chains(root, "test")
    norm = fit_normalizer(
        train_chains,
        args.input_len,
        args.horizon,
        args.stride,
    )

    train_ds = UAVPECNetPaperDataset(
        train_chains,
        args.input_len,
        args.horizon,
        args.stride,
        norm,
        args.past_feature_mode,
    )
    val_ds = UAVPECNetPaperDataset(
        val_chains,
        args.input_len,
        args.horizon,
        args.stride,
        norm,
        args.past_feature_mode,
    )
    test_ds = UAVPECNetPaperDataset(
        test_chains,
        args.input_len,
        args.horizon,
        args.stride,
        norm,
        args.past_feature_mode,
    )

    feature_dim = STATE_DIM if args.past_feature_mode == "state6" else POS_DIM
    model = PECNetUAVPaper(
        past_input_dim=args.input_len * feature_dim,
        horizon=args.horizon,
        fdim=args.fdim,
        zdim=args.zdim,
        enc_past_sizes=parse_int_list(args.enc_past_sizes),
        enc_dest_sizes=parse_int_list(args.enc_dest_sizes),
        enc_latent_sizes=parse_int_list(args.enc_latent_sizes),
        dec_sizes=parse_int_list(args.dec_sizes),
        predictor_sizes=parse_int_list(args.predictor_sizes),
        dropout=args.dropout,
    ).to(DEVICE)

    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    def module() -> PECNetUAVPaper:
        return model.module if isinstance(model, nn.DataParallel) else model

    print(
        f"samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )
    print(f"device={DEVICE}, cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"past_feature_mode={args.past_feature_mode}")
    print(
        "IMPORTANT: pecnet_uav_paper_z0 is deployable; "
        "pecnet_uav_paper_oracle_minfde uses test future endpoint and is an upper bound."
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = float("inf")
    best_path = out_dir / "pecnet_uav_paper_best.pt"
    training_log: List[Dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        part_sums = {
            "endpoint_reconstruction": 0.0,
            "kld": 0.0,
            "intermediate_trajectory": 0.0,
        }

        for (
            past_input,
            future_pos_std,
            endpoint_gt,
            history_span_std,
            _hist_local,
            _future_global,
            _origin,
            _c,
            _s,
            _meta,
        ) in train_loader:
            past_input = past_input.to(DEVICE, non_blocking=True)
            future_pos_std = future_pos_std.to(DEVICE, non_blocking=True)
            endpoint_gt = endpoint_gt.to(DEVICE, non_blocking=True)
            history_span_std = history_span_std.to(DEVICE, non_blocking=True)

            output = model(
                past_input,
                endpoint_gt,
                history_span_std,
            )

            loss, parts = pecnet_paper_loss(
                output,
                future_pos_std,
                endpoint_gt,
                args.kld_reg,
                args.adl_reg,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            batch = past_input.shape[0]
            total_loss += float(loss.detach().cpu()) * batch
            total_count += batch
            for key, value in parts.items():
                part_sums[key] += float(value) * batch

        val_df, _, _ = evaluate(
            module(),
            val_ds,
            norm,
            args.batch_size,
            [args.horizon],
            num_samples=max(1, min(args.num_samples, 8)),
            sigma=args.sigma,
            truncation=args.truncation,
            dt=args.dt,
        )
        val_row = val_df[
            val_df["method"] == "pecnet_uav_paper_z0"
        ].iloc[0].to_dict()
        val_score = score_from_metrics(
            val_row,
            args.val_metric,
            args.combo_ade_weight,
            args.combo_fde_weight,
        )

        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val_score": float(val_score),
            "val_pos_rmse": float(val_row["pos_rmse"]),
            "val_ade": float(val_row["ade"]),
            "val_fde": float(val_row["fde"]),
            **{
                key: value / max(total_count, 1)
                for key, value in part_sums.items()
            },
        }
        training_log.append(record)
        print(
            f"epoch={epoch:03d} loss={record['train_loss']:.6f} "
            f"RCL={record['endpoint_reconstruction']:.6f} "
            f"KLD={record['kld']:.6f} ADL={record['intermediate_trajectory']:.6f} "
            f"val_pos_rmse={record['val_pos_rmse']:.6f} "
            f"val_ADE={record['val_ade']:.6f} val_FDE={record['val_fde']:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            bad_epochs = 0
            torch.save(module().state_dict(), best_path)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    state = torch.load(best_path, map_location=DEVICE)
    module().load_state_dict(state)

    pec_df, pec_sample_df, arrays = evaluate(
        module(),
        test_ds,
        norm,
        args.batch_size,
        report_horizons,
        num_samples=args.num_samples,
        sigma=args.sigma,
        truncation=args.truncation,
        dt=args.dt,
    )

    baseline_rows: List[pd.DataFrame] = []
    baseline_samples: List[pd.DataFrame] = []
    for baseline in ["constant_velocity", "constant_acceleration"]:
        result_df, sample_df = eval_baseline(
            test_ds,
            baseline,
            report_horizons,
            args.dt,
        )
        baseline_rows.append(result_df)
        baseline_samples.append(sample_df)

    compare_df = pd.concat([*baseline_rows, pec_df], ignore_index=True)
    sample_df = pd.concat([*baseline_samples, pec_sample_df], ignore_index=True)

    compare_df.to_csv(out_dir / "compare_horizon_metrics_test.csv", index=False)
    sample_df.to_csv(out_dir / "sample_metrics_test.csv", index=False)
    pd.DataFrame(training_log).to_csv(out_dir / "training_log.csv", index=False)
    group_sample_metrics(sample_df, "scene_type").to_csv(
        out_dir / "grouped_metrics_by_scene_test.csv", index=False
    )
    group_sample_metrics(sample_df, "dominant_behavior").to_csv(
        out_dir / "grouped_metrics_by_behavior_test.csv", index=False
    )
    np.savez_compressed(
        out_dir / "test_predictions_global.npz",
        y_true=arrays["y_true"],
        pecnet_z0=arrays["y_z0"],
        pecnet_oracle_minfde=arrays["y_oracle"],
    )

    report = {
        "version": "pecnet_uav_lite_v2_paper_faithful",
        "data_root": str(root),
        "out_dir": str(out_dir),
        "input_len": args.input_len,
        "horizon": args.horizon,
        "report_horizons": report_horizons,
        "stride": args.stride,
        "dt": args.dt,
        "past_feature_mode": args.past_feature_mode,
        "fdim": args.fdim,
        "zdim": args.zdim,
        "enc_past_sizes": parse_int_list(args.enc_past_sizes),
        "enc_dest_sizes": parse_int_list(args.enc_dest_sizes),
        "enc_latent_sizes": parse_int_list(args.enc_latent_sizes),
        "dec_sizes": parse_int_list(args.dec_sizes),
        "predictor_sizes": parse_int_list(args.predictor_sizes),
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "kld_reg": args.kld_reg,
        "adl_reg": args.adl_reg,
        "num_samples": args.num_samples,
        "sigma": args.sigma,
        "truncation": args.truncation,
        "seed": args.seed,
        "normalizer": norm.to_json(),
        "primary_method": "pecnet_uav_paper_z0",
        "upper_bound_method": "pecnet_uav_paper_oracle_minfde",
        "notes": [
            "The model follows PECNet's posterior-CVAE endpoint generator and endpoint-conditioned MLP trajectory predictor.",
            "There is no learned conditional prior; inference samples z from a zero-mean Gaussian, as in the public PECNet implementation.",
            "The generated endpoint, not the ground-truth endpoint, conditions the training trajectory predictor.",
            "No social pooling is used because each chain contains one UAV and no synchronized neighboring agents.",
            "The z=0 result is the deployable single-prediction baseline used in the main comparison.",
            "Oracle-minFDE uses the test future endpoint for candidate selection and is only a multimodal upper bound.",
            "Velocities are reconstructed from predicted positions for common-format reporting; the PECNet loss supervises positions only.",
        ],
    }
    with open(
        out_dir / "report_pecnet_uav_lite_v2_paper_faithful.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n=== PECNet-UAV V2 Paper-Faithful finished ===")
    print(compare_df.to_string(index=False))
    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
