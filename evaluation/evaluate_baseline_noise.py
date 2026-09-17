#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified clean-trained -> noisy-tested evaluator for the paper-faithful UAV baselines:
  - FlightBERT++ regression adaptation
  - PECNet z=0 deployable prediction (+ optional oracle)
  - MTR top-1 deployable prediction (+ optional oracle)

Fairness with evaluate_topokoop_noise_robustness.py
---------------------------------------------------
1. Test windows are loaded through the SAME TopoKoop source module and make_split_data().
2. The sample selection routine and NumPy RNG convention match the TopoKoop evaluator.
3. For each (sigma, noise_seed), all methods receive the same global-frame noise tensor.
4. Noise is added in physical units before each baseline's local-frame transform/normalizer.
5. Only history is noised; future ground truth remains clean.
6. Model checkpoints and training normalizers are loaded from each seed's report/checkpoint.

Typical use:
python -u evaluate_paper_faithful_baselines_noise_robustness.py \
  --model flightbertpp \
  --model_source_py flightbertpp_uav_regression_v3_paper_faithful.py \
  --topokoop_source_py topokoop_e6_cab_future_moe_v4_swapyz_no_global_actions_realtime_export.py \
  --checkpoints_root DATA/topokoop_v4_320_label_free/baselines/flightbertpp_v3_paper_faithful_multiseed \
  --data_root DATA/topokoop_v4_320_label_free/processed \
  --out_dir DATA/topokoop_v4_320_label_free/baselines/noise_robustness/flightbertpp \
  --noise_mode position_only \
  --position_sigmas 0,0.005,0.01,0.02,0.03,0.05,0.1 \
  --velocity_sigmas 0 \
  --noise_seeds 42,43,44,45,46 \
  --max_samples 0 \
  --sample_mode evenly \
  --report_horizons 1,3,5,10 \
  --device cuda:0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("float list is empty")
    return vals


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("int list is empty")
    return vals


def load_module(path: Path, prefix: str):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"{prefix}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def choose_sample_indices(total: int, max_samples: int, mode: str, seed: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("empty test set")
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    if mode == "first":
        return np.arange(max_samples, dtype=np.int64)
    if mode == "evenly":
        return np.linspace(0, total - 1, num=max_samples, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(total, size=max_samples, replace=False)).astype(np.int64)
    raise ValueError(mode)


def build_noise_levels(mode: str, ps: Sequence[float], vs: Sequence[float]):
    if mode == "joint":
        if len(ps) != len(vs):
            raise ValueError("joint mode requires equal-length position/velocity sigma lists")
        pairs = list(zip(ps, vs))
    elif mode == "position_only":
        pairs = [(p, 0.0) for p in ps]
    elif mode == "velocity_only":
        pairs = [(0.0, v) for v in vs]
    else:
        raise ValueError(mode)
    out = []
    for i, (p, v) in enumerate(pairs):
        if p < 0 or v < 0:
            raise ValueError("sigmas must be nonnegative")
        out.append((f"N{i}", float(p), float(v)))
    return out


def trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)[:, :horizon]
    yp = np.asarray(y_pred, dtype=np.float64)[:, :horizon]
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    err = yp - yt
    pe = err[..., :3]
    disp = np.linalg.norm(pe, axis=-1)
    return {
        "pos_rmse": float(np.sqrt(np.mean(pe ** 2))),
        "pos_mae": float(np.mean(np.abs(pe))),
        "ade": float(np.mean(disp)),
        "fde": float(np.mean(disp[:, -1])),
        "ade_xy": float(np.mean(np.linalg.norm(pe[..., :2], axis=-1))),
        "fde_xy": float(np.mean(np.linalg.norm(pe[:, -1, :2], axis=-1))),
        "ade_z": float(np.mean(np.abs(pe[..., 2]))),
        "fde_z": float(np.mean(np.abs(pe[:, -1, 2]))),
    }


def per_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int):
    err = np.asarray(y_pred, dtype=np.float64)[:, :horizon, :3] - np.asarray(y_true, dtype=np.float64)[:, :horizon, :3]
    d = np.linalg.norm(err, axis=-1)
    return (
        np.sqrt(np.mean(err ** 2, axis=(1, 2))),
        np.mean(d, axis=1),
        d[:, -1],
    )


def summary_mean_std(df: pd.DataFrame, group_cols: Sequence[str], exclude: Sequence[str] = ()):
    excluded = set(group_cols) | set(exclude)
    nums = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    g = df.groupby(list(group_cols), dropna=False)[nums].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def load_reference_windows(topokoop_source_py: Path, data_root: Path, input_len: int, horizon: int, stride: int):
    source = load_module(topokoop_source_py, "reference_topokoop_source")
    bk = source.behavior_mode_endpoint_koopman_v2_hmm_selector
    chains = bk.load_split_chains(data_root, "test")
    data = bk.make_split_data(chains, input_len, horizon, stride)
    hist = np.asarray(data.hist_global, dtype=np.float64)
    fut = np.asarray(data.future_global, dtype=np.float64)
    return hist, fut


def build_window_metadata(data_root: Path, input_len: int, horizon: int, stride: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    files = sorted((data_root / "test" / "chains").glob("*.csv"))
    for f in files:
        df = pd.read_csv(f)
        n = len(df)
        max_start = n - input_len - horizon
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            anchor = df.iloc[start + input_len - 1]
            rows.append({
                "chain_file": f.name,
                "window_start": int(start),
                "sample_id": f"{f.name}::start={start}",
                "scene_type": str(anchor.get("scene_type", "unknown")),
                "mission_intent": str(anchor.get("mission_intent", "unknown")),
                "region_name": str(anchor.get("region_name", "unknown")),
            })
    return pd.DataFrame(rows)


def frames_from_histories(src, histories: np.ndarray):
    origins, cs, ss, locals_ = [], [], [], []
    for h in histories:
        frame = src.make_local_frame(h[-1], h[-2] if len(h) > 1 else None)
        origins.append(np.asarray(frame.origin, dtype=np.float64))
        cs.append(float(frame.c))
        ss.append(float(frame.s))
        locals_.append(src.global_to_local(h, frame))
    return (
        np.stack(locals_, axis=0),
        np.stack(origins, axis=0),
        np.asarray(cs, dtype=np.float64),
        np.asarray(ss, dtype=np.float64),
    )


@dataclass
class LoadedModel:
    train_seed: str
    report: Dict[str, Any]
    norm: Any
    model: torch.nn.Module
    intention_points_std: Optional[np.ndarray] = None


def discover_seed_dirs(root: Path) -> List[Path]:
    dirs = [p for p in sorted(root.glob("seed*")) if p.is_dir()]
    if not dirs:
        raise RuntimeError(f"No seed* directories in {root}")
    return dirs


def model_files(model_name: str):
    if model_name == "flightbertpp":
        return "flightbertpp_uav_regression_best.pt", "report_flightbertpp_uav_regression_v2.json"
    if model_name == "pecnet":
        return "pecnet_uav_paper_best.pt", "report_pecnet_uav_lite_v2_paper_faithful.json"
    if model_name == "mtr":
        return "mtr_uav_paper_best.pt", "report_mtr_uav_lite_v3_paper_faithful.json"
    raise ValueError(model_name)


def load_seed_model(model_name: str, src, seed_dir: Path, device: torch.device) -> LoadedModel:
    ckpt_name, report_name = model_files(model_name)
    ckpt = seed_dir / ckpt_name
    report_path = seed_dir / report_name
    if not ckpt.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"missing checkpoint/report in {seed_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    n = report["normalizer"]
    norm = src.StateNormalizer(
        mean=np.asarray(n["mean"], dtype=np.float64),
        std=np.asarray(n["std"], dtype=np.float64),
    )

    if model_name == "flightbertpp":
        model = src.FlightBERTPPUAVRegressor(
            input_len=int(report["input_len"]),
            horizon=int(report["horizon"]),
            state_dim=6,
            d_model=int(report["d_model"]),
            n_heads=int(report["n_heads"]),
            n_layers_enc=int(report["n_layers_enc"]),
            n_layers_dec=int(report["n_layers_dec"]),
            dropout=float(report["dropout"]),
            embedding_type=str(report.get("embedding_type", "conv")),
            zero_head_init=bool(report.get("zero_head_init", False)),
        )
        intention = None
    elif model_name == "pecnet":
        mode = str(report["past_feature_mode"])
        per_step = 6 if mode == "state6" else 3
        model = src.PECNetUAVPaper(
            past_input_dim=int(report["input_len"]) * per_step,
            horizon=int(report["horizon"]),
            fdim=int(report["fdim"]),
            zdim=int(report["zdim"]),
            enc_past_sizes=list(map(int, report["enc_past_sizes"])),
            enc_dest_sizes=list(map(int, report["enc_dest_sizes"])),
            enc_latent_sizes=list(map(int, report["enc_latent_sizes"])),
            dec_sizes=list(map(int, report["dec_sizes"])),
            predictor_sizes=list(map(int, report["predictor_sizes"])),
            dropout=float(report["dropout"]),
        )
        intention = None
    elif model_name == "mtr":
        intention_path = seed_dir / "intention_points_std.npy"
        if not intention_path.is_file():
            raise FileNotFoundError(intention_path)
        intention = np.load(intention_path).astype(np.float32)
        model = src.MTRUAVPaperFaithful(
            intention_points_std=intention,
            input_len=int(report["input_len"]),
            horizon=int(report["horizon"]),
            state_dim=6,
            d_model=int(report["d_model"]),
            n_heads=int(report["n_heads"]),
            num_encoder_layers=int(report["num_encoder_layers"]),
            num_decoder_layers=int(report["num_decoder_layers"]),
            dropout=float(report["dropout"]),
            min_log_std=float(report.get("min_log_std", -5.0)),
            max_log_std=float(report.get("max_log_std", 3.0)),
        )
    else:
        raise ValueError(model_name)

    state = torch.load(ckpt, map_location=device)
    # Defensive compatibility with checkpoints accidentally saved with DataParallel prefix.
    if isinstance(state, dict) and state and all(str(k).startswith("module.") for k in state.keys()):
        state = {str(k)[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return LoadedModel(seed_dir.name, report, norm, model, intention)


def infer_flight(src, loaded: LoadedModel, histories: np.ndarray, device: torch.device, batch_size: int):
    hist_local, origins, cs, ss = frames_from_histories(src, histories)
    hist_std = loaded.norm.transform(hist_local).astype(np.float32)
    hist_diff = (hist_std[:, 1:] - hist_std[:, :-1]).astype(np.float32)
    preds = []
    with torch.no_grad():
        for a in range(0, len(histories), batch_size):
            b = min(a + batch_size, len(histories))
            h = torch.from_numpy(hist_std[a:b]).to(device)
            d = torch.from_numpy(hist_diff[a:b]).to(device)
            pred_std, _ = loaded.model(h, d)
            preds.append(pred_std.detach().cpu().numpy())
    pred_std = np.concatenate(preds, axis=0)
    pred_local = loaded.norm.inverse(pred_std)
    return {"flightbertpp_uav_regression": src.batch_local_to_global(pred_local, origins, cs, ss)}


def infer_pecnet(src, loaded: LoadedModel, histories: np.ndarray, future: np.ndarray, device: torch.device, batch_size: int, include_oracle: bool, inference_seed: int):
    hist_local, origins, cs, ss = frames_from_histories(src, histories)
    hist_std = loaded.norm.transform(hist_local).astype(np.float32)
    mode = str(loaded.report["past_feature_mode"])
    past = hist_std if mode == "state6" else hist_std[..., :3]
    span = ((hist_local[:, -1, :3] - hist_local[:, 0, :3]) / loaded.norm.std[:3]).astype(np.float32)
    z0_list, oracle_list = [], []
    torch.manual_seed(int(inference_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(inference_seed))
    with torch.no_grad():
        for a in range(0, len(histories), batch_size):
            b = min(a + batch_size, len(histories))
            p = torch.from_numpy(past[a:b]).to(device)
            sp = torch.from_numpy(span[a:b]).to(device)
            z0_std, _ = loaded.model.z0_prediction(p, sp)
            z0_np = z0_std.detach().cpu().numpy()
            z0_g = src.positions_std_to_global_state(
                z0_np, hist_local[a:b], origins[a:b], cs[a:b], ss[a:b], loaded.norm, float(loaded.report["dt"])
            )
            z0_list.append(z0_g)
            if include_oracle:
                samples = loaded.model.sample_endpoints(
                    p,
                    num_samples=int(loaded.report.get("num_samples", 20)),
                    sigma=float(loaded.report.get("sigma", 1.3)),
                    truncation=float(loaded.report.get("truncation", 0.0)),
                )
                samples_np = samples.detach().cpu().numpy()
                endpoints_g = src.endpoints_std_to_global(samples_np, origins[a:b], cs[a:b], ss[a:b], loaded.norm)
                errs = np.linalg.norm(endpoints_g - future[a:b, None, -1, :3], axis=-1)
                best = errs.argmin(axis=1)
                chosen = samples[torch.arange(b - a, device=device), torch.from_numpy(best).to(device)]
                traj = loaded.model.trajectory_from_endpoint(p, chosen, sp)
                traj_g = src.positions_std_to_global_state(
                    traj.detach().cpu().numpy(), hist_local[a:b], origins[a:b], cs[a:b], ss[a:b], loaded.norm, float(loaded.report["dt"])
                )
                oracle_list.append(traj_g)
    out = {"pecnet_uav_paper_z0": np.concatenate(z0_list, axis=0)}
    if include_oracle:
        out["pecnet_uav_paper_oracle_minfde"] = np.concatenate(oracle_list, axis=0)
    return out


def infer_mtr(src, loaded: LoadedModel, histories: np.ndarray, future: np.ndarray, device: torch.device, batch_size: int, include_oracle: bool):
    hist_local, origins, cs, ss = frames_from_histories(src, histories)
    hist_std = loaded.norm.transform(hist_local).astype(np.float32)
    top_list, oracle_list = [], []
    with torch.no_grad():
        for a in range(0, len(histories), batch_size):
            b = min(a + batch_size, len(histories))
            h = torch.from_numpy(hist_std[a:b]).to(device)
            output = loaded.model(h)[-1]
            mean_std = output["mean_pos"].detach().cpu().numpy()
            vel_std = output["pred_vel"].detach().cpu().numpy()
            logits = output["scores"].detach().cpu().numpy()
            full_local = loaded.norm.inverse(np.concatenate([mean_std, vel_std], axis=-1))
            selected_local, _, _ = src.select_modes_with_nms(
                pred_local=full_local,
                scores=logits,
                num_output_modes=int(loaded.report["num_output_modes"]),
                distance_threshold=float(loaded.report["nms_dist_thresh_m"]),
            )
            bs, modes, H, D = selected_local.shape
            flat = selected_local.reshape(bs * modes, H, D)
            pred_g = src.batch_local_to_global(
                flat,
                np.repeat(origins[a:b], modes, axis=0),
                np.repeat(cs[a:b], modes),
                np.repeat(ss[a:b], modes),
            ).reshape(bs, modes, H, D)
            top_list.append(pred_g[:, 0])
            if include_oracle:
                fde = np.linalg.norm(pred_g[:, :, -1, :3] - future[a:b, None, -1, :3], axis=-1)
                idx = fde.argmin(axis=1)
                oracle_list.append(pred_g[np.arange(bs), idx])
    out = {"mtr_uav_paper_top1": np.concatenate(top_list, axis=0)}
    if include_oracle:
        out["mtr_uav_paper_oracle_minfde"] = np.concatenate(oracle_list, axis=0)
    return out


def infer_model(model_name: str, src, loaded: LoadedModel, histories: np.ndarray, future: np.ndarray, device: torch.device, batch_size: int, include_oracle: bool, inference_seed: int):
    if model_name == "flightbertpp":
        return infer_flight(src, loaded, histories, device, batch_size)
    if model_name == "pecnet":
        return infer_pecnet(src, loaded, histories, future, device, batch_size, include_oracle, inference_seed)
    if model_name == "mtr":
        return infer_mtr(src, loaded, histories, future, device, batch_size, include_oracle)
    raise ValueError(model_name)


def write_plots(summary: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] plots skipped: {e}")
        return
    h10 = summary[summary["report_horizon"] == 10].copy()
    if h10.empty:
        return
    h10["order"] = h10["noise_level"].str.extract(r"(\d+)").astype(int)
    for metric, ylabel, fn in [
        ("pos_rmse", "H10 position RMSE [m]", "noise_vs_h10_pos_rmse.png"),
        ("ade", "H10 ADE [m]", "noise_vs_ade.png"),
        ("fde", "H10 FDE [m]", "noise_vs_fde.png"),
    ]:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for method, sub in h10.groupby("method"):
            sub = sub.sort_values("order")
            x = sub["position_sigma_m_mean"].to_numpy(float)
            y = sub[f"{metric}_mean"].to_numpy(float)
            e = sub[f"{metric}_std"].fillna(0).to_numpy(float)
            plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=method)
        plt.xlabel("Position noise sigma [m]")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fn, dpi=180)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["flightbertpp", "pecnet", "mtr"])
    ap.add_argument("--model_source_py", required=True)
    ap.add_argument("--topokoop_source_py", required=True, help="Used only to load the exact same test-window order as TopoKoop evaluation.")
    ap.add_argument("--checkpoints_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_train_seeds", type=int, default=0, help="0 means all seed* directories; positive values keep the first N seeds for smoke tests.")
    ap.add_argument("--noise_mode", choices=["joint", "position_only", "velocity_only"], default="position_only")
    ap.add_argument("--position_sigmas", default="0,0.005,0.01,0.02,0.03,0.05,0.1")
    ap.add_argument("--velocity_sigmas", default="0")
    ap.add_argument("--noise_seeds", default="42,43,44,45,46")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--sample_mode", choices=["first", "evenly", "random"], default="evenly")
    ap.add_argument("--sample_seed", type=int, default=2026)
    ap.add_argument("--report_horizons", default="1,3,5,10")
    ap.add_argument("--include_oracle", action="store_true")
    ap.add_argument("--save_sample_metrics", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    src = load_module(Path(args.model_source_py), f"{args.model}_source")
    src.DEVICE = str(device)

    seed_dirs = discover_seed_dirs(Path(args.checkpoints_root))
    if int(args.max_train_seeds) > 0:
        seed_dirs = seed_dirs[: int(args.max_train_seeds)]
    first_report = json.loads((seed_dirs[0] / model_files(args.model)[1]).read_text(encoding="utf-8"))
    input_len = int(first_report["input_len"])
    horizon = int(first_report["horizon"])
    stride = int(first_report["stride"])
    report_horizons = sorted(set(parse_int_list(args.report_horizons)))
    if any(h < 1 or h > horizon for h in report_horizons):
        raise ValueError(f"report horizons must be within [1,{horizon}]")

    clean_all, future_all = load_reference_windows(Path(args.topokoop_source_py), Path(args.data_root), input_len, horizon, stride)
    metadata_all = build_window_metadata(Path(args.data_root), input_len, horizon, stride)
    if len(metadata_all) != len(clean_all):
        raise RuntimeError(f"metadata/reference sample count mismatch: {len(metadata_all)} vs {len(clean_all)}")

    idx = choose_sample_indices(len(clean_all), args.max_samples, args.sample_mode, args.sample_seed)
    np.save(out_dir / "selected_test_sample_indices.npy", idx)
    clean = clean_all[idx]
    future = future_all[idx]
    metadata = metadata_all.iloc[idx].reset_index(drop=True)
    metadata.to_csv(out_dir / "selected_test_samples.csv", index=False)

    levels = build_noise_levels(args.noise_mode, parse_float_list(args.position_sigmas), parse_float_list(args.velocity_sigmas))
    noise_seeds = parse_int_list(args.noise_seeds)
    print(f"[DATA] total={len(clean_all)}, selected={len(clean)}, input_len={input_len}, horizon={horizon}, stride={stride}")
    print(f"[MODELS] {len(seed_dirs)} training seeds: {[p.name for p in seed_dirs]}")
    print(f"[NOISE] {levels}, seeds={noise_seeds}")

    metric_rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for seed_dir in seed_dirs:
        loaded = load_seed_model(args.model, src, seed_dir, device)
        # Validate all report-level data settings.
        for key, expected in [("input_len", input_len), ("horizon", horizon), ("stride", stride)]:
            if int(loaded.report[key]) != int(expected):
                raise RuntimeError(f"{seed_dir}: inconsistent {key}")
        print(f"\n[LOAD] {args.model}/{loaded.train_seed}", flush=True)

        clean_preds = infer_model(
            args.model, src, loaded, clean, future, device, args.batch_size, args.include_oracle,
            inference_seed=int(loaded.report.get("seed", 0)) + 999,
        )

        for level_name, p_sigma, v_sigma in levels:
            for noise_seed in noise_seeds:
                started = time.perf_counter()
                if p_sigma == 0 and v_sigma == 0:
                    noisy = clean
                    preds = clean_preds
                    pos_rms = 0.0
                    vel_rms = 0.0
                else:
                    # EXACTLY matches TopoKoop evaluator RNG order: position draw first, velocity draw second.
                    rng = np.random.default_rng(int(noise_seed))
                    noisy = clean.copy()
                    pn = rng.normal(0.0, p_sigma, size=noisy[..., :3].shape)
                    vn = rng.normal(0.0, v_sigma, size=noisy[..., 3:6].shape)
                    noisy[..., :3] += pn
                    noisy[..., 3:6] += vn
                    pos_rms = float(np.sqrt(np.mean(pn ** 2)))
                    vel_rms = float(np.sqrt(np.mean(vn ** 2)))
                    preds = infer_model(
                        args.model, src, loaded, noisy, future, device, args.batch_size, args.include_oracle,
                        inference_seed=int(loaded.report.get("seed", 0)) * 100000 + int(noise_seed),
                    )

                for method, pred in preds.items():
                    for H in report_horizons:
                        met = trajectory_metrics(future, pred, H)
                        metric_rows.append({
                            "model_family": args.model,
                            "method": method,
                            "train_seed": loaded.train_seed,
                            "noise_level": level_name,
                            "position_sigma_m": p_sigma,
                            "velocity_sigma_mps": v_sigma,
                            "noise_seed": int(noise_seed),
                            "realized_position_noise_rms_m": pos_rms,
                            "realized_velocity_noise_rms_mps": vel_rms,
                            "report_horizon": int(H),
                            "num_samples": int(len(clean)),
                            **met,
                        })

                    # Per-scene final-H metrics.
                    for scene, ids in metadata.groupby("scene_type").groups.items():
                        ids = np.asarray(list(ids), dtype=np.int64)
                        met = trajectory_metrics(future[ids], pred[ids], horizon)
                        scene_rows.append({
                            "model_family": args.model,
                            "method": method,
                            "train_seed": loaded.train_seed,
                            "noise_level": level_name,
                            "position_sigma_m": p_sigma,
                            "velocity_sigma_mps": v_sigma,
                            "noise_seed": int(noise_seed),
                            "scene_type": scene,
                            "report_horizon": horizon,
                            "num_samples": int(len(ids)),
                            **met,
                        })

                    if args.save_sample_metrics:
                        rmse, ade, fde = per_sample_metrics(future, pred, horizon)
                        for i in range(len(clean)):
                            sample_rows.append({
                                "model_family": args.model,
                                "method": method,
                                "train_seed": loaded.train_seed,
                                "noise_level": level_name,
                                "position_sigma_m": p_sigma,
                                "velocity_sigma_mps": v_sigma,
                                "noise_seed": int(noise_seed),
                                "sample_idx": int(idx[i]),
                                "sample_id": metadata.iloc[i]["sample_id"],
                                "scene_type": metadata.iloc[i]["scene_type"],
                                "h10_pos_rmse": float(rmse[i]),
                                "h10_ade": float(ade[i]),
                                "h10_fde": float(fde[i]),
                            })

                pd.DataFrame(metric_rows).to_csv(out_dir / "noise_metrics_per_run.csv", index=False)
                pd.DataFrame(scene_rows).to_csv(out_dir / "noise_scene_metrics_per_run.csv", index=False)
                if args.save_sample_metrics:
                    pd.DataFrame(sample_rows).to_csv(out_dir / "noise_sample_metrics_h10.csv", index=False)
                print(
                    f"[DONE] train={loaded.train_seed} {level_name} noise_seed={noise_seed} "
                    f"sigma_p={p_sigma:g} sigma_v={v_sigma:g} elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

        del loaded.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_df = pd.DataFrame(metric_rows)
    scene_df = pd.DataFrame(scene_rows)
    summary = summary_mean_std(
        metric_df,
        ["model_family", "method", "noise_level", "report_horizon"],
        exclude=["noise_seed"],
    )
    train_summary = summary_mean_std(
        metric_df.groupby([
            "model_family", "method", "train_seed", "noise_level", "report_horizon"
        ], dropna=False).mean(numeric_only=True).reset_index(),
        ["model_family", "method", "noise_level", "report_horizon"],
    )
    scene_summary = summary_mean_std(
        scene_df,
        ["model_family", "method", "noise_level", "scene_type", "report_horizon"],
        exclude=["noise_seed"],
    )
    summary.to_csv(out_dir / "noise_metrics_summary_all_runs.csv", index=False)
    train_summary.to_csv(out_dir / "noise_metrics_summary_across_training_seeds.csv", index=False)
    scene_summary.to_csv(out_dir / "noise_scene_metrics_summary.csv", index=False)

    # Relative degradation from N0 using all-run mean.
    deg_rows = []
    for (model_family, method, H), sub in summary.groupby(["model_family", "method", "report_horizon"]):
        clean_row = sub[sub["noise_level"] == "N0"]
        if clean_row.empty:
            continue
        c = clean_row.iloc[0]
        for _, r in sub.iterrows():
            rec = {
                "model_family": model_family,
                "method": method,
                "noise_level": r["noise_level"],
                "report_horizon": int(H),
            }
            for m in ["pos_rmse", "ade", "fde"]:
                base = float(c[f"{m}_mean"])
                val = float(r[f"{m}_mean"])
                rec[f"{m}_degradation_percent"] = 100.0 * (val - base) / max(abs(base), 1e-12)
            deg_rows.append(rec)
    pd.DataFrame(deg_rows).to_csv(out_dir / "noise_relative_degradation.csv", index=False)

    report = {
        "model": args.model,
        "model_source_py": str(Path(args.model_source_py)),
        "topokoop_source_py": str(Path(args.topokoop_source_py)),
        "checkpoints_root": str(Path(args.checkpoints_root)),
        "data_root": str(Path(args.data_root)),
        "num_training_seeds": len(seed_dirs),
        "training_seed_dirs": [p.name for p in seed_dirs],
        "num_total_test_samples": int(len(clean_all)),
        "num_selected_test_samples": int(len(clean)),
        "input_len": input_len,
        "horizon": horizon,
        "stride": stride,
        "noise_mode": args.noise_mode,
        "noise_levels": [{"name": n, "position_sigma_m": p, "velocity_sigma_mps": v} for n,p,v in levels],
        "noise_seeds": noise_seeds,
        "include_oracle": bool(args.include_oracle),
        "fairness_notes": [
            "Test windows are loaded through the TopoKoop source module to preserve exact sample order.",
            "NumPy RNG draw order matches evaluate_topokoop_noise_robustness.py.",
            "Noise is added to global physical states before model-specific local-frame conversion and normalization.",
            "Future ground truth is never perturbed.",
            "Primary deployable methods are FlightBERT++ regression, PECNet z0, and MTR top1.",
            "Oracle outputs, when requested, are upper bounds and must not enter the main deployable ranking.",
        ],
    }
    (out_dir / "noise_robustness_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plots:
        write_plots(summary, out_dir)

    h10 = summary[summary["report_horizon"] == horizon]
    cols = [
        "method", "noise_level", "position_sigma_m_mean", "velocity_sigma_mps_mean",
        "pos_rmse_mean", "pos_rmse_std", "ade_mean", "ade_std", "fde_mean", "fde_std"
    ]
    print("\n=== Final baseline H10 noise robustness summary ===")
    print(h10[[c for c in cols if c in h10.columns]].to_string(index=False))
    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
