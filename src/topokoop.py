#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace as _SimpleNamespace




# ========================================================================================
# SECTION: BEHAVIOR MODE ENDPOINT KOOPMAN V2 HMM SELECTOR
# ========================================================================================
def _init_behavior_mode_endpoint_koopman_v2_hmm_selector():
    """
    Behavior-Mode Endpoint-Conditioned Direct Hankel-Koopman V2 HMM Selector

    Core idea:
        Endpoint decides where to go;
        behavior mode decides how to maneuver toward it;
        Koopman decides how the trajectory satisfies local dynamics.

    Data layout:
        DATA_ROOT/train/chains/*.csv, val/chains/*.csv, test/chains/*.csv
    Required columns: x,y,z,vx,vy,vz
    Optional columns: behavior, phase_type, scene_type

    Quick:
        python behavior_mode_endpoint_koopman_v2_hmm_selector.py       --data_root DATA/uav_koopman_v2_city_intent_env/processed       --fast       --out_dir DATA/uav_koopman_v2_city_intent_env/processed/behavior_mode_endpoint_koopman_v1_fast
    """
    import argparse, json, math, random
    from dataclasses import dataclass
    from pathlib import Path
    from typing import List, Optional, Dict, Tuple
    import numpy as np
    import pandas as pd
    STATE_COLS = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    BEHAVIOR_IGNORE = {'', 'none', 'mixed', 'transition', 'unknown', 'nan', None}

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    def parse_int_list(s: str):
        return [int(x.strip()) for x in s.split(',') if x.strip()]

    def parse_float_list(s: str):
        return [float(x.strip()) for x in s.split(',') if x.strip()]

    def is_normal_behavior(phase_type, behavior):
        phase = str(phase_type).strip().lower()
        beh = str(behavior).strip().lower()
        return phase != 'transition' and beh not in BEHAVIOR_IGNORE

    def compute_metrics(y_true, y_pred):
        err = y_pred - y_true
        pe, ve = (err[..., :3], err[..., 3:])
        disp = np.linalg.norm(pe, axis=-1)
        return {'pos_rmse': float(np.sqrt(np.mean(pe ** 2))), 'pos_mae': float(np.mean(np.abs(pe))), 'vel_rmse': float(np.sqrt(np.mean(ve ** 2))), 'vel_mae': float(np.mean(np.abs(ve))), 'all_rmse': float(np.sqrt(np.mean(err ** 2))), 'all_mae': float(np.mean(np.abs(err))), 'ade': float(np.mean(disp)), 'fde': float(np.mean(disp[:, -1]))}

    def selection_score(m, metric, ade_w, fde_w):
        if metric == 'pos_rmse':
            return m['pos_rmse']
        if metric == 'ade':
            return m['ade']
        if metric == 'fde':
            return m['fde']
        if metric == 'combo':
            return m['pos_rmse'] + ade_w * m['ade'] + fde_w * m['fde']
        raise ValueError(metric)

    def softmax_np(z, axis=-1):
        z = z - np.max(z, axis=axis, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)

    def row_behavior_key(row):
        beh = str(row.get('behavior', 'unknown')).strip().lower()
        phase = str(row.get('phase_type', 'normal')).strip().lower()
        return beh if is_normal_behavior(phase, beh) else 'global'

    def map_to_mode(label, modes):
        return label if label in modes else 'global'

    def estimate_behavior_transition(chains, modes, smooth=1.0):
        """Estimate T[i,j] = P(b_{t+1}=j | b_t=i) from training chains."""
        m2i = {m: i for i, m in enumerate(modes)}
        K = len(modes)
        counts = np.ones((K, K), dtype=np.float64) * float(smooth)
        for df in chains:
            if len(df) < 2:
                continue
            labs = [map_to_mode(row_behavior_key(row), modes) for _, row in df.iterrows()]
            for a, b in zip(labs[:-1], labs[1:]):
                counts[m2i[a], m2i[b]] += 1.0
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

    def transition_power(T, steps):
        return np.linalg.matrix_power(T, max(int(steps), 1))

    @dataclass
    class LocalFrameBatch:
        origins: np.ndarray
        c: np.ndarray
        s: np.ndarray

    @dataclass
    class SplitData:
        hist_global: np.ndarray
        future_global: np.ndarray
        hist_local: np.ndarray
        future_local: np.ndarray
        frames: LocalFrameBatch
        behavior_key: List[str]
        dominant_behavior: List[str]
        scene_type: List[str]
        chain_file: List[str]
        horizon: int

    @dataclass
    class Normalizer:
        mean: np.ndarray
        std: np.ndarray

        def transform(self, x):
            return ((x - self.mean) / self.std).astype(np.float64)

        def inverse(self, x):
            return (x * self.std + self.mean).astype(np.float64)

        def to_json(self):
            return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @dataclass
    class FeatureStandardizer:
        mean: np.ndarray
        std: np.ndarray

        @classmethod
        def fit(cls, X):
            m, s = (X.mean(axis=0), X.std(axis=0))
            s[s < 1e-08] = 1.0
            return cls(m, s)

        def transform(self, X):
            return (X - self.mean) / self.std

    def load_split_chains(root: Path, split: str):
        chain_dir = root / split / 'chains'
        files = sorted(chain_dir.glob('*.csv'))
        if not files:
            raise RuntimeError(f'No CSV files found in {chain_dir}')
        chains = []
        for f in files:
            df = pd.read_csv(f)
            missing = [c for c in STATE_COLS if c not in df.columns]
            if missing:
                raise RuntimeError(f'{f} missing columns: {missing}')
            if len(df) < 3:
                continue
            df = df.copy()
            df['_chain_file'] = f.name
            chains.append(df)
        if not chains:
            raise RuntimeError(f'No valid chains loaded from {chain_dir}')
        return chains

    def make_one_frame(last_state, prev_state=None):
        origin = last_state[:3].astype(np.float64).copy()
        vx, vy = (float(last_state[3]), float(last_state[4]))
        sp = math.sqrt(vx * vx + vy * vy)
        if sp < 1e-06 and prev_state is not None:
            dx, dy = (float(last_state[0] - prev_state[0]), float(last_state[1] - prev_state[1]))
            sp = math.sqrt(dx * dx + dy * dy)
            if sp >= 1e-06:
                vx, vy = (dx, dy)
        if sp < 1e-06:
            return (origin, 1.0, 0.0)
        return (origin, float(vx / sp), float(vy / sp))

    def global_to_local_one(x, origin, c, s):
        p, v = (x[..., :3], x[..., 3:])
        r = p - origin
        lx = c * r[..., 0] + s * r[..., 1]
        ly = -s * r[..., 0] + c * r[..., 1]
        lz = r[..., 2]
        lvx = c * v[..., 0] + s * v[..., 1]
        lvy = -s * v[..., 0] + c * v[..., 1]
        lvz = v[..., 2]
        return np.stack([lx, ly, lz, lvx, lvy, lvz], axis=-1)

    def batch_local_to_global(z, frames):
        c, s = (frames.c[:, None], frames.s[:, None])
        origin = frames.origins[:, None, :]
        lx, ly, lz = (z[..., 0], z[..., 1], z[..., 2])
        lvx, lvy, lvz = (z[..., 3], z[..., 4], z[..., 5])
        gx = origin[..., 0] + c * lx - s * ly
        gy = origin[..., 1] + s * lx + c * ly
        gz = origin[..., 2] + lz
        gvx = c * lvx - s * lvy
        gvy = s * lvx + c * lvy
        gvz = lvz
        return np.stack([gx, gy, gz, gvx, gvy, gvz], axis=-1)

    def make_split_data(chains, input_len, horizon, stride):
        hg, fg, hl, fl = ([], [], [], [])
        origins, cs, ss = ([], [], [])
        bkeys, doms, scenes, cfiles = ([], [], [], [])
        for df in chains:
            arr = df[STATE_COLS].to_numpy(dtype=np.float64)
            max_start = len(df) - input_len - horizon
            if max_start < 0:
                continue
            for s0 in range(0, max_start + 1, stride):
                hist_g = arr[s0:s0 + input_len]
                fut_g = arr[s0 + input_len:s0 + input_len + horizon]
                origin, c, s = make_one_frame(hist_g[-1], hist_g[-2] if input_len > 1 else None)
                hist_l = global_to_local_one(hist_g, origin, c, s)
                fut_l = global_to_local_one(fut_g, origin, c, s)
                row = df.iloc[s0 + input_len - 1]
                beh = str(row.get('behavior', 'unknown')).strip().lower()
                phase = str(row.get('phase_type', 'normal')).strip().lower()
                bkey = beh if is_normal_behavior(phase, beh) else 'global'
                if 'behavior' in df.columns:
                    seq = [str(x).strip().lower() for x in df.iloc[s0:s0 + input_len]['behavior'].tolist()]
                    dom = pd.Series(seq).mode().iloc[0] if seq else 'unknown'
                else:
                    dom = 'unknown'
                hg.append(hist_g)
                fg.append(fut_g)
                hl.append(hist_l)
                fl.append(fut_l)
                origins.append(origin)
                cs.append(c)
                ss.append(s)
                bkeys.append(bkey)
                doms.append(dom)
                scenes.append(str(row.get('scene_type', 'unknown')))
                cfiles.append(str(row.get('_chain_file', '')))
        if not hg:
            raise RuntimeError(f'No samples for horizon={horizon}')
        return SplitData(np.stack(hg), np.stack(fg), np.stack(hl), np.stack(fl), LocalFrameBatch(np.stack(origins), np.asarray(cs), np.asarray(ss)), bkeys, doms, scenes, cfiles, horizon)

    def fit_state_normalizer(train):
        states = np.concatenate([train.hist_local.reshape(-1, 6), train.future_local.reshape(-1, 6)], axis=0)
        m, s = (states.mean(axis=0), states.std(axis=0))
        s[s < 1e-08] = 1.0
        return Normalizer(m, s)

    def make_hankel_features(data, norm):
        return norm.transform(data.hist_local).reshape(data.hist_local.shape[0], -1)

    def add_bias(X):
        return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)

    def ridge_fit(X, Y, ridge):
        D = X.shape[1]
        return Y.T @ X @ np.linalg.inv(X.T @ X + ridge * np.eye(D))

    def full_target_std(data, norm):
        return norm.transform(data.future_local).reshape(data.future_local.shape[0], -1)

    def endpoint_target_std(data, norm):
        return norm.transform(data.future_local)[:, -1, :3]

    def std_full_to_global(Ystd_flat, data, norm):
        N, H = (data.hist_local.shape[0], data.horizon)
        local = norm.inverse(Ystd_flat.reshape(N, H, 6))
        return batch_local_to_global(local, data.frames)

    def make_anchor(Ybase, endpoint_pred, H, alpha_max, gamma):
        Y = Ybase.copy().reshape(-1, H, 6)
        base_end = Y[:, -1, :3]
        r_e = endpoint_pred - base_end
        alphas = alpha_max * (np.arange(1, H + 1, dtype=np.float64) / H) ** gamma
        alphas = np.clip(alphas, 0.0, 1.0)
        for k, a in enumerate(alphas):
            Y[:, k, :3] += a * r_e
        return (Y.reshape(Ybase.shape[0], -1), r_e)

    def make_aug_feature(X, r_e, H, alpha_max, gamma):
        alphas = alpha_max * (np.arange(1, H + 1, dtype=np.float64) / H) ** gamma
        alphas = np.clip(alphas, 0.0, 1.0)
        shaped = np.concatenate([a * r_e for a in alphas], axis=1)
        return np.concatenate([X, r_e, shaped], axis=1)

    @dataclass
    class BEKoopman:
        H: int
        norm: Normalizer
        fstd: FeatureStandardizer
        modes: List[str]
        W0: np.ndarray
        We: np.ndarray
        Wr: np.ndarray
        Wcls: np.ndarray
        Wde: Dict[str, np.ndarray]
        Wdr: Dict[str, np.ndarray]
        alpha_max: float
        gamma: float

        def X(self, data):
            return add_bias(self.fstd.transform(make_hankel_features(data, self.norm)))

        def base(self, X):
            return X @ self.W0.T

        def ep_global(self, X):
            return X @ self.We.T

        def ep_mode(self, X, mode):
            ep = self.ep_global(X)
            if mode in self.Wde:
                ep = ep + X @ self.Wde[mode].T
            return ep

        def mode_std(self, X, mode):
            Y0 = self.base(X)
            ep = self.ep_mode(X, mode)
            Ya, re = make_anchor(Y0, ep, self.H, self.alpha_max, self.gamma)
            Xaug = make_aug_feature(X, re, self.H, self.alpha_max, self.gamma)
            R = Xaug @ self.Wr.T
            if mode in self.Wdr:
                R = R + Xaug @ self.Wdr[mode].T
            return Ya + R

        def global_anchor_std(self, X):
            return make_anchor(self.base(X), self.ep_global(X), self.H, self.alpha_max, self.gamma)[0]

        def global_residual_std(self, X):
            Y0 = self.base(X)
            ep = self.ep_global(X)
            Ya, re = make_anchor(Y0, ep, self.H, self.alpha_max, self.gamma)
            return Ya + make_aug_feature(X, re, self.H, self.alpha_max, self.gamma) @ self.Wr.T

        def behavior_probs(self, X):
            return softmax_np(X @ self.Wcls.T, axis=1)

        def label_to_mode(self, b):
            return b if b in self.modes else 'global'

        def hard_std(self, data):
            X = self.X(data)
            out = np.zeros((X.shape[0], self.H * 6), dtype=np.float64)
            labs = [self.label_to_mode(b) for b in data.behavior_key]
            for mode in self.modes:
                idx = np.asarray([b == mode for b in labs])
                if idx.any():
                    out[idx] = self.mode_std(X[idx], mode)
            return out

        def soft_std(self, data):
            X = self.X(data)
            probs = self.behavior_probs(X)
            Y = np.zeros((X.shape[0], self.H * 6), dtype=np.float64)
            for j, mode in enumerate(self.modes):
                Y += probs[:, j:j + 1] * self.mode_std(X, mode)
            return Y

        def oracle_std(self, data):
            X = self.X(data)
            true = data.future_global
            bestY, best = (None, None)
            for mode in self.modes:
                Ystd = self.mode_std(X, mode)
                Yg = std_full_to_global(Ystd, data, self.norm)
                fde = np.linalg.norm(Yg[:, -1, :3] - true[:, -1, :3], axis=-1)
                if bestY is None:
                    bestY, best = (Ystd.copy(), fde)
                else:
                    mask = fde < best
                    bestY[mask] = Ystd[mask]
                    best[mask] = fde[mask]
            return bestY

    def stack_mode_predictions_std(model, X):
        """Return all behavior-mode candidates: [N, B, H*6]."""
        return np.stack([model.mode_std(X, mode) for mode in model.modes], axis=1)

    def start_probs_from_rule_labels(data, modes):
        m2i = {m: i for i, m in enumerate(modes)}
        P = np.zeros((len(data.behavior_key), len(modes)), dtype=np.float64)
        for i, b in enumerate(data.behavior_key):
            P[i, m2i.get(b, 0)] = 1.0
        return P

    def future_behavior_priors(model, data, X, T, steps, start='classifier'):
        if start == 'onehot':
            P0 = start_probs_from_rule_labels(data, model.modes)
        elif start == 'classifier':
            P0 = model.behavior_probs(X)
        else:
            raise ValueError(start)
        TH = transition_power(T, steps)
        P = P0 @ TH
        return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

    def hmm_select_candidates_std(model, data, T, steps, start='classifier', combine='top1', lambda_shift=0.0, temperature=1.0):
        """
        Transition-aware selector:
            score_b = log P(b_{t+H}|history) - lambda_shift * ||end_b - end_global||^2

        This does not use future ground truth. It only uses current behavior/posterior,
        the learned transition matrix, and candidate trajectory geometry.
        """
        X = model.X(data)
        cand = stack_mode_predictions_std(model, X)
        N, B, D = cand.shape
        priors = future_behavior_priors(model, data, X, T, steps=steps, start=start)
        Y0 = model.base(X).reshape(N, model.H, 6)
        cand3 = cand.reshape(N, B, model.H, 6)
        endpoint_shift = cand3[:, :, -1, :3] - Y0[:, None, -1, :3]
        shift_penalty = np.sum(endpoint_shift ** 2, axis=-1)
        scores = np.log(np.maximum(priors, 1e-12)) - float(lambda_shift) * shift_penalty
        if combine == 'top1':
            idx = np.argmax(scores, axis=1)
            return cand[np.arange(N), idx]
        if combine == 'soft':
            w = softmax_np(scores / max(float(temperature), 1e-08), axis=1)
            return np.sum(w[:, :, None] * cand, axis=1)
        raise ValueError(combine)

    def fit_behavior_model(train, norm, fstd, W0, We, modes, ridge_res, ridge_delta, ridge_cls, min_samples, alpha, gamma):
        H = train.horizon
        X = add_bias(fstd.transform(make_hankel_features(train, norm)))
        Y = full_target_std(train, norm)
        E = endpoint_target_std(train, norm)
        Y0 = X @ W0.T
        Eg = X @ We.T
        Ya, re = make_anchor(Y0, Eg, H, alpha, gamma)
        Xaug = make_aug_feature(X, re, H, alpha, gamma)
        Wr = ridge_fit(Xaug, Y - Ya, ridge_res)
        labels = np.asarray([b if b in modes else 'global' for b in train.behavior_key], dtype=object)
        Wde, Wdr = ({}, {})
        for mode in modes:
            if mode == 'global':
                continue
            idx = labels == mode
            if idx.sum() < min_samples:
                continue
            Wde_m = ridge_fit(X[idx], E[idx] - Eg[idx], ridge_delta)
            Wde[mode] = Wde_m
            Em = Eg[idx] + X[idx] @ Wde_m.T
            Yam, rem = make_anchor(Y0[idx], Em, H, alpha, gamma)
            Xaugm = make_aug_feature(X[idx], rem, H, alpha, gamma)
            target_delta = Y[idx] - Yam - Xaugm @ Wr.T
            Wdr[mode] = ridge_fit(Xaugm, target_delta, ridge_delta)
        onehot = np.zeros((X.shape[0], len(modes)), dtype=np.float64)
        m2i = {m: i for i, m in enumerate(modes)}
        for i, b in enumerate(labels):
            onehot[i, m2i.get(b, 0)] = 1.0
        Wcls = ridge_fit(X, onehot, ridge_cls)
        return BEKoopman(H, norm, fstd, modes, W0, We, Wr, Wcls, Wde, Wdr, alpha, gamma)

    def cv_predict(data, dt):
        N, H = (data.hist_global.shape[0], data.horizon)
        out = np.zeros((N, H, 6), dtype=np.float64)
        p = data.hist_global[:, -1, :3].copy()
        v = data.hist_global[:, -1, 3:].copy()
        for k in range(H):
            p = p + dt * v
            out[:, k, :3] = p
            out[:, k, 3:] = v
        return out

    def ca_predict(data, dt):
        N, H = (data.hist_global.shape[0], data.horizon)
        out = np.zeros((N, H, 6), dtype=np.float64)
        last = data.hist_global[:, -1, :]
        prev = data.hist_global[:, -2, :] if data.hist_global.shape[1] > 1 else last
        p = last[:, :3].copy()
        v = last[:, 3:].copy()
        a = (last[:, 3:] - prev[:, 3:]) / max(dt, 1e-08)
        for k in range(H):
            p = p + dt * v + 0.5 * dt * dt * a
            v = v + dt * a
            out[:, k, :3] = p
            out[:, k, 3:] = v
        return out

    def eval_array(data, pred, method):
        met = compute_metrics(data.future_global, pred)
        rows = []
        for i in range(pred.shape[0]):
            im = compute_metrics(data.future_global[i:i + 1], pred[i:i + 1])
            rows.append({'method': method, 'horizon': data.horizon, 'scene_type': data.scene_type[i], 'dominant_behavior': data.dominant_behavior[i], 'behavior_key': data.behavior_key[i], 'chain_file': data.chain_file[i], **im})
        return (met, pd.DataFrame(rows))

    def group_metrics(df, col):
        rows = []
        cols = ['pos_rmse', 'pos_mae', 'vel_rmse', 'vel_mae', 'all_rmse', 'all_mae', 'ade', 'fde']
        for key, g in df.groupby(['method', 'horizon', col], dropna=False):
            m, H, val = key
            row = {'method': m, 'horizon': H, col: val, 'num_samples': len(g)}
            for c in cols:
                row[c] = float(g[c].mean())
            rows.append(row)
        return pd.DataFrame(rows)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument('--data_root', required=True)
        ap.add_argument('--out_dir', default=None)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizons', default='1,3,5,10')
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--direct_ridge_grid', default='1e-3,3e-3,1e-2,3e-2,1e-1')
        ap.add_argument('--endpoint_ridge', type=float, default=0.01)
        ap.add_argument('--residual_ridge', type=float, default=0.01)
        ap.add_argument('--behavior_delta_ridge', type=float, default=1.0)
        ap.add_argument('--cls_ridge', type=float, default=0.01)
        ap.add_argument('--min_behavior_samples', type=int, default=40)
        ap.add_argument('--hmm_smooth', type=float, default=1.0)
        ap.add_argument('--hmm_lambda_grid', default='0,0.03,0.1,0.3,1.0')
        ap.add_argument('--hmm_temperature', type=float, default=1.0)
        ap.add_argument('--alpha_grid', default='0.3,0.5,0.6,0.8,1.0')
        ap.add_argument('--gamma_grid', default='1,2,3,4')
        ap.add_argument('--selection_metric', default='combo', choices=['pos_rmse', 'ade', 'fde', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--fast', action='store_true')
        args = ap.parse_args()
        set_seed(args.seed)
        root = Path(args.data_root)
        out_dir = Path(args.out_dir) if args.out_dir else root / 'behavior_mode_endpoint_koopman_v2_hmm_selector'
        ensure_dir(out_dir)
        horizons = parse_int_list(args.horizons)
        if args.fast:
            horizons = [10]
        direct_ridges = parse_float_list(args.direct_ridge_grid)
        alphas = parse_float_list(args.alpha_grid)
        gammas = parse_float_list(args.gamma_grid)
        hmm_lambdas = parse_float_list(args.hmm_lambda_grid)
        train_chains = load_split_chains(root, 'train')
        val_chains = load_split_chains(root, 'val')
        test_chains = load_split_chains(root, 'test')
        compare_rows = []
        sample_dfs = []
        sweep_rows = []
        selected = {}
        for H in horizons:
            print(f'\n=== Horizon H={H} ===')
            train = make_split_data(train_chains, args.input_len, H, args.stride)
            val = make_split_data(val_chains, args.input_len, H, args.stride)
            test = make_split_data(test_chains, args.input_len, H, args.stride)
            print(f'samples: train={train.hist_global.shape[0]}, val={val.hist_global.shape[0]}, test={test.hist_global.shape[0]}')
            norm = fit_state_normalizer(train)
            Xtr_raw = make_hankel_features(train, norm)
            Xva_raw = make_hankel_features(val, norm)
            Xte_raw = make_hankel_features(test, norm)
            fstd = FeatureStandardizer.fit(Xtr_raw)
            Xtr = add_bias(fstd.transform(Xtr_raw))
            Xva = add_bias(fstd.transform(Xva_raw))
            Xte = add_bias(fstd.transform(Xte_raw))
            Ytr = full_target_std(train, norm)
            Etr = endpoint_target_std(train, norm)
            counts = pd.Series(train.behavior_key).value_counts()
            modes = ['global'] + sorted([b for b, c in counts.items() if b != 'global' and c >= args.min_behavior_samples])
            print('behavior modes:', modes)
            print('behavior counts:\n' + counts.to_string())
            T_behavior = estimate_behavior_transition(train_chains, modes, smooth=args.hmm_smooth)
            print('behavior transition matrix T rows=current, cols=next:')
            print(pd.DataFrame(T_behavior, index=modes, columns=modes).round(4).to_string())
            for name, pred in [('constant_velocity', cv_predict(test, args.dt)), ('constant_acceleration', ca_predict(test, args.dt))]:
                met, sdf = eval_array(test, pred, name)
                compare_rows.append({'method': name, 'horizon': H, **met})
                sample_dfs.append(sdf)
                print(f'{name}: {met}')
            best_direct = None
            W0 = None
            for r in direct_ridges:
                W = ridge_fit(Xtr, Ytr, r)
                met = compute_metrics(val.future_global, std_full_to_global(Xva @ W.T, val, norm))
                sc = selection_score(met, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight)
                row = {'horizon': H, 'method_group': 'global_direct_hankel_koopman', 'ridge': r, 'score': sc, **met}
                sweep_rows.append(row)
                if best_direct is None or sc < best_direct['score']:
                    best_direct = row
                    W0 = W
            pred = std_full_to_global(Xte @ W0.T, test, norm)
            dname = f"global_direct_hankel_koopman_r{best_direct['ridge']:g}"
            met, sdf = eval_array(test, pred, dname)
            compare_rows.append({'method': dname, 'horizon': H, **met})
            sample_dfs.append(sdf)
            print(f'{dname}: val_best={best_direct}; test={met}')
            We = ridge_fit(Xtr, Etr, args.endpoint_ridge)
            best = {k: None for k in ['global_endpoint_anchor_koopman', 'global_endpoint_residual_koopman', 'hard_behavior_endpoint_koopman', 'soft_behavior_mixture_koopman', 'hmm_onehot_top1_behavior_koopman', 'hmm_classifier_top1_behavior_koopman', 'hmm_classifier_soft_behavior_koopman']}
            models = {}
            for a in alphas:
                for g in gammas:
                    model = fit_behavior_model(train, norm, fstd, W0, We, modes, args.residual_ridge, args.behavior_delta_ridge, args.cls_ridge, args.min_behavior_samples, a, g)
                    Xv = model.X(val)
                    predstds = {'global_endpoint_anchor_koopman': model.global_anchor_std(Xv), 'global_endpoint_residual_koopman': model.global_residual_std(Xv), 'hard_behavior_endpoint_koopman': model.hard_std(val), 'soft_behavior_mixture_koopman': model.soft_std(val)}
                    for mn, Ys in predstds.items():
                        met = compute_metrics(val.future_global, std_full_to_global(Ys, val, norm))
                        sc = selection_score(met, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight)
                        row = {'horizon': H, 'method_group': mn, 'alpha_max': a, 'gamma': g, 'score': sc, **met}
                        sweep_rows.append(row)
                        if best[mn] is None or sc < best[mn]['score']:
                            best[mn] = row
                            models[mn] = model
                    for lam in hmm_lambdas:
                        hmm_predstds = {'hmm_onehot_top1_behavior_koopman': hmm_select_candidates_std(model, val, T_behavior, steps=H, start='onehot', combine='top1', lambda_shift=lam, temperature=args.hmm_temperature), 'hmm_classifier_top1_behavior_koopman': hmm_select_candidates_std(model, val, T_behavior, steps=H, start='classifier', combine='top1', lambda_shift=lam, temperature=args.hmm_temperature), 'hmm_classifier_soft_behavior_koopman': hmm_select_candidates_std(model, val, T_behavior, steps=H, start='classifier', combine='soft', lambda_shift=lam, temperature=args.hmm_temperature)}
                        for mn, Ys in hmm_predstds.items():
                            met = compute_metrics(val.future_global, std_full_to_global(Ys, val, norm))
                            sc = selection_score(met, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight)
                            row = {'horizon': H, 'method_group': mn, 'alpha_max': a, 'gamma': g, 'lambda_shift': lam, 'score': sc, **met}
                            sweep_rows.append(row)
                            if best[mn] is None or sc < best[mn]['score']:
                                best[mn] = row
                                models[mn] = model
            selected[f'H{H}'] = {'direct': best_direct, 'modes': modes, 'best_by_method': best, 'behavior_counts': counts.to_dict(), 'transition_matrix': T_behavior.tolist()}
            for mn, model in models.items():
                if mn == 'global_endpoint_anchor_koopman':
                    Ys = model.global_anchor_std(model.X(test))
                elif mn == 'global_endpoint_residual_koopman':
                    Ys = model.global_residual_std(model.X(test))
                elif mn == 'hard_behavior_endpoint_koopman':
                    Ys = model.hard_std(test)
                elif mn == 'soft_behavior_mixture_koopman':
                    Ys = model.soft_std(test)
                elif mn == 'hmm_onehot_top1_behavior_koopman':
                    br = best[mn]
                    Ys = hmm_select_candidates_std(model, test, T_behavior, steps=H, start='onehot', combine='top1', lambda_shift=br.get('lambda_shift', 0.0), temperature=args.hmm_temperature)
                elif mn == 'hmm_classifier_top1_behavior_koopman':
                    br = best[mn]
                    Ys = hmm_select_candidates_std(model, test, T_behavior, steps=H, start='classifier', combine='top1', lambda_shift=br.get('lambda_shift', 0.0), temperature=args.hmm_temperature)
                elif mn == 'hmm_classifier_soft_behavior_koopman':
                    br = best[mn]
                    Ys = hmm_select_candidates_std(model, test, T_behavior, steps=H, start='classifier', combine='soft', lambda_shift=br.get('lambda_shift', 0.0), temperature=args.hmm_temperature)
                else:
                    raise ValueError(mn)
                pred = std_full_to_global(Ys, test, norm)
                br = best[mn]
                if 'lambda_shift' in br:
                    outname = f"{mn}_a{br['alpha_max']:g}_g{br['gamma']:g}_lam{br['lambda_shift']:g}"
                else:
                    outname = f"{mn}_a{br['alpha_max']:g}_g{br['gamma']:g}"
                met, sdf = eval_array(test, pred, outname)
                compare_rows.append({'method': outname, 'horizon': H, **met})
                sample_dfs.append(sdf)
                print(f'{outname}: val_best={br}; test={met}')
            oracle_model = models['hard_behavior_endpoint_koopman']
            pred = std_full_to_global(oracle_model.oracle_std(test), test, norm)
            met, sdf = eval_array(test, pred, 'oracle_behavior_endpoint_koopman')
            compare_rows.append({'method': 'oracle_behavior_endpoint_koopman', 'horizon': H, **met})
            sample_dfs.append(sdf)
            print(f'oracle_behavior_endpoint_koopman: test={met}')
        compare_df = pd.DataFrame(compare_rows)
        sample_df = pd.concat(sample_dfs, ignore_index=True)
        sweep_df = pd.DataFrame(sweep_rows)
        compare_df.to_csv(out_dir / 'compare_horizon_metrics_test.csv', index=False)
        sample_df.to_csv(out_dir / 'sample_metrics_test.csv', index=False)
        sweep_df.to_csv(out_dir / 'model_sweep_val.csv', index=False)
        group_metrics(sample_df, 'scene_type').to_csv(out_dir / 'grouped_metrics_by_scene_test.csv', index=False)
        group_metrics(sample_df, 'dominant_behavior').to_csv(out_dir / 'grouped_metrics_by_behavior_test.csv', index=False)
        group_metrics(sample_df, 'behavior_key').to_csv(out_dir / 'grouped_metrics_by_behavior_key_test.csv', index=False)
        summary = []
        for m in compare_df['method'].unique():
            for h in [horizons[0], horizons[-1]]:
                sub = compare_df[(compare_df.method == m) & (compare_df.horizon == h)]
                if len(sub):
                    summary.append(sub.iloc[0].to_dict())
        pd.DataFrame(summary).to_csv(out_dir / 'ablation_summary_hfirst_hlast.csv', index=False)
        report = {'version': 'behavior_mode_endpoint_koopman_v2_hmm_selector', 'data_root': str(root), 'out_dir': str(out_dir), 'input_len': args.input_len, 'horizons': horizons, 'stride': args.stride, 'dt': args.dt, 'seed': args.seed, 'selected_report': selected, 'hmm_smooth': args.hmm_smooth, 'hmm_lambda_grid': hmm_lambdas, 'hmm_temperature': args.hmm_temperature, 'notes': ['Endpoint decides where to go; behavior mode decides how to maneuver; Koopman decides local dynamic trajectory shape.', 'V2 adds HMM/transition-aware future behavior selection on top of V1 behavior candidates.', 'hmm_onehot_top1 starts from current rule behavior and propagates through transition matrix.', 'hmm_classifier_top1/soft starts from ridge behavior posterior and propagates through transition matrix.', 'oracle_behavior_endpoint_koopman is a behavior-candidate upper bound.']}
        with open(out_dir / 'report_behavior_mode_endpoint_koopman_v2_hmm_selector.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('\n=== Behavior-Mode Endpoint-Conditioned Direct Hankel-Koopman V2 HMM Selector finished ===')
        print(compare_df.to_string(index=False))
        print(f'\nOutput directory: {out_dir}')

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

behavior_mode_endpoint_koopman_v2_hmm_selector = _init_behavior_mode_endpoint_koopman_v2_hmm_selector()


# ========================================================================================
# SECTION: AXIS FEATURE DECOUPLED POSITION BEHAVIOR KOOPMAN V5
# ========================================================================================
def _init_axis_feature_decoupled_position_behavior_koopman_v5():
    """
    axis_feature_decoupled_position_behavior_koopman_v5.py

    V5: Axis-feature-decoupled, position-only Behavior Koopman.

    Compared with V4:
        - Still predicts position only.
        - Still splits XY branch and Z branch.
        - New: XY and Z branches use different feature spaces.
            XY branch = common Hankel features + planar motion features
            Z  branch = common Hankel features + vertical motion features

    Dependency:
        behavior_mode_endpoint_koopman_v2_hmm_selector.py

    Fast test:
        python axis_feature_decoupled_position_behavior_koopman_v5.py --data_root DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed --fast --out_dir DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed/axis_feature_decoupled_posonly_koopman_v5_fast
    """
    import argparse
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    bk = behavior_mode_endpoint_koopman_v2_hmm_selector

    def parse_float_list(s):
        return [float(x.strip()) for x in str(s).split(',') if x.strip()]

    def parse_int_list(s):
        return [int(x.strip()) for x in str(s).split(',') if x.strip()]

    def ensure_dir(p):
        Path(p).mkdir(parents=True, exist_ok=True)

    def score_pos_metrics(met, metric='combo', ade_weight=0.5, fde_weight=0.25):
        if metric == 'pos_rmse':
            return met['pos_rmse']
        if metric == 'ade':
            return met['ade']
        if metric == 'fde':
            return met['fde']
        if metric == 'fde_xy':
            return met['fde_xy']
        if metric == 'fde_z':
            return met['fde_z']
        return met['pos_rmse'] + ade_weight * met['ade'] + fde_weight * met['fde']

    def compute_position_metrics(y_true, y_pred):
        yt = y_true[:, :, :3]
        yp = y_pred[:, :, :3]
        diff = yp - yt
        diff_xy = diff[:, :, :2]
        diff_z = diff[:, :, 2]
        step_err = np.linalg.norm(diff, axis=-1)
        step_err_xy = np.linalg.norm(diff_xy, axis=-1)
        step_err_z = np.abs(diff_z)
        final_err = step_err[:, -1]
        final_err_xy = step_err_xy[:, -1]
        final_err_z = step_err_z[:, -1]
        return {'pos_rmse': float(np.sqrt(np.mean(diff ** 2))), 'pos_mae': float(np.mean(np.abs(diff))), 'ade': float(np.mean(step_err)), 'fde': float(np.mean(final_err)), 'ade_xy': float(np.mean(step_err_xy)), 'fde_xy': float(np.mean(final_err_xy)), 'ade_z': float(np.mean(step_err_z)), 'fde_z': float(np.mean(final_err_z)), 'z_ratio_fde': float(np.mean(final_err_z / (final_err + 1e-12))), 'z_ratio_ade': float(np.mean(step_err_z.mean(axis=1) / (step_err.mean(axis=1) + 1e-12)))}

    def eval_position_array(data, pred_global, method):
        met = compute_position_metrics(data.future_global, pred_global)
        yt = data.future_global[:, :, :3]
        yp = pred_global[:, :, :3]
        diff = yp - yt
        step_err = np.linalg.norm(diff, axis=-1)
        step_err_xy = np.linalg.norm(diff[:, :, :2], axis=-1)
        step_err_z = np.abs(diff[:, :, 2])
        n = yt.shape[0]

        def safe_attr(name, default):
            if hasattr(data, name):
                v = getattr(data, name)
                if len(v) == n:
                    return v
            return [default] * n
        sdf = pd.DataFrame({'method': method, 'sample_idx': np.arange(n, dtype=int), 'chain_file': safe_attr('chain_file', ''), 'scene_type': safe_attr('scene_type', ''), 'dominant_behavior': safe_attr('dominant_behavior', ''), 'behavior_key': safe_attr('behavior_key', ''), 'ade': step_err.mean(axis=1), 'fde': step_err[:, -1], 'ade_xy': step_err_xy.mean(axis=1), 'fde_xy': step_err_xy[:, -1], 'ade_z': step_err_z.mean(axis=1), 'fde_z': step_err_z[:, -1]})
        sdf['z_ratio_fde'] = sdf['fde_z'] / (sdf['fde'] + 1e-12)
        sdf['z_ratio_ade'] = sdf['ade_z'] / (sdf['ade'] + 1e-12)
        return (met, sdf)

    def group_position_metrics(sample_df, group_col):
        rows = []
        if group_col not in sample_df.columns:
            return pd.DataFrame()
        for (method, group_value), sub in sample_df.groupby(['method', group_col], dropna=False):
            rows.append({'method': method, group_col: group_value, 'num_samples': len(sub), 'ade_mean': sub['ade'].mean(), 'fde_mean': sub['fde'].mean(), 'ade_xy_mean': sub['ade_xy'].mean(), 'fde_xy_mean': sub['fde_xy'].mean(), 'ade_z_mean': sub['ade_z'].mean(), 'fde_z_mean': sub['fde_z'].mean(), 'z_ratio_fde_mean': sub['z_ratio_fde'].mean(), 'fde_p90': sub['fde'].quantile(0.9), 'fde_z_p90': sub['fde_z'].quantile(0.9)})
        return pd.DataFrame(rows).sort_values([group_col, 'fde_mean']).reset_index(drop=True)

    def _tail(arr, k):
        """Return last k time steps, or all if shorter."""
        if arr.shape[1] <= k:
            return arr
        return arr[:, -k:, :]

    def _last_delta(a, lag):
        """a: [N,T,D], return a[-1]-a[-1-lag], robust for short T."""
        t = a.shape[1]
        lag = min(lag, t - 1)
        return a[:, -1, :] - a[:, -1 - lag, :]

    def planar_extra_features(data, dt=0.1):
        """
        Features designed for XY branch:
            displacement, planar velocity, acceleration, path length,
            heading / turn proxy / curvature proxy.
        """
        hist = data.hist_global
        p = hist[:, :, :3]
        v = hist[:, :, 3:6]
        xy = p[:, :, :2]
        vxy = v[:, :, :2]
        n = hist.shape[0]
        d1 = _last_delta(xy, 1)
        d3 = _last_delta(xy, 3)
        d5 = _last_delta(xy, 5)
        d10 = _last_delta(xy, 10)
        v_last = vxy[:, -1, :]
        v_mean5 = _tail(vxy, 5).mean(axis=1)
        v_std5 = _tail(vxy, 5).std(axis=1)
        v_prev = vxy[:, -2, :] if vxy.shape[1] >= 2 else v_last
        a_last = (v_last - v_prev) / max(dt, 1e-08)
        step = np.diff(xy, axis=1)
        step_norm = np.linalg.norm(step, axis=-1)
        path_len5 = _tail(step_norm[:, :, None], 5).sum(axis=1)
        path_len_all = step_norm.sum(axis=1, keepdims=True)
        net = xy[:, -1, :] - xy[:, 0, :]
        net_norm = np.linalg.norm(net, axis=-1, keepdims=True)
        straightness = net_norm / (path_len_all + 1e-08)
        heading_last = np.arctan2(v_last[:, 1], v_last[:, 0] + 1e-12)
        speed_last = np.linalg.norm(v_last, axis=-1, keepdims=True)
        speed_mean5 = np.linalg.norm(_tail(vxy, 5), axis=-1).mean(axis=1, keepdims=True)
        speed_std5 = np.linalg.norm(_tail(vxy, 5), axis=-1).std(axis=1, keepdims=True)
        if step.shape[1] >= 2:
            h = np.arctan2(step[:, :, 1], step[:, :, 0] + 1e-12)
            dh = np.diff(np.unwrap(h, axis=1), axis=1)
            turn_sum5 = _tail(np.abs(dh)[:, :, None], 5).sum(axis=1)
            turn_last = dh[:, -1:] if dh.shape[1] else np.zeros((n, 1))
        else:
            turn_sum5 = np.zeros((n, 1))
            turn_last = np.zeros((n, 1))
        cross_va = v_last[:, 0:1] * a_last[:, 1:2] - v_last[:, 1:2] * a_last[:, 0:1]
        curvature_proxy = cross_va / (speed_last ** 3 + 1e-06)
        return np.concatenate([d1, d3, d5, d10, v_last, v_mean5, v_std5, a_last, path_len5, path_len_all, net_norm, straightness, speed_last, speed_mean5, speed_std5, np.sin(heading_last)[:, None], np.cos(heading_last)[:, None], turn_sum5, turn_last, cross_va, curvature_proxy], axis=1)

    def vertical_extra_features(data, dt=0.1):
        """
        Features designed for Z branch:
            z trend, vertical velocity/acceleration, z range, vertical slope.
        """
        hist = data.hist_global
        z = hist[:, :, 2:3]
        vz = hist[:, :, 5:6]
        n = hist.shape[0]
        dz1 = _last_delta(z, 1)
        dz3 = _last_delta(z, 3)
        dz5 = _last_delta(z, 5)
        dz10 = _last_delta(z, 10)
        vz_last = vz[:, -1, :]
        vz_mean5 = _tail(vz, 5).mean(axis=1)
        vz_std5 = _tail(vz, 5).std(axis=1)
        vz_prev = vz[:, -2, :] if vz.shape[1] >= 2 else vz_last
        az_last = (vz_last - vz_prev) / max(dt, 1e-08)
        z_tail5 = _tail(z, 5)
        z_range5 = z_tail5.max(axis=1) - z_tail5.min(axis=1)
        z_range_all = z.max(axis=1) - z.min(axis=1)
        z_last = z[:, -1, :]
        z_mean5 = z_tail5.mean(axis=1)
        z_std5 = z_tail5.std(axis=1)
        t = np.arange(z.shape[1], dtype=np.float64)
        t = (t - t.mean()) / (t.std() + 1e-08)
        slope = (z[:, :, 0] * t[None, :]).mean(axis=1, keepdims=True)
        abs_dz_sum5 = _tail(np.abs(np.diff(z, axis=1)), 5).sum(axis=1)
        sign_vz = np.sign(vz_last)
        return np.concatenate([z_last, z_mean5, z_std5, dz1, dz3, dz5, dz10, vz_last, vz_mean5, vz_std5, az_last, z_range5, z_range_all, slope, abs_dz_sum5, sign_vz], axis=1)

    def make_axis_features(data, norm, branch, dt=0.1):
        """
        Common features are the original V2 Hankel features.
        Extra features are branch-specific.
        """
        common = bk.make_hankel_features(data, norm)
        if branch == 'xy':
            extra = planar_extra_features(data, dt)
        elif branch == 'z':
            extra = vertical_extra_features(data, dt)
        else:
            raise ValueError(f'Unknown branch: {branch}')
        return np.concatenate([common, extra], axis=1)

    def full_target_pos_std(data, norm):
        n = data.future_global.shape[0]
        h = data.future_global.shape[1]
        yfull = bk.full_target_std(data, norm).reshape(n, h, 6)
        return yfull[:, :, :3]

    def target_xy_std(data, norm):
        y = full_target_pos_std(data, norm)
        return y[:, :, :2].reshape(y.shape[0], -1)

    def target_z_std(data, norm):
        y = full_target_pos_std(data, norm)
        return y[:, :, 2].reshape(y.shape[0], -1)

    def pos_std_to_global(pos_std, data, norm):
        n, h, _ = pos_std.shape
        full = np.zeros((n, h, 6), dtype=np.float64)
        full[:, :, :3] = pos_std
        return bk.std_full_to_global(full.reshape(n, h * 6), data, norm)

    class AxisFeatureDecoupledBehaviorKoopman:

        def __init__(self, H, norm, fstd_xy, fstd_z, modes, W_xy, W_z, R_xy, R_z, dt=0.1):
            self.H = H
            self.norm = norm
            self.fstd_xy = fstd_xy
            self.fstd_z = fstd_z
            self.modes = list(modes)
            self.mode_to_idx = {m: i for i, m in enumerate(self.modes)}
            self.W_xy = W_xy
            self.W_z = W_z
            self.R_xy = dict(R_xy)
            self.R_z = dict(R_z)
            self.dt = float(dt)

        def Xxy(self, data):
            return bk.add_bias(self.fstd_xy.transform(make_axis_features(data, self.norm, 'xy', self.dt)))

        def Xz(self, data):
            return bk.add_bias(self.fstd_z.transform(make_axis_features(data, self.norm, 'z', self.dt)))

        def base_pos_std(self, data):
            xxy = self.Xxy(data)
            xz = self.Xz(data)
            n = xxy.shape[0]
            xy = (xxy @ self.W_xy.T).reshape(n, self.H, 2)
            z = (xz @ self.W_z.T).reshape(n, self.H, 1)
            return np.concatenate([xy, z], axis=-1)

        def mode_pos_std(self, data, mode):
            xxy = self.Xxy(data)
            xz = self.Xz(data)
            n = xxy.shape[0]
            xy = (xxy @ self.W_xy.T).reshape(n, self.H, 2)
            z = (xz @ self.W_z.T).reshape(n, self.H, 1)
            if mode in self.R_xy and self.R_xy[mode] is not None:
                xy = xy + (xxy @ self.R_xy[mode].T).reshape(n, self.H, 2)
            if mode in self.R_z and self.R_z[mode] is not None:
                z = z + (xz @ self.R_z[mode].T).reshape(n, self.H, 1)
            return np.concatenate([xy, z], axis=-1)

        def stack_candidates_pos_std(self, data):
            return np.stack([self.mode_pos_std(data, m) for m in self.modes], axis=1)

        def hard_indices(self, data):
            idx = np.zeros(len(data.behavior_key), dtype=np.int64)
            for i, b in enumerate(data.behavior_key):
                idx[i] = self.mode_to_idx.get(str(b), 0)
            return idx

        def hard_pos_std(self, data):
            cand = self.stack_candidates_pos_std(data)
            idx = self.hard_indices(data)
            return cand[np.arange(cand.shape[0]), idx]

        def oracle_pos_std(self, data):
            cand = self.stack_candidates_pos_std(data)
            return oracle_select_pos_std(cand, data, self.norm)

        def hmm_onehot_pos_std(self, data, T_behavior, steps):
            cand = self.stack_candidates_pos_std(data)
            idx_current = self.hard_indices(data)
            p = np.zeros((len(idx_current), len(self.modes)), dtype=np.float64)
            p[np.arange(len(idx_current)), idx_current] = 1.0
            ph = p @ bk.transition_power(T_behavior, steps)
            idx = np.argmax(ph, axis=1)
            return cand[np.arange(cand.shape[0]), idx]

    def oracle_select_pos_std(cand_pos_std, data, norm):
        n, b, h, _ = cand_pos_std.shape
        best_idx = np.zeros(n, dtype=np.int64)
        best_fde = None
        for j in range(b):
            pred_g = pos_std_to_global(cand_pos_std[:, j], data, norm)
            fde = np.linalg.norm(pred_g[:, -1, :3] - data.future_global[:, -1, :3], axis=-1)
            if best_fde is None:
                best_fde = fde
                best_idx[:] = j
            else:
                mask = fde < best_fde
                best_fde[mask] = fde[mask]
                best_idx[mask] = j
        return cand_pos_std[np.arange(n), best_idx]

    def fit_global(train, val, norm, fstd_xy, fstd_z, ridge_xy_grid, ridge_z_grid, dt, selection_metric, ade_w, fde_w):
        Xxy_tr = bk.add_bias(fstd_xy.transform(make_axis_features(train, norm, 'xy', dt)))
        Xz_tr = bk.add_bias(fstd_z.transform(make_axis_features(train, norm, 'z', dt)))
        Xxy_va = bk.add_bias(fstd_xy.transform(make_axis_features(val, norm, 'xy', dt)))
        Xz_va = bk.add_bias(fstd_z.transform(make_axis_features(val, norm, 'z', dt)))
        yxy_tr = target_xy_std(train, norm)
        yz_tr = target_z_std(train, norm)
        h = train.future_global.shape[1]
        best, best_pack, rows = (None, None, [])
        for rxy in ridge_xy_grid:
            Wxy = bk.ridge_fit(Xxy_tr, yxy_tr, rxy)
            xy_va = Xxy_va @ Wxy.T
            for rz in ridge_z_grid:
                Wz = bk.ridge_fit(Xz_tr, yz_tr, rz)
                z_va = Xz_va @ Wz.T
                nva = Xxy_va.shape[0]
                pos = np.zeros((nva, h, 3), dtype=np.float64)
                pos[:, :, :2] = xy_va.reshape(nva, h, 2)
                pos[:, :, 2] = z_va.reshape(nva, h)
                pred = pos_std_to_global(pos, val, norm)
                met = compute_position_metrics(val.future_global, pred)
                sc = score_pos_metrics(met, selection_metric, ade_w, fde_w)
                row = {'method_group': 'axis_feature_global_posonly', 'ridge_xy': rxy, 'ridge_z': rz, 'score': sc, **met}
                rows.append(row)
                if best is None or sc < best['score']:
                    best = row
                    best_pack = (Wxy, Wz)
        return (best_pack[0], best_pack[1], best, rows)

    def fit_residuals(train, norm, fstd_xy, fstd_z, Wxy, Wz, modes, ridge_xy, ridge_z, dt, min_behavior_samples, operator_mode='shared_residual'):
        """Fit mode-dependent direct multi-step / condensed operators.

        shared_residual reproduces the original implementation:
            E_m = E_global + DeltaE_m, with DeltaE_m fitted on residual targets.

        independent fits each complete mode operator E_m directly on samples assigned
        to that mode.  For backward-compatible downstream inference, it stores
        R_m = E_m - E_global, so the legacy evaluation E_global + R_m equals E_m.
        """
        if operator_mode not in {'shared_residual', 'independent'}:
            raise ValueError(f'Unknown condensed operator mode: {operator_mode}')
        Xxy = bk.add_bias(fstd_xy.transform(make_axis_features(train, norm, 'xy', dt)))
        Xz = bk.add_bias(fstd_z.transform(make_axis_features(train, norm, 'z', dt)))
        yxy = target_xy_std(train, norm)
        yz = target_z_std(train, norm)
        base_xy = Xxy @ Wxy.T
        base_z = Xz @ Wz.T
        behavior = np.asarray(train.behavior_key).astype(str)
        Rxy, Rz = ({}, {})
        for m in modes:
            if m == 'global':
                Rxy[m] = None
                Rz[m] = None
                continue
            idx = np.where(behavior == m)[0]
            if len(idx) < min_behavior_samples:
                Rxy[m] = None
                Rz[m] = None
                continue
            if operator_mode == 'shared_residual':
                Rxy[m] = bk.ridge_fit(Xxy[idx], yxy[idx] - base_xy[idx], ridge_xy)
                Rz[m] = bk.ridge_fit(Xz[idx], yz[idx] - base_z[idx], ridge_z)
            else:
                Wxy_m = bk.ridge_fit(Xxy[idx], yxy[idx], ridge_xy)
                Wz_m = bk.ridge_fit(Xz[idx], yz[idx], ridge_z)
                Rxy[m] = Wxy_m - Wxy
                Rz[m] = Wz_m - Wz
        return (Rxy, Rz)

    def fit_behavior_model(train, val, norm, fstd_xy, fstd_z, Wxy, Wz, modes, ridge_xy_grid, ridge_z_grid, dt, min_behavior_samples, candidate_selection, selection_metric, ade_w, fde_w, operator_mode='shared_residual'):
        best, best_model, rows = (None, None, [])
        for rxy in ridge_xy_grid:
            for rz in ridge_z_grid:
                Rxy, Rz = fit_residuals(train, norm, fstd_xy, fstd_z, Wxy, Wz, modes, rxy, rz, dt, min_behavior_samples, operator_mode=operator_mode)
                model = AxisFeatureDecoupledBehaviorKoopman(train.future_global.shape[1], norm, fstd_xy, fstd_z, modes, Wxy, Wz, Rxy, Rz, dt=dt)
                if candidate_selection == 'oracle':
                    pos_va = model.oracle_pos_std(val)
                    group = 'axis_feature_behavior_oracle_val'
                else:
                    pos_va = model.hard_pos_std(val)
                    group = 'axis_feature_behavior_hard_val'
                pred_va = pos_std_to_global(pos_va, val, norm)
                met = compute_position_metrics(val.future_global, pred_va)
                sc = score_pos_metrics(met, selection_metric, ade_w, fde_w)
                row = {'method_group': group, 'operator_mode': operator_mode, 'behavior_ridge_xy': rxy, 'behavior_ridge_z': rz, 'score': sc, **met}
                rows.append(row)
                if best is None or sc < best['score']:
                    best = row
                    best_model = model
        return (best_model, best, rows)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument('--data_root', type=str, required=True)
        ap.add_argument('--out_dir', type=str, default=None)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizons', type=str, default='1,3,5,10')
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--ridge_xy_grid', type=str, default='1e-3,3e-3,1e-2,3e-2,1e-1')
        ap.add_argument('--ridge_z_grid', type=str, default='1e-3,3e-3,1e-2,3e-2,1e-1')
        ap.add_argument('--behavior_ridge_xy_grid', type=str, default='1e-3,1e-2,1e-1,1,3')
        ap.add_argument('--behavior_ridge_z_grid', type=str, default='1e-3,1e-2,1e-1,1,3')
        ap.add_argument('--min_behavior_samples', type=int, default=40)
        ap.add_argument('--hmm_smooth', type=float, default=1.0)
        ap.add_argument('--candidate_selection', type=str, default='hard', choices=['hard', 'oracle'])
        ap.add_argument('--selection_metric', type=str, default='combo', choices=['pos_rmse', 'ade', 'fde', 'fde_xy', 'fde_z', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--fast', action='store_true')
        args = ap.parse_args()
        root = Path(args.data_root)
        out_dir = Path(args.out_dir) if args.out_dir else root / 'axis_feature_decoupled_position_behavior_koopman_v5'
        ensure_dir(out_dir)
        horizons = parse_int_list(args.horizons)
        if args.fast:
            horizons = [10]
        ridge_xy_grid = parse_float_list(args.ridge_xy_grid)
        ridge_z_grid = parse_float_list(args.ridge_z_grid)
        behavior_ridge_xy_grid = parse_float_list(args.behavior_ridge_xy_grid)
        behavior_ridge_z_grid = parse_float_list(args.behavior_ridge_z_grid)
        train_chains = bk.load_split_chains(root, 'train')
        val_chains = bk.load_split_chains(root, 'val')
        test_chains = bk.load_split_chains(root, 'test')
        compare_rows, sample_dfs, sweep_rows, selected = ([], [], [], {})
        for H in horizons:
            print(f'\n=== Axis-Feature-Decoupled Position-only Behavior Koopman V5 | H={H} ===')
            train = bk.make_split_data(train_chains, args.input_len, H, args.stride)
            val = bk.make_split_data(val_chains, args.input_len, H, args.stride)
            test = bk.make_split_data(test_chains, args.input_len, H, args.stride)
            print(f'samples: train={train.hist_global.shape[0]}, val={val.hist_global.shape[0]}, test={test.hist_global.shape[0]}')
            norm = bk.fit_state_normalizer(train)
            Fxy_tr = make_axis_features(train, norm, 'xy', args.dt)
            Fz_tr = make_axis_features(train, norm, 'z', args.dt)
            fstd_xy = bk.FeatureStandardizer.fit(Fxy_tr)
            fstd_z = bk.FeatureStandardizer.fit(Fz_tr)
            counts = pd.Series(train.behavior_key).value_counts()
            eligible = [b for b, c in counts.items() if b != 'global' and c >= args.min_behavior_samples]
            modes = ['global'] + sorted(eligible)
            print('behavior modes:', modes)
            print('behavior counts:\n' + counts.to_string())
            Wxy, Wz, best_global, global_sweep = fit_global(train, val, norm, fstd_xy, fstd_z, ridge_xy_grid, ridge_z_grid, args.dt, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight)
            for r in global_sweep:
                r['horizon'] = H
                sweep_rows.append(r)
            print('selected axis-feature global:', best_global)
            model, best_behavior, behavior_sweep = fit_behavior_model(train, val, norm, fstd_xy, fstd_z, Wxy, Wz, modes, behavior_ridge_xy_grid, behavior_ridge_z_grid, args.dt, args.min_behavior_samples, args.candidate_selection, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight)
            for r in behavior_sweep:
                r['horizon'] = H
                sweep_rows.append(r)
            print('selected axis-feature behavior:', best_behavior)
            T_behavior = bk.estimate_behavior_transition(train_chains, modes, smooth=args.hmm_smooth)
            for name, pred in [('constant_velocity', bk.cv_predict(test, args.dt)), ('constant_acceleration', bk.ca_predict(test, args.dt))]:
                met, sdf = eval_position_array(test, pred, name)
                compare_rows.append({'method': name, 'horizon': H, **met})
                sample_dfs.append(sdf)
            pred_global = pos_std_to_global(model.base_pos_std(test), test, norm)
            met, sdf = eval_position_array(test, pred_global, 'axis_feature_global_posonly')
            compare_rows.append({'method': 'axis_feature_global_posonly', 'horizon': H, **met})
            sample_dfs.append(sdf)
            pred_hard = pos_std_to_global(model.hard_pos_std(test), test, norm)
            met, sdf = eval_position_array(test, pred_hard, 'axis_feature_hard_behavior_posonly')
            compare_rows.append({'method': 'axis_feature_hard_behavior_posonly', 'horizon': H, **met})
            sample_dfs.append(sdf)
            pred_hmm = pos_std_to_global(model.hmm_onehot_pos_std(test, T_behavior, steps=H), test, norm)
            met, sdf = eval_position_array(test, pred_hmm, 'axis_feature_hmm_onehot_behavior_posonly')
            compare_rows.append({'method': 'axis_feature_hmm_onehot_behavior_posonly', 'horizon': H, **met})
            sample_dfs.append(sdf)
            pred_oracle = pos_std_to_global(model.oracle_pos_std(test), test, norm)
            met, sdf = eval_position_array(test, pred_oracle, 'axis_feature_oracle_behavior_posonly')
            compare_rows.append({'method': 'axis_feature_oracle_behavior_posonly', 'horizon': H, **met})
            sample_dfs.append(sdf)
            selected[f'H{H}'] = {'modes': modes, 'behavior_counts': counts.to_dict(), 'best_global': best_global, 'best_behavior': best_behavior, 'transition_matrix': T_behavior.tolist()}
        compare_df = pd.DataFrame(compare_rows)
        sample_df = pd.concat(sample_dfs, ignore_index=True)
        sweep_df = pd.DataFrame(sweep_rows)
        compare_df.to_csv(out_dir / 'compare_horizon_position_metrics_test.csv', index=False)
        sample_df.to_csv(out_dir / 'sample_position_metrics_test.csv', index=False)
        sweep_df.to_csv(out_dir / 'model_sweep_position_val.csv', index=False)
        group_position_metrics(sample_df, 'scene_type').to_csv(out_dir / 'grouped_position_metrics_by_scene_test.csv', index=False)
        group_position_metrics(sample_df, 'dominant_behavior').to_csv(out_dir / 'grouped_position_metrics_by_behavior_test.csv', index=False)
        group_position_metrics(sample_df, 'behavior_key').to_csv(out_dir / 'grouped_position_metrics_by_behavior_key_test.csv', index=False)
        report = {'version': 'axis_feature_decoupled_position_behavior_koopman_v5', 'data_root': str(root), 'out_dir': str(out_dir), 'input_len': args.input_len, 'horizons': horizons, 'stride': args.stride, 'position_only': True, 'axis_decoupled': True, 'axis_specific_features': True, 'selected': selected, 'notes': ['Input still uses historical states x,y,z,vx,vy,vz.', 'Targets are position-only.', 'XY branch uses original Hankel features plus planar motion features.', 'Z branch uses original Hankel features plus vertical motion features.', 'Behavior residuals are separately fitted for XY and Z branches.']}
        with open(out_dir / 'report_axis_feature_decoupled_position_behavior_koopman_v5.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('\n=== Axis-Feature-Decoupled Position-only Behavior Koopman V5 finished ===')
        print(compare_df.to_string(index=False))
        print(f'\nOutput directory: {out_dir}')

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

axis_feature_decoupled_position_behavior_koopman_v5 = _init_axis_feature_decoupled_position_behavior_koopman_v5()


# ========================================================================================
# SECTION: UNSUP SWITCH SELECTOR ENHANCE STANDALONE V2
# ========================================================================================
def _init_unsup_switch_selector_enhance_standalone_v2():
    """
    unsup_switch_selector_enhance_standalone.py

    Self-contained enhanced selector experiments for unsupervised Switching Koopman.

    This version does NOT import unsupervised_switching_koopman_no_action.py.
    It only depends on:
        behavior_mode_endpoint_koopman_v2_hmm_selector.py
        axis_feature_decoupled_position_behavior_koopman_v5.py

    Goal:
        Improve no-action-label unsupervised switching Koopman by improving latent-mode
        selection / fusion, without using action labels.

    Tested methods:
        v5_no_action_global
        unsup_switch_linear_gate
        selector_global_switch_alpha
        selector_confidence_fallback
        selector_soft_mixture
        selector_global_soft_fusion
        selector_residual_aware_gate
        unsup_switch_oracle

    Fast:
        python unsup_switch_selector_enhance_standalone.py --data_root DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed --fast --out_dir DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed/unsup_switch_selector_enhance_standalone_fast
    """
    import argparse
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    bk = behavior_mode_endpoint_koopman_v2_hmm_selector
    v5 = axis_feature_decoupled_position_behavior_koopman_v5

    def ensure_dir(p):
        Path(p).mkdir(parents=True, exist_ok=True)

    def parse_int_list(s):
        return [int(x.strip()) for x in str(s).split(',') if x.strip()]

    def parse_float_list(s):
        return [float(x.strip()) for x in str(s).split(',') if x.strip()]

    def ridge_fit(X, Y, ridge):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        D = X.shape[1]
        A = X.T @ X + float(ridge) * np.eye(D, dtype=np.float64)
        B = X.T @ Y
        return np.linalg.solve(A, B).T

    def softmax_np(x, axis=1, temp=1.0):
        x = np.asarray(x, dtype=np.float64) / max(float(temp), 1e-08)
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)

    class Standardizer:

        def __init__(self, mean, std):
            self.mean = np.asarray(mean, dtype=np.float64)
            self.std = np.asarray(std, dtype=np.float64)

        @classmethod
        def fit(cls, X, eps=1e-08):
            X = np.asarray(X, dtype=np.float64)
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std = np.maximum(std, eps)
            return cls(mean, std)

        def transform(self, X):
            return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def compute_position_metrics(y_true, y_pred):
        yt = y_true[:, :, :3]
        yp = y_pred[:, :, :3]
        diff = yp - yt
        step_err = np.linalg.norm(diff, axis=-1)
        step_err_xy = np.linalg.norm(diff[:, :, :2], axis=-1)
        step_err_z = np.abs(diff[:, :, 2])
        final_err = step_err[:, -1]
        final_err_xy = step_err_xy[:, -1]
        final_err_z = step_err_z[:, -1]
        return {'pos_rmse': float(np.sqrt(np.mean(diff ** 2))), 'pos_mae': float(np.mean(np.abs(diff))), 'ade': float(np.mean(step_err)), 'fde': float(np.mean(final_err)), 'ade_xy': float(np.mean(step_err_xy)), 'fde_xy': float(np.mean(final_err_xy)), 'ade_z': float(np.mean(step_err_z)), 'fde_z': float(np.mean(final_err_z)), 'z_ratio_fde': float(np.mean(final_err_z / (final_err + 1e-12))), 'z_ratio_ade': float(np.mean(step_err_z.mean(axis=1) / (step_err.mean(axis=1) + 1e-12)))}

    def metric_score(met, args):
        if args.selection_metric == 'pos_rmse':
            return met['pos_rmse']
        if args.selection_metric == 'ade':
            return met['ade']
        if args.selection_metric == 'fde':
            return met['fde']
        if args.selection_metric == 'fde_xy':
            return met['fde_xy']
        if args.selection_metric == 'fde_z':
            return met['fde_z']
        return met['pos_rmse'] + args.combo_ade_weight * met['ade'] + args.combo_fde_weight * met['fde']

    def make_sample_df(data, pred_pos, method):
        y_true = data.future_global[:, :, :3]
        y_pred = pred_pos[:, :, :3]
        diff = y_pred - y_true
        step_err = np.linalg.norm(diff, axis=-1)
        step_err_xy = np.linalg.norm(diff[:, :, :2], axis=-1)
        step_err_z = np.abs(diff[:, :, 2])
        n = y_true.shape[0]

        def safe_attr(name, default):
            if hasattr(data, name):
                vals = getattr(data, name)
                if len(vals) == n:
                    return vals
            return [default] * n
        df = pd.DataFrame({'method': method, 'sample_idx': np.arange(n, dtype=int), 'chain_file': safe_attr('chain_file', ''), 'scene_type': safe_attr('scene_type', ''), 'dominant_behavior': safe_attr('dominant_behavior', ''), 'behavior_key': safe_attr('behavior_key', ''), 'ade': step_err.mean(axis=1), 'fde': step_err[:, -1], 'ade_xy': step_err_xy.mean(axis=1), 'fde_xy': step_err_xy[:, -1], 'ade_z': step_err_z.mean(axis=1), 'fde_z': step_err_z[:, -1]})
        df['z_ratio_fde'] = df['fde_z'] / (df['fde'] + 1e-12)
        df['z_ratio_ade'] = df['ade_z'] / (df['ade'] + 1e-12)
        return df

    def group_metrics(sample_df, group_col):
        if group_col not in sample_df.columns:
            return pd.DataFrame()
        rows = []
        for (method, group), sub in sample_df.groupby(['method', group_col], dropna=False):
            rows.append({'method': method, group_col: group, 'num_samples': len(sub), 'ade_mean': sub['ade'].mean(), 'fde_mean': sub['fde'].mean(), 'ade_xy_mean': sub['ade_xy'].mean(), 'fde_xy_mean': sub['fde_xy'].mean(), 'ade_z_mean': sub['ade_z'].mean(), 'fde_z_mean': sub['fde_z'].mean(), 'z_ratio_fde_mean': sub['z_ratio_fde'].mean(), 'fde_p90': sub['fde'].quantile(0.9), 'fde_z_p90': sub['fde_z'].quantile(0.9)})
        return pd.DataFrame(rows).sort_values([group_col, 'fde_mean']).reset_index(drop=True)

    def fit_no_action_global(train, val, args):
        norm = bk.fit_state_normalizer(train)
        Fxy_tr = v5.make_axis_features(train, norm, 'xy', args.dt)
        Fz_tr = v5.make_axis_features(train, norm, 'z', args.dt)
        fstd_xy = bk.FeatureStandardizer.fit(Fxy_tr)
        fstd_z = bk.FeatureStandardizer.fit(Fz_tr)
        Wxy, Wz, best, rows = v5.fit_global(train=train, val=val, norm=norm, fstd_xy=fstd_xy, fstd_z=fstd_z, ridge_xy_grid=parse_float_list(args.global_ridge_xy_grid), ridge_z_grid=parse_float_list(args.global_ridge_z_grid), dt=args.dt, selection_metric=args.selection_metric, ade_w=args.combo_ade_weight, fde_w=args.combo_fde_weight)
        return {'norm': norm, 'fstd_xy': fstd_xy, 'fstd_z': fstd_z, 'Wxy': Wxy, 'Wz': Wz, 'best': best, 'rows': rows}

    def predict_no_action_global(pack, data, args):
        norm = pack['norm']
        Fxy = v5.make_axis_features(data, norm, 'xy', args.dt)
        Fz = v5.make_axis_features(data, norm, 'z', args.dt)
        Xxy = bk.add_bias(pack['fstd_xy'].transform(Fxy))
        Xz = bk.add_bias(pack['fstd_z'].transform(Fz))
        n = data.future_global.shape[0]
        H = data.future_global.shape[1]
        xy = (Xxy @ pack['Wxy'].T).reshape(n, H, 2)
        z = (Xz @ pack['Wz'].T).reshape(n, H, 1)
        pos_std = np.concatenate([xy, z], axis=-1)
        return v5.pos_std_to_global(pos_std, data, norm)[:, :, :3]

    def unwrap_angle_diff(theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    def make_motion_signature(data, dt=0.1):
        hist = np.asarray(data.hist_global, dtype=np.float64)
        p = hist[:, :, :3]
        v = hist[:, :, 3:6]
        n, L, _ = hist.shape
        vxy = v[:, :, :2]
        vz = v[:, :, 2]
        speed_xy = np.linalg.norm(vxy, axis=-1)
        speed_3d = np.linalg.norm(v, axis=-1)
        heading = np.arctan2(vxy[:, :, 1], vxy[:, :, 0] + 1e-12)
        dtheta = unwrap_angle_diff(np.diff(heading, axis=1)) if L >= 2 else np.zeros((n, 1), dtype=np.float64)
        feats = [v[:, -1, :], speed_xy[:, -1:], speed_3d[:, -1:], vz[:, -1:]]
        for lag in [1, 2, 3, 5, 8, 10, min(15, L - 1)]:
            if lag <= 0:
                continue
            lag = min(lag, L - 1)
            dp = p[:, -1, :] - p[:, -1 - lag, :]
            dv = v[:, -1, :] - v[:, -1 - lag, :]
            feats += [dp, dv, dp / max(lag * dt, 1e-08), dv / max(lag * dt, 1e-08), ((speed_3d[:, -1] - speed_3d[:, -1 - lag]) / max(lag * dt, 1e-08))[:, None], ((p[:, -1, 2] - p[:, -1 - lag, 2]) / max(lag * dt, 1e-08))[:, None]]
        for k in [3, 5, 10, min(20, L)]:
            k = min(k, L)
            feats += [speed_xy[:, -k:].mean(axis=1, keepdims=True), speed_xy[:, -k:].std(axis=1, keepdims=True), speed_3d[:, -k:].mean(axis=1, keepdims=True), speed_3d[:, -k:].std(axis=1, keepdims=True), vz[:, -k:].mean(axis=1, keepdims=True), vz[:, -k:].std(axis=1, keepdims=True)]
        if dtheta.shape[1] > 0:
            for k in [3, 5, 10, min(19, dtheta.shape[1])]:
                k = min(k, dtheta.shape[1])
                if k <= 0:
                    continue
                feats += [dtheta[:, -k:].sum(axis=1, keepdims=True), np.abs(dtheta[:, -k:]).sum(axis=1, keepdims=True), dtheta[:, -k:].mean(axis=1, keepdims=True) / max(dt, 1e-08)]
        if L >= 2:
            step = np.diff(p, axis=1)
            step_norm = np.linalg.norm(step, axis=-1)
            path_len = step_norm.sum(axis=1, keepdims=True)
            net = np.linalg.norm(p[:, -1, :] - p[:, 0, :], axis=1, keepdims=True)
            feats += [path_len, net, net / (path_len + 1e-08)]
        X = np.concatenate(feats, axis=1)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def kmeans_np(X, K, seed=42, max_iter=50):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        n = X.shape[0]
        centers = np.empty((K, X.shape[1]), dtype=np.float64)
        first = rng.integers(0, n)
        centers[0] = X[first]
        dist2 = np.sum((X - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            probs = dist2 / max(dist2.sum(), 1e-12)
            idx = rng.choice(n, p=probs)
            centers[k] = X[idx]
            dist2 = np.minimum(dist2, np.sum((X - centers[k]) ** 2, axis=1))
        assign = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_assign = d.argmin(axis=1)
            if np.array_equal(assign, new_assign):
                break
            assign = new_assign
            for k in range(K):
                idx = assign == k
                centers[k] = X[idx].mean(axis=0) if np.any(idx) else X[rng.integers(0, n)]
        return (assign, centers)

    def full_target_pos_std(data, norm):
        return v5.full_target_pos_std(data, norm)[:, :, :3]

    def fit_latent_residuals(Xxy, Xz, yxy, yz, base_xy, base_z, assign, K, ridge_model, min_cluster_samples, Wxy_g=None, Wz_g=None, operator_mode='shared_residual'):
        """Fit latent mode-dependent condensed operators with legacy-compatible storage.

        independent: fit the complete finite-horizon operator W_k directly for each
        latent cluster, then store R_k = W_k - W_global so existing candidate code
        evaluates exactly W_k.
        """
        if operator_mode not in {'shared_residual', 'independent'}:
            raise ValueError(f'Unknown condensed operator mode: {operator_mode}')
        if operator_mode == 'independent' and (Wxy_g is None or Wz_g is None):
            raise ValueError('independent operator mode requires Wxy_g and Wz_g')
        Rxy, Rz = ([], [])
        for k in range(K):
            idx = np.where(assign == k)[0]
            if len(idx) < min_cluster_samples:
                Rxy.append(None)
                Rz.append(None)
            elif operator_mode == 'shared_residual':
                Rxy.append(ridge_fit(Xxy[idx], yxy[idx] - base_xy[idx], ridge_model))
                Rz.append(ridge_fit(Xz[idx], yz[idx] - base_z[idx], ridge_model))
            else:
                Wxy_k = ridge_fit(Xxy[idx], yxy[idx], ridge_model)
                Wz_k = ridge_fit(Xz[idx], yz[idx], ridge_model)
                Rxy.append(Wxy_k - Wxy_g)
                Rz.append(Wz_k - Wz_g)
        return (Rxy, Rz)

    def latent_candidates_from_arrays(Xxy, Xz, H, Wxy_g, Wz_g, Rxy, Rz):
        n = Xxy.shape[0]
        K = len(Rxy)
        base_xy = Xxy @ Wxy_g.T
        base_z = Xz @ Wz_g.T
        cands = np.zeros((n, K, H, 3), dtype=np.float64)
        for k in range(K):
            xy = base_xy if Rxy[k] is None else base_xy + Xxy @ Rxy[k].T
            z = base_z if Rz[k] is None else base_z + Xz @ Rz[k].T
            cands[:, k, :, :2] = xy.reshape(n, H, 2)
            cands[:, k, :, 2] = z.reshape(n, H)
        return cands

    def assignment_error(cands, y_pos_std):
        diff = cands - y_pos_std[:, None, :, :]
        return np.sqrt(np.mean(diff ** 2, axis=(2, 3)))

    def reinitialize_empty_clusters(assign, err, K, min_cluster_samples):
        counts = np.bincount(assign, minlength=K)
        if np.all(counts >= min_cluster_samples):
            return assign
        best_err = err[np.arange(err.shape[0]), assign]
        order = np.argsort(-best_err)
        used = set()
        for k in range(K):
            if counts[k] >= min_cluster_samples:
                continue
            need = min_cluster_samples - counts[k]
            chosen = []
            for idx in order:
                if idx in used:
                    continue
                old = assign[idx]
                if counts[old] <= min_cluster_samples:
                    continue
                chosen.append(idx)
                used.add(idx)
                if len(chosen) >= need:
                    break
            if chosen:
                chosen = np.asarray(chosen, dtype=int)
                old_vals = assign[chosen]
                for old in old_vals:
                    counts[old] -= 1
                assign[chosen] = k
                counts[k] += len(chosen)
        return assign

    class UnsupervisedSwitchingKoopman:

        def __init__(self, K, norm, fstd_xy, fstd_z, gate_std, Wxy_g, Wz_g, Rxy, Rz, centroids, Wgate, dt):
            self.K = int(K)
            self.norm = norm
            self.fstd_xy = fstd_xy
            self.fstd_z = fstd_z
            self.gate_std = gate_std
            self.Wxy_g = Wxy_g
            self.Wz_g = Wz_g
            self.Rxy = list(Rxy)
            self.Rz = list(Rz)
            self.centroids = np.asarray(centroids, dtype=np.float64)
            self.Wgate = Wgate
            self.dt = float(dt)

        def model_features(self, data):
            Fxy = v5.make_axis_features(data, self.norm, 'xy', self.dt)
            Fz = v5.make_axis_features(data, self.norm, 'z', self.dt)
            return (bk.add_bias(self.fstd_xy.transform(Fxy)), bk.add_bias(self.fstd_z.transform(Fz)))

        def gate_features(self, data):
            return self.gate_std.transform(make_motion_signature(data, self.dt))

        def candidates_pos_std(self, data):
            Xxy, Xz = self.model_features(data)
            H = data.future_global.shape[1]
            return latent_candidates_from_arrays(Xxy, Xz, H, self.Wxy_g, self.Wz_g, self.Rxy, self.Rz)

    def fit_switching_model(train, K, ridge_model, ridge_gate, args, seed=42):
        norm = bk.fit_state_normalizer(train)
        Fxy_tr = v5.make_axis_features(train, norm, 'xy', args.dt)
        Fz_tr = v5.make_axis_features(train, norm, 'z', args.dt)
        fstd_xy = bk.FeatureStandardizer.fit(Fxy_tr)
        fstd_z = bk.FeatureStandardizer.fit(Fz_tr)
        Xxy_tr = bk.add_bias(fstd_xy.transform(Fxy_tr))
        Xz_tr = bk.add_bias(fstd_z.transform(Fz_tr))
        yxy_tr = v5.target_xy_std(train, norm)
        yz_tr = v5.target_z_std(train, norm)
        y_pos_tr = full_target_pos_std(train, norm)
        H = train.future_global.shape[1]
        Wxy_g = ridge_fit(Xxy_tr, yxy_tr, ridge_model)
        Wz_g = ridge_fit(Xz_tr, yz_tr, ridge_model)
        base_xy = Xxy_tr @ Wxy_g.T
        base_z = Xz_tr @ Wz_g.T
        G_raw = make_motion_signature(train, args.dt)
        gate_std = Standardizer.fit(G_raw)
        G = gate_std.transform(G_raw)
        assign, _ = kmeans_np(G, K, seed=seed, max_iter=args.kmeans_iter)
        operator_mode = getattr(args, 'condensed_operator_mode', 'shared_residual')
        if operator_mode not in {'shared_residual', 'independent'}:
            raise ValueError(f'Unknown --condensed_operator_mode={operator_mode}')
        print(f'latent condensed operator parameterization: {operator_mode}')
        history = []
        for it in range(args.alt_iter):
            Rxy, Rz = fit_latent_residuals(Xxy_tr, Xz_tr, yxy_tr, yz_tr, base_xy, base_z, assign, K, ridge_model, args.min_cluster_samples, Wxy_g=Wxy_g, Wz_g=Wz_g, operator_mode=operator_mode)
            cands = latent_candidates_from_arrays(Xxy_tr, Xz_tr, H, Wxy_g, Wz_g, Rxy, Rz)
            err = assignment_error(cands, y_pos_tr)
            new_assign = err.argmin(axis=1)
            new_assign = reinitialize_empty_clusters(new_assign, err, K, args.min_cluster_samples)
            counts = np.bincount(new_assign, minlength=K)
            obj = float(np.mean(err[np.arange(err.shape[0]), new_assign]))
            changes = int(np.sum(new_assign != assign))
            history.append({'iter': it + 1, 'operator_mode': operator_mode, 'objective': obj, 'changes': changes, **{f'count_{k}': int(counts[k]) for k in range(K)}})
            assign = new_assign
            if changes == 0:
                break
        Rxy, Rz = fit_latent_residuals(Xxy_tr, Xz_tr, yxy_tr, yz_tr, base_xy, base_z, assign, K, ridge_model, args.min_cluster_samples, Wxy_g=Wxy_g, Wz_g=Wz_g, operator_mode=operator_mode)
        centroids = np.zeros((K, G.shape[1]), dtype=np.float64)
        for k in range(K):
            idx = assign == k
            centroids[k] = G[idx].mean(axis=0) if np.any(idx) else G.mean(axis=0)
        Ygate = np.zeros((len(assign), K), dtype=np.float64)
        Ygate[np.arange(len(assign)), assign] = 1.0
        Wgate = ridge_fit(bk.add_bias(G), Ygate, ridge_gate)
        model = UnsupervisedSwitchingKoopman(K, norm, fstd_xy, fstd_z, gate_std, Wxy_g, Wz_g, Rxy, Rz, centroids, Wgate, args.dt)
        return (model, assign, pd.DataFrame(history))

    def candidates_global(model, data):
        cand_std = model.candidates_pos_std(data)
        n, K, H, _ = cand_std.shape
        flat = cand_std.reshape(n * K, H, 3)

        class Obj:
            pass
        rep = Obj()
        rep.hist_global = np.repeat(data.hist_global, K, axis=0)
        rep.future_global = np.repeat(data.future_global, K, axis=0)
        rep.horizon = H
        rep.input_len = data.hist_global.shape[1]
        if hasattr(data, 'hist_local'):
            rep.hist_local = np.repeat(data.hist_local, K, axis=0)
        else:
            rep.hist_local = np.repeat(data.hist_global, K, axis=0)
        if hasattr(data, 'future_local'):
            rep.future_local = np.repeat(data.future_local, K, axis=0)
        else:
            rep.future_local = np.repeat(data.future_global, K, axis=0)
        rep.frames = bk.LocalFrameBatch(origins=np.repeat(data.frames.origins, K, axis=0), c=np.repeat(data.frames.c, K, axis=0), s=np.repeat(data.frames.s, K, axis=0))
        pred_flat = v5.pos_std_to_global(flat, rep, model.norm)[:, :, :3]
        return pred_flat.reshape(n, K, H, 3)

    def linear_gate_scores(model, data):
        G = model.gate_features(data)
        return bk.add_bias(G) @ model.Wgate.T

    def hard_linear_pred(model, data):
        cand = candidates_global(model, data)
        scores = linear_gate_scores(model, data)
        idx = np.argmax(scores, axis=1)
        return (cand[np.arange(cand.shape[0]), idx], idx, scores, cand)

    def oracle_pred(model, data):
        cand = candidates_global(model, data)
        y = data.future_global[:, :, :3]
        diff = cand - y[:, None, :, :]
        err = np.sqrt(np.mean(diff ** 2, axis=(2, 3)))
        idx = np.argmin(err, axis=1)
        return (cand[np.arange(cand.shape[0]), idx], idx, err, cand)

    def soft_mixture_pred(cand, scores, temp):
        pi = softmax_np(scores, axis=1, temp=temp)
        return (np.sum(cand * pi[:, :, None, None], axis=1), pi)

    def select_alpha_fusion(val, pred_global, pred_switch, alpha_grid, args):
        best, rows = (None, [])
        for a in alpha_grid:
            pred = (1.0 - a) * pred_global + a * pred_switch
            met = compute_position_metrics(val.future_global[:, :, :3], pred)
            sc = metric_score(met, args)
            row = {'selector': 'global_switch_alpha', 'alpha': float(a), 'score': sc, **met}
            rows.append(row)
            if best is None or sc < best['score']:
                best = row
        return (best, pd.DataFrame(rows))

    def confidence_from_scores(scores, temp=1.0):
        p = softmax_np(scores, axis=1, temp=temp)
        part = np.sort(p, axis=1)
        top1 = part[:, -1]
        top2 = part[:, -2] if p.shape[1] >= 2 else np.zeros_like(top1)
        return (top1, top1 - top2, p)

    def select_conf_fallback(val, pred_global, pred_switch, scores, args):
        top1, margin, _ = confidence_from_scores(scores, temp=args.conf_temp)
        conf = top1 if args.confidence_type == 'top1' else margin
        best, rows = (None, [])
        for tau in parse_float_list(args.conf_threshold_grid):
            use = conf >= tau
            pred = pred_global.copy()
            pred[use] = pred_switch[use]
            met = compute_position_metrics(val.future_global[:, :, :3], pred)
            sc = metric_score(met, args)
            row = {'selector': 'confidence_fallback', 'threshold': float(tau), 'use_switch_ratio': float(np.mean(use)), 'score': sc, **met}
            rows.append(row)
            if best is None or sc < best['score']:
                best = row
        return (best, pd.DataFrame(rows))

    def apply_conf_fallback(pred_global, pred_switch, scores, tau, args):
        top1, margin, _ = confidence_from_scores(scores, temp=args.conf_temp)
        conf = top1 if args.confidence_type == 'top1' else margin
        use = conf >= float(tau)
        pred = pred_global.copy()
        pred[use] = pred_switch[use]
        return (pred, use)

    def select_soft_mixture(val, cand, scores, args):
        best, rows = (None, [])
        for T in parse_float_list(args.temperature_grid):
            pred, _ = soft_mixture_pred(cand, scores, T)
            met = compute_position_metrics(val.future_global[:, :, :3], pred)
            sc = metric_score(met, args)
            row = {'selector': 'soft_mixture', 'temperature': float(T), 'score': sc, **met}
            rows.append(row)
            if best is None or sc < best['score']:
                best = row
        return (best, pd.DataFrame(rows))

    def select_global_soft_fusion(val, pred_global, cand, scores, args):
        best, rows = (None, [])
        for T in parse_float_list(args.temperature_grid):
            pred_soft, _ = soft_mixture_pred(cand, scores, T)
            for a in parse_float_list(args.alpha_grid):
                pred = (1.0 - a) * pred_global + a * pred_soft
                met = compute_position_metrics(val.future_global[:, :, :3], pred)
                sc = metric_score(met, args)
                row = {'selector': 'global_soft_fusion', 'temperature': float(T), 'alpha': float(a), 'score': sc, **met}
                rows.append(row)
                if best is None or sc < best['score']:
                    best = row
        return (best, pd.DataFrame(rows))

    def candidate_errors(data, cand, metric='combo', fde_weight=0.5):
        y = data.future_global[:, :, :3]
        diff = cand - y[:, None, :, :]
        step = np.linalg.norm(diff, axis=-1)
        ade = step.mean(axis=-1)
        fde = step[:, :, -1]
        rmse = np.sqrt(np.mean(diff ** 2, axis=(2, 3)))
        if metric == 'ade':
            return ade
        if metric == 'fde':
            return fde
        if metric == 'rmse':
            return rmse
        return ade + float(fde_weight) * fde

    def select_residual_gate(model, train, val, cand_train, cand_val, args):
        best, rows, best_W = (None, [], None)
        Gtr = model.gate_features(train)
        Xtr = bk.add_bias(Gtr)
        Etr = candidate_errors(train, cand_train, metric=args.residual_gate_error_metric, fde_weight=args.residual_gate_fde_weight)
        Gva = model.gate_features(val)
        Xva = bk.add_bias(Gva)
        for r in parse_float_list(args.residual_gate_ridge_grid):
            W = ridge_fit(Xtr, Etr, r)
            pred_err = Xva @ W.T
            idx = np.argmin(pred_err, axis=1)
            pred = cand_val[np.arange(cand_val.shape[0]), idx]
            met = compute_position_metrics(val.future_global[:, :, :3], pred)
            sc = metric_score(met, args)
            row = {'selector': 'residual_aware_gate', 'ridge_err_gate': float(r), 'score': sc, **met}
            rows.append(row)
            if best is None or sc < best['score']:
                best, best_W = (row, W)
        return (best, best_W, pd.DataFrame(rows))

    def apply_residual_gate(model, data, cand, W):
        X = bk.add_bias(model.gate_features(data))
        pred_err = X @ W.T
        idx = np.argmin(pred_err, axis=1)
        return (cand[np.arange(cand.shape[0]), idx], idx, pred_err)

    def eval_pred(data, pred, method, extra=None):
        met = compute_position_metrics(data.future_global[:, :, :3], pred[:, :, :3])
        row = {'method': method, **met}
        if extra:
            row.update(extra)
        return (row, make_sample_df(data, pred, method))

    def main():
        ap = argparse.ArgumentParser(description='Standalone enhanced selector for unsupervised switching Koopman.')
        ap.add_argument('--data_root', type=str, required=True)
        ap.add_argument('--out_dir', type=str, default=None)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizon', type=int, default=10)
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--K_grid', type=str, default='2,3,4,5,6,8')
        ap.add_argument('--ridge_model_grid', type=str, default='1e-2,3e-2,1e-1,3e-1,1,3')
        ap.add_argument('--ridge_gate_grid', type=str, default='1e-2,1e-1,1,3')
        ap.add_argument('--min_cluster_samples', type=int, default=80)
        ap.add_argument('--alt_iter', type=int, default=12)
        ap.add_argument('--kmeans_iter', type=int, default=50)
        ap.add_argument('--global_ridge_xy_grid', type=str, default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--global_ridge_z_grid', type=str, default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--alpha_grid', type=str, default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        ap.add_argument('--temperature_grid', type=str, default='0.1,0.2,0.3,0.5,0.7,1,1.5,2,3,5')
        ap.add_argument('--confidence_type', type=str, default='margin', choices=['margin', 'top1'])
        ap.add_argument('--conf_temp', type=float, default=1.0)
        ap.add_argument('--conf_threshold_grid', type=str, default='0,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
        ap.add_argument('--residual_gate_ridge_grid', type=str, default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10')
        ap.add_argument('--residual_gate_error_metric', type=str, default='combo', choices=['rmse', 'ade', 'fde', 'combo'])
        ap.add_argument('--residual_gate_fde_weight', type=float, default=0.5)
        ap.add_argument('--selection_metric', type=str, default='combo', choices=['pos_rmse', 'ade', 'fde', 'fde_xy', 'fde_z', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--fast', action='store_true')
        args = ap.parse_args()
        root = Path(args.data_root)
        out_dir = Path(args.out_dir) if args.out_dir else root / 'unsup_switch_selector_enhance_standalone'
        ensure_dir(out_dir)
        if args.fast:
            args.K_grid = '2,3,4,6'
            args.ridge_model_grid = '3e-2,1e-1,3e-1,1'
            args.ridge_gate_grid = '1e-1,1'
            args.alt_iter = min(args.alt_iter, 8)
        train_chains = bk.load_split_chains(root, 'train')
        val_chains = bk.load_split_chains(root, 'val')
        test_chains = bk.load_split_chains(root, 'test')
        train = bk.make_split_data(train_chains, args.input_len, args.horizon, args.stride)
        val = bk.make_split_data(val_chains, args.input_len, args.horizon, args.stride)
        test = bk.make_split_data(test_chains, args.input_len, args.horizon, args.stride)
        print(f'samples: train={train.hist_global.shape[0]}, val={val.hist_global.shape[0]}, test={test.hist_global.shape[0]}')
        print('IMPORTANT: no action labels are used for training or selector tuning.')
        rows, sample_dfs, selector_sweeps, fit_sweeps = ([], [], [], [])
        selected = {}
        global_pack = fit_no_action_global(train, val, args)
        pred_global_val = predict_no_action_global(global_pack, val, args)
        pred_global_test = predict_no_action_global(global_pack, test, args)
        row, sdf = eval_pred(test, pred_global_test, 'v5_no_action_global')
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        print('v5_no_action_global:', row)
        best_val, best_model, best_desc = (None, None, None)
        for K in parse_int_list(args.K_grid):
            for ridge_model in parse_float_list(args.ridge_model_grid):
                for ridge_gate in parse_float_list(args.ridge_gate_grid):
                    print(f'fitting unsup switch K={K}, ridge_model={ridge_model}, ridge_gate={ridge_gate}')
                    model, assign, hist_df = fit_switching_model(train, K, ridge_model, ridge_gate, args, seed=args.seed + K)
                    pred_val, idx_val, scores_val_tmp, cand_val_tmp = hard_linear_pred(model, val)
                    met_val = compute_position_metrics(val.future_global[:, :, :3], pred_val)
                    sc = metric_score(met_val, args)
                    fit_row = {'K': K, 'ridge_model': ridge_model, 'ridge_gate': ridge_gate, 'score': sc, **met_val}
                    fit_sweeps.append(fit_row)
                    if best_val is None or sc < best_val['score']:
                        best_val = fit_row
                        best_model = model
                        best_desc = {'K': K, 'ridge_model': ridge_model, 'ridge_gate': ridge_gate}
        print('\nselected base unsup switch model:')
        print(best_val)
        model = best_model
        pred_switch_val, idx_val, scores_val, cand_val = hard_linear_pred(model, val)
        pred_switch_test, idx_test, scores_test, cand_test = hard_linear_pred(model, test)
        cand_train = candidates_global(model, train)
        row, sdf = eval_pred(test, pred_switch_test, 'unsup_switch_linear_gate', best_desc)
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        selected['base_linear_gate'] = best_val
        pred_oracle_test, idx_oracle, err_oracle, _ = oracle_pred(model, test)
        row, sdf = eval_pred(test, pred_oracle_test, 'unsup_switch_oracle', best_desc)
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        best_alpha, alpha_df = select_alpha_fusion(val, pred_global_val, pred_switch_val, parse_float_list(args.alpha_grid), args)
        selector_sweeps.append(alpha_df)
        pred_alpha_test = (1.0 - best_alpha['alpha']) * pred_global_test + best_alpha['alpha'] * pred_switch_test
        row, sdf = eval_pred(test, pred_alpha_test, 'selector_global_switch_alpha', {**best_desc, 'alpha': best_alpha['alpha']})
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        selected['global_switch_alpha'] = best_alpha
        best_conf, conf_df = select_conf_fallback(val, pred_global_val, pred_switch_val, scores_val, args)
        selector_sweeps.append(conf_df)
        pred_conf_test, use_switch_test = apply_conf_fallback(pred_global_test, pred_switch_test, scores_test, best_conf['threshold'], args)
        row, sdf = eval_pred(test, pred_conf_test, 'selector_confidence_fallback', {**best_desc, 'threshold': best_conf['threshold'], 'use_switch_ratio_test': float(np.mean(use_switch_test))})
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        selected['confidence_fallback'] = best_conf
        best_soft, soft_df = select_soft_mixture(val, cand_val, scores_val, args)
        selector_sweeps.append(soft_df)
        pred_soft_test, pi_test = soft_mixture_pred(cand_test, scores_test, best_soft['temperature'])
        row, sdf = eval_pred(test, pred_soft_test, 'selector_soft_mixture', {**best_desc, 'temperature': best_soft['temperature']})
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        selected['soft_mixture'] = best_soft
        best_gsoft, gsoft_df = select_global_soft_fusion(val, pred_global_val, cand_val, scores_val, args)
        selector_sweeps.append(gsoft_df)
        pred_soft_test_g, _ = soft_mixture_pred(cand_test, scores_test, best_gsoft['temperature'])
        pred_gsoft_test = (1.0 - best_gsoft['alpha']) * pred_global_test + best_gsoft['alpha'] * pred_soft_test_g
        row, sdf = eval_pred(test, pred_gsoft_test, 'selector_global_soft_fusion', {**best_desc, 'temperature': best_gsoft['temperature'], 'alpha': best_gsoft['alpha']})
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        selected['global_soft_fusion'] = best_gsoft
        best_res, Werr, res_df = select_residual_gate(model, train, val, cand_train, cand_val, args)
        selector_sweeps.append(res_df)
        pred_res_test, idx_res_test, pred_err_test = apply_residual_gate(model, test, cand_test, Werr)
        row, sdf = eval_pred(test, pred_res_test, 'selector_residual_aware_gate', {**best_desc, 'ridge_err_gate': best_res['ridge_err_gate']})
        rows.append({'horizon': args.horizon, **row})
        sample_dfs.append(sdf)
        selected['residual_aware_gate'] = best_res
        compare_df = pd.DataFrame(rows)
        sample_df = pd.concat(sample_dfs, ignore_index=True)
        selector_sweep_df = pd.concat(selector_sweeps, ignore_index=True)
        fit_sweep_df = pd.DataFrame(fit_sweeps)
        compare_df.to_csv(out_dir / 'compare_selector_metrics_test.csv', index=False)
        sample_df.to_csv(out_dir / 'sample_selector_metrics_test.csv', index=False)
        selector_sweep_df.to_csv(out_dir / 'selector_val_sweeps.csv', index=False)
        fit_sweep_df.to_csv(out_dir / 'base_unsup_switch_val_sweep.csv', index=False)
        group_metrics(sample_df, 'scene_type').to_csv(out_dir / 'grouped_by_scene_test.csv', index=False)
        group_metrics(sample_df, 'dominant_behavior').to_csv(out_dir / 'grouped_by_behavior_test.csv', index=False)
        group_metrics(sample_df, 'behavior_key').to_csv(out_dir / 'grouped_by_behavior_key_test.csv', index=False)
        report = {'version': 'unsup_switch_selector_enhance_standalone', 'data_root': str(root), 'out_dir': str(out_dir), 'input_len': args.input_len, 'horizon': args.horizon, 'stride': args.stride, 'selection_metric': args.selection_metric, 'selected_base_model': best_desc, 'selected_selectors': selected, 'notes': ['No action labels are used for training or selector tuning.', 'Selectors are tuned by validation trajectory error.', 'Oracle is an upper bound and not deployable.']}
        with open(out_dir / 'report_unsup_switch_selector_enhance_standalone.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('\n=== Standalone enhanced unsup switching selector experiments finished ===')
        print(compare_df.to_string(index=False))
        print(f'\nOutput directory: {out_dir}')
        print('Main file: compare_selector_metrics_test.csv')

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

unsup_switch_selector_enhance_standalone_v2 = _init_unsup_switch_selector_enhance_standalone_v2()


# ========================================================================================
# SECTION: UNSUP KOOPMAN NEURAL HSMM SELECTOR V2
# ========================================================================================
def _init_unsup_koopman_neural_hsmm_selector_v2():
    """
    unsup_koopman_neural_hsmm_selector.py

    Experiments for improving no-action-label Switching Koopman trajectory prediction.

    Experiment 1: Oracle-distilled GRU latent-mode selector + global safety fusion
        - Unsupervised Switching Koopman first generates K latent trajectory candidates.
        - No human action labels are used.
        - Training trajectory targets are used to calculate each candidate's future error.
        - A GRU gate learns soft oracle candidate targets and outputs candidate mixture weights.
        - A disagreement-aware global safety fusion is tuned on validation data.

    Experiment 2: Duration-aware HMM / HSMM temporal smoothing
        - Latent assignments from the no-label Switching Koopman training stage are used
          to estimate latent transition probabilities and empirical duration distributions.
        - GRU gate probabilities are temporally decoded by HMM / explicit-duration HSMM.
        - Decoded mode trajectory is fused with the no-action global Koopman predictor.

    Important setting statement:
        This script does NOT use behavior_key or any action label in training, model selection,
        gating, or temporal decoding. It uses normal trajectory prediction supervision
        (historical trajectory -> future trajectory), which is allowed in a trajectory
        forecasting experiment.

    Required files in the same directory:
        behavior_mode_endpoint_koopman_v2_hmm_selector.py
        axis_feature_decoupled_position_behavior_koopman_v5.py
        unsup_switch_selector_enhance_standalone_v2.py

    Fast run:
        python unsup_koopman_neural_hsmm_selector.py --data_root DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed --fast --out_dir DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed/unsup_koopman_neural_hsmm_fast

    Full run:
        python unsup_koopman_neural_hsmm_selector.py --data_root DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed --out_dir DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed/unsup_koopman_neural_hsmm
    """
    import argparse
    import copy
    import json
    import random
    from pathlib import Path
    import numpy as np
    import pandas as pd
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError('PyTorch is required for the neural selector experiment.') from e
    base = unsup_switch_selector_enhance_standalone_v2
    bk = base.bk
    v5 = base.v5

    def ensure_dir(p):
        Path(p).mkdir(parents=True, exist_ok=True)

    def parse_float_list(s):
        return [float(x.strip()) for x in str(s).split(',') if x.strip()]

    def set_seed(seed):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def score_from_metrics(met, args):
        if args.selection_metric == 'pos_rmse':
            return met['pos_rmse']
        if args.selection_metric == 'ade':
            return met['ade']
        if args.selection_metric == 'fde':
            return met['fde']
        if args.selection_metric == 'fde_xy':
            return met['fde_xy']
        if args.selection_metric == 'fde_z':
            return met['fde_z']
        return met['pos_rmse'] + args.combo_ade_weight * met['ade'] + args.combo_fde_weight * met['fde']

    def add_result(rows, sample_dfs, data, pred, method, extra=None):
        met = base.compute_position_metrics(data.future_global[:, :, :3], pred[:, :, :3])
        row = {'method': method, 'horizon': data.future_global.shape[1], **met}
        if extra:
            row.update(extra)
        rows.append(row)
        sample_dfs.append(base.make_sample_df(data, pred, method))
        return row

    def candidate_error_matrix(data, cand, args):
        """
        cand: [N,K,H,3], in global positions.
        Returns sample-wise candidate forecasting error [N,K].
        This uses trajectory target, not action labels.
        """
        y = np.asarray(data.future_global[:, :, :3], dtype=np.float64)
        diff = cand - y[:, None, :, :]
        step = np.linalg.norm(diff, axis=-1)
        rmse = np.sqrt(np.mean(diff ** 2, axis=(2, 3)))
        ade = np.mean(step, axis=-1)
        fde = step[:, :, -1]
        if args.distill_error_metric == 'rmse':
            return rmse
        if args.distill_error_metric == 'ade':
            return ade
        if args.distill_error_metric == 'fde':
            return fde
        return rmse + args.distill_ade_weight * ade + args.distill_fde_weight * fde

    def soft_targets_from_error(err, tau):
        z = -np.asarray(err, dtype=np.float64) / max(float(tau), 1e-08)
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z)
        return (q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)

    def trajectory_loss_torch(pred, true, args):
        diff = pred - true
        rmse = torch.sqrt(torch.mean(diff ** 2) + 1e-10)
        step = torch.linalg.vector_norm(diff, dim=-1)
        ade = step.mean()
        fde = step[:, -1].mean()
        return rmse + args.train_ade_weight * ade + args.train_fde_weight * fde

    def unwrap_angle_np(theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    def make_sequence_features(data, dt=0.1):
        """
        Trajectory-only neural gate input [N,L,D].
        No action labels or duration labels are included.
        """
        hist = np.asarray(data.hist_global, dtype=np.float32)
        p = hist[:, :, :3]
        v = hist[:, :, 3:6]
        N, L, _ = hist.shape
        p_rel = p - p[:, -1:, :]
        vxy = v[:, :, :2]
        speed_xy = np.linalg.norm(vxy, axis=-1, keepdims=True)
        speed_3d = np.linalg.norm(v, axis=-1, keepdims=True)
        vz = v[:, :, 2:3]
        heading = np.arctan2(vxy[:, :, 1], vxy[:, :, 0] + 1e-12)
        head_sin = np.sin(heading)[..., None]
        head_cos = np.cos(heading)[..., None]
        accel = np.zeros_like(v)
        dp = np.zeros_like(p)
        head_rate = np.zeros((N, L, 1), dtype=np.float32)
        if L >= 2:
            accel[:, 1:, :] = (v[:, 1:, :] - v[:, :-1, :]) / max(float(dt), 1e-08)
            accel[:, 0, :] = accel[:, 1, :]
            dp[:, 1:, :] = p[:, 1:, :] - p[:, :-1, :]
            dp[:, 0, :] = dp[:, 1, :]
            dhead = unwrap_angle_np(np.diff(heading, axis=1)) / max(float(dt), 1e-08)
            head_rate[:, 1:, 0] = dhead
            head_rate[:, 0, 0] = head_rate[:, 1, 0]
        acc_xy = np.linalg.norm(accel[:, :, :2], axis=-1, keepdims=True)
        acc_3d = np.linalg.norm(accel, axis=-1, keepdims=True)
        dp_xy = np.linalg.norm(dp[:, :, :2], axis=-1, keepdims=True)
        dp_3d = np.linalg.norm(dp, axis=-1, keepdims=True)
        tau = np.repeat(np.linspace(-1.0, 0.0, L, dtype=np.float32)[None, :, None], N, axis=0)
        X = np.concatenate([p_rel, v, speed_xy, speed_3d, vz, head_sin, head_cos, head_rate, accel, acc_xy, acc_3d, dp, dp_xy, dp_3d, tau], axis=-1)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    class SeqStandardizer:

        def __init__(self, mean, std):
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        @classmethod
        def fit(cls, X, eps=1e-06):
            flat = X.reshape(-1, X.shape[-1])
            mean = flat.mean(axis=0)
            std = np.maximum(flat.std(axis=0), eps)
            return cls(mean, std)

        def transform(self, X):
            return ((X - self.mean) / self.std).astype(np.float32)

    class GateDataset(Dataset):

        def __init__(self, X, q, cand, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.q = torch.from_numpy(q.astype(np.float32))
            self.cand = torch.from_numpy(cand.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return (self.X[idx], self.q[idx], self.cand[idx], self.y[idx])

    class GRULatentSelector(nn.Module):

        def __init__(self, input_dim, K, hidden_dim=128, layers=2, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, K))

        def forward(self, x):
            _, h = self.gru(x)
            z = torch.cat([h[-2], h[-1]], dim=-1)
            return self.head(z)

    def predict_gate_probs(model, X, batch_size, device):
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[s:s + batch_size]).float().to(device)
                pi = torch.softmax(model(xb), dim=-1).cpu().numpy()
                outs.append(pi)
        return np.concatenate(outs, axis=0)

    def mixture_prediction(cand, probs):
        return np.sum(cand * probs[:, :, None, None], axis=1)

    def tune_constant_global_fusion(val, pred_global, pred_switch, args, prefix='neural_alpha'):
        rows = []
        best = None
        for alpha in parse_float_list(args.alpha_grid):
            pred = pred_global + float(alpha) * (pred_switch - pred_global)
            met = base.compute_position_metrics(val.future_global[:, :, :3], pred)
            row = {'selector': prefix, 'alpha': float(alpha), 'score': score_from_metrics(met, args), **met}
            rows.append(row)
            if best is None or row['score'] < best['score']:
                best = row
        return (best, pd.DataFrame(rows))

    def tune_disagreement_safety_fusion(val, pred_global, pred_switch, args, prefix='neural_safety'):
        """
        Adaptive global fallback:
            pred = global + alpha_i * (switch - global)
            alpha_i = alpha * min(1, tau / disagreement_i)
        """
        delta = np.sqrt(np.mean((pred_switch - pred_global) ** 2, axis=(1, 2)))
        quantiles = parse_float_list(args.safety_quantile_grid)
        rows = []
        best = None
        for q in quantiles:
            tau = float(np.quantile(delta, q))
            for alpha in parse_float_list(args.alpha_grid):
                scale = np.minimum(1.0, tau / np.maximum(delta, 1e-12))
                ai = float(alpha) * scale
                pred = pred_global + ai[:, None, None] * (pred_switch - pred_global)
                met = base.compute_position_metrics(val.future_global[:, :, :3], pred)
                row = {'selector': prefix, 'alpha': float(alpha), 'safety_quantile': float(q), 'safety_tau': tau, 'mean_effective_alpha': float(ai.mean()), 'score': score_from_metrics(met, args), **met}
                rows.append(row)
                if best is None or row['score'] < best['score']:
                    best = row
        return (best, pd.DataFrame(rows))

    def apply_disagreement_safety(pred_global, pred_switch, alpha, tau):
        delta = np.sqrt(np.mean((pred_switch - pred_global) ** 2, axis=(1, 2)))
        scale = np.minimum(1.0, float(tau) / np.maximum(delta, 1e-12))
        ai = float(alpha) * scale
        pred = pred_global + ai[:, None, None] * (pred_switch - pred_global)
        return (pred, ai, delta)

    def train_one_gate(Xtr, qtr, cand_tr, ytr, Xva, cand_va, yva, K, args, device, tau, lam):
        train_loader = DataLoader(GateDataset(Xtr, qtr, cand_tr, ytr), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
        model = GRULatentSelector(input_dim=Xtr.shape[-1], K=K, hidden_dim=args.hidden_dim, layers=args.gru_layers, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_score = None
        best_state = None
        best_epoch = -1
        patience_count = 0
        history = []
        yva_t = np.asarray(yva, dtype=np.float64)
        for epoch in range(1, args.epochs + 1):
            model.train()
            sum_loss = 0.0
            n_seen = 0
            for xb, qb, cb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                qb = qb.to(device, non_blocking=True)
                cb = cb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                logpi = torch.log_softmax(logits, dim=-1)
                pi = torch.softmax(logits, dim=-1)
                loss_distill = -(qb * logpi).sum(dim=-1).mean()
                pred = (pi[:, :, None, None] * cb).sum(dim=1)
                loss_traj = trajectory_loss_torch(pred, yb, args)
                loss = loss_distill + float(lam) * loss_traj
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sum_loss += float(loss.item()) * xb.shape[0]
                n_seen += xb.shape[0]
            pva = predict_gate_probs(model, Xva, args.batch_size, device)
            pred_va = mixture_prediction(cand_va, pva)
            met_va = base.compute_position_metrics(yva_t, pred_va)
            sc = score_from_metrics(met_va, args)
            history.append({'oracle_tau': tau, 'traj_lambda': lam, 'epoch': epoch, 'train_loss': sum_loss / max(n_seen, 1), 'val_score_soft': sc, **{f'val_{k}': v for k, v in met_va.items()}})
            if best_score is None or sc < best_score:
                best_score = sc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
            if epoch == 1 or epoch % args.log_every == 0 or patience_count == 0:
                print(f"[GRU gate tau={tau:g} lambda={lam:g}] epoch={epoch:03d} loss={sum_loss / max(n_seen, 1):.5f} val_score={sc:.6f} ADE={met_va['ade']:.6f} FDE={met_va['fde']:.6f}")
            if patience_count >= args.patience:
                break
        model.load_state_dict(best_state)
        return (model, pd.DataFrame(history), {'oracle_tau': tau, 'traj_lambda': lam, 'best_epoch': best_epoch, 'best_soft_score': best_score})

    def sequence_groups(data):
        n = data.hist_global.shape[0]
        if hasattr(data, 'chain_file') and len(data.chain_file) == n:
            keys = list(data.chain_file)
        else:
            keys = ['single_chain'] * n
        groups = {}
        order = []
        for i, key in enumerate(keys):
            key = str(key)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(i)
        return [np.asarray(groups[k], dtype=np.int64) for k in order]

    def estimate_latent_temporal_stats(train, latent_assign, K, args):
        groups = sequence_groups(train)
        smooth = float(args.temporal_smooth)
        init = np.full(K, smooth, dtype=np.float64)
        trans = np.full((K, K), smooth, dtype=np.float64)
        runs = [[] for _ in range(K)]
        for idxs in groups:
            z = np.asarray(latent_assign[idxs], dtype=np.int64)
            if len(z) == 0:
                continue
            init[z[0]] += 1.0
            for a, b in zip(z[:-1], z[1:]):
                trans[a, b] += 1.0
            cur = int(z[0])
            length = 1
            for value in z[1:]:
                value = int(value)
                if value == cur:
                    length += 1
                else:
                    runs[cur].append(length)
                    cur = value
                    length = 1
            runs[cur].append(length)
        init = init / init.sum()
        trans = trans / trans.sum(axis=1, keepdims=True)
        dmax = int(args.max_duration)
        duration = np.full((K, dmax + 1), smooth, dtype=np.float64)
        for k in range(K):
            for d in runs[k]:
                duration[k, min(int(d), dmax)] += 1.0
        duration[:, 0] = 0.0
        duration = duration / np.maximum(duration.sum(axis=1, keepdims=True), 1e-12)
        run_summary = []
        for k, values in enumerate(runs):
            arr = np.asarray(values if values else [1], dtype=np.float64)
            run_summary.append({'latent_mode': k, 'num_runs': int(len(values)), 'mean_duration': float(arr.mean()), 'median_duration': float(np.median(arr)), 'p90_duration': float(np.quantile(arr, 0.9))})
        return (init, trans, duration, pd.DataFrame(run_summary))

    def viterbi_hmm_one(prob_seq, init, trans, trans_weight):
        T, K = prob_seq.shape
        loge = np.log(np.maximum(prob_seq, 1e-12))
        logi = np.log(np.maximum(init, 1e-12))
        logt = np.log(np.maximum(trans, 1e-12))
        dp = np.full((T, K), -np.inf, dtype=np.float64)
        back = np.zeros((T, K), dtype=np.int64)
        dp[0] = loge[0] + float(trans_weight) * logi
        for t in range(1, T):
            for k in range(K):
                scores = dp[t - 1] + float(trans_weight) * logt[:, k]
                j = int(np.argmax(scores))
                dp[t, k] = scores[j] + loge[t, k]
                back[t, k] = j
        z = np.zeros(T, dtype=np.int64)
        z[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 1, 0, -1):
            z[t - 1] = back[t, z[t]]
        return z

    def decode_hmm(probs, data, init, trans, trans_weight):
        out = np.zeros(probs.shape[0], dtype=np.int64)
        for idxs in sequence_groups(data):
            out[idxs] = viterbi_hmm_one(probs[idxs], init, trans, trans_weight)
        return out

    def viterbi_hsmm_one(prob_seq, init, trans, duration, trans_weight, duration_weight, max_duration):
        T, K = prob_seq.shape
        loge = np.log(np.maximum(prob_seq, 1e-12))
        logi = np.log(np.maximum(init, 1e-12))
        logt = np.log(np.maximum(trans, 1e-12))
        logd = np.log(np.maximum(duration, 1e-12))
        prefix = np.vstack([np.zeros((1, K), dtype=np.float64), np.cumsum(loge, axis=0)])
        dp = np.full((T + 1, K), -np.inf, dtype=np.float64)
        prev_t = np.full((T + 1, K), -1, dtype=np.int64)
        prev_k = np.full((T + 1, K), -1, dtype=np.int64)
        prev_d = np.full((T + 1, K), -1, dtype=np.int64)
        for end in range(1, T + 1):
            for k in range(K):
                best = -np.inf
                best_start = -1
                best_prev_k = -1
                best_d = -1
                for d in range(1, min(max_duration, end) + 1):
                    start = end - d
                    seg_emit = prefix[end, k] - prefix[start, k]
                    dur_term = float(duration_weight) * logd[k, d]
                    if start == 0:
                        score = float(trans_weight) * logi[k] + seg_emit + dur_term
                        j = -1
                    else:
                        prior = dp[start].copy() + float(trans_weight) * logt[:, k]
                        prior[k] = -np.inf
                        j = int(np.argmax(prior))
                        score = prior[j] + seg_emit + dur_term
                    if score > best:
                        best = score
                        best_start = start
                        best_prev_k = j
                        best_d = d
                dp[end, k] = best
                prev_t[end, k] = best_start
                prev_k[end, k] = best_prev_k
                prev_d[end, k] = best_d
        z = np.zeros(T, dtype=np.int64)
        end = T
        k = int(np.argmax(dp[T]))
        while end > 0:
            d = int(prev_d[end, k])
            start = int(prev_t[end, k])
            if d <= 0 or start < 0:
                z[:end] = k
                break
            z[start:end] = k
            pk = int(prev_k[end, k])
            end = start
            if end > 0:
                k = pk if pk >= 0 else k
        return z

    def decode_hsmm(probs, data, init, trans, duration, trans_weight, duration_weight, max_duration):
        out = np.zeros(probs.shape[0], dtype=np.int64)
        for idxs in sequence_groups(data):
            out[idxs] = viterbi_hsmm_one(probs[idxs], init, trans, duration, trans_weight, duration_weight, max_duration)
        return out

    def candidate_by_mode(cand, idx):
        return cand[np.arange(cand.shape[0]), np.asarray(idx, dtype=np.int64)]

    def tune_temporal_model(kind, probs_val, val, cand_val, pred_global_val, init, trans, duration, args):
        rows = []
        best = None
        for tw in parse_float_list(args.transition_weight_grid):
            dw_grid = [0.0] if kind == 'hmm' else parse_float_list(args.duration_weight_grid)
            for dw in dw_grid:
                if kind == 'hmm':
                    idx = decode_hmm(probs_val, val, init, trans, tw)
                else:
                    idx = decode_hsmm(probs_val, val, init, trans, duration, tw, dw, args.max_duration)
                hard = candidate_by_mode(cand_val, idx)
                for alpha in parse_float_list(args.temporal_alpha_grid):
                    pred = pred_global_val + float(alpha) * (hard - pred_global_val)
                    met = base.compute_position_metrics(val.future_global[:, :, :3], pred)
                    row = {'selector': f'neural_{kind}_global_alpha', 'transition_weight': float(tw), 'duration_weight': float(dw), 'alpha': float(alpha), 'score': score_from_metrics(met, args), **met}
                    rows.append(row)
                    if best is None or row['score'] < best['score']:
                        best = row
        return (best, pd.DataFrame(rows))

    def main():
        ap = argparse.ArgumentParser(description='No-action-label neural selector and HSMM experiments.')
        ap.add_argument('--data_root', type=str, required=True)
        ap.add_argument('--out_dir', type=str, default=None)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizon', type=int, default=10)
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--K', type=int, default=4)
        ap.add_argument('--ridge_model', type=float, default=1.0)
        ap.add_argument('--ridge_gate', type=float, default=1.0)
        ap.add_argument('--min_cluster_samples', type=int, default=80)
        ap.add_argument('--alt_iter', type=int, default=12)
        ap.add_argument('--kmeans_iter', type=int, default=50)
        ap.add_argument('--global_ridge_xy_grid', type=str, default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--global_ridge_z_grid', type=str, default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--alpha_grid', type=str, default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        ap.add_argument('--temperature_grid', type=str, default='0.1,0.2,0.3,0.5,0.7,1,1.5,2,3,5')
        ap.add_argument('--selection_metric', type=str, default='combo', choices=['pos_rmse', 'ade', 'fde', 'fde_xy', 'fde_z', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--distill_error_metric', type=str, default='combo', choices=['rmse', 'ade', 'fde', 'combo'])
        ap.add_argument('--distill_ade_weight', type=float, default=0.5)
        ap.add_argument('--distill_fde_weight', type=float, default=0.25)
        ap.add_argument('--oracle_tau_grid', type=str, default='0.005,0.01,0.02,0.05')
        ap.add_argument('--traj_lambda_grid', type=str, default='0,2,5,10')
        ap.add_argument('--train_ade_weight', type=float, default=0.5)
        ap.add_argument('--train_fde_weight', type=float, default=0.25)
        ap.add_argument('--hidden_dim', type=int, default=128)
        ap.add_argument('--gru_layers', type=int, default=2)
        ap.add_argument('--dropout', type=float, default=0.2)
        ap.add_argument('--epochs', type=int, default=100)
        ap.add_argument('--patience', type=int, default=15)
        ap.add_argument('--batch_size', type=int, default=512)
        ap.add_argument('--lr', type=float, default=0.001)
        ap.add_argument('--weight_decay', type=float, default=0.0001)
        ap.add_argument('--grad_clip', type=float, default=1.0)
        ap.add_argument('--log_every', type=int, default=10)
        ap.add_argument('--safety_quantile_grid', type=str, default='0.25,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0')
        ap.add_argument('--temporal_smooth', type=float, default=1.0)
        ap.add_argument('--max_duration', type=int, default=30)
        ap.add_argument('--transition_weight_grid', type=str, default='0,0.05,0.1,0.2,0.3,0.5,0.8,1')
        ap.add_argument('--duration_weight_grid', type=str, default='0.05,0.1,0.2,0.3,0.5,0.8,1')
        ap.add_argument('--temporal_alpha_grid', type=str, default='0,0.2,0.4,0.6,0.8,1')
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--device', type=str, default='auto')
        ap.add_argument('--fast', action='store_true')
        args = ap.parse_args()
        set_seed(args.seed)
        if args.fast:
            args.alt_iter = min(args.alt_iter, 8)
            args.epochs = min(args.epochs, 50)
            args.patience = min(args.patience, 10)
            args.oracle_tau_grid = '0.01,0.02'
            args.traj_lambda_grid = '2,5'
            args.safety_quantile_grid = '0.4,0.6,0.8,0.95,1.0'
            args.transition_weight_grid = '0,0.1,0.3,0.5,1'
            args.duration_weight_grid = '0.1,0.3,0.5,1'
            args.temporal_alpha_grid = '0,0.4,0.6,0.8,1'
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        print(f'device={device}')
        root = Path(args.data_root)
        out_dir = Path(args.out_dir) if args.out_dir else root / 'unsup_koopman_neural_hsmm'
        ensure_dir(out_dir)
        train_chains = bk.load_split_chains(root, 'train')
        val_chains = bk.load_split_chains(root, 'val')
        test_chains = bk.load_split_chains(root, 'test')
        train = bk.make_split_data(train_chains, args.input_len, args.horizon, args.stride)
        val = bk.make_split_data(val_chains, args.input_len, args.horizon, args.stride)
        test = bk.make_split_data(test_chains, args.input_len, args.horizon, args.stride)
        print(f'samples: train={train.hist_global.shape[0]}, val={val.hist_global.shape[0]}, test={test.hist_global.shape[0]}')
        print('IMPORTANT: behavior/action labels are not used in training or selection.')
        rows, sample_dfs, sweep_dfs = ([], [], [])
        selected = {}
        global_pack = base.fit_no_action_global(train, val, args)
        pred_global_val = base.predict_no_action_global(global_pack, val, args)
        pred_global_test = base.predict_no_action_global(global_pack, test, args)
        print('fitted no-action global.')
        latent_model, latent_assign_train, alt_hist = base.fit_switching_model(train, args.K, args.ridge_model, args.ridge_gate, args, seed=args.seed + args.K)
        alt_hist.to_csv(out_dir / 'latent_alternating_history.csv', index=False)
        cand_train = base.candidates_global(latent_model, train)
        cand_val = base.candidates_global(latent_model, val)
        cand_test = base.candidates_global(latent_model, test)
        pred_linear_val, _, linear_scores_val, _ = base.hard_linear_pred(latent_model, val)
        pred_linear_test, _, linear_scores_test, _ = base.hard_linear_pred(latent_model, test)
        pred_oracle_test, _, _, _ = base.oracle_pred(latent_model, test)
        add_result(rows, sample_dfs, test, pred_global_test, 'v5_no_action_global')
        add_result(rows, sample_dfs, test, pred_linear_test, 'unsup_switch_linear_gate', {'K': args.K, 'ridge_model': args.ridge_model, 'ridge_gate': args.ridge_gate})
        add_result(rows, sample_dfs, test, pred_oracle_test, 'unsup_switch_oracle', {'K': args.K, 'ridge_model': args.ridge_model, 'ridge_gate': args.ridge_gate})
        best_linear_gsoft, linear_gsoft_sweep = base.select_global_soft_fusion(val, pred_global_val, cand_val, linear_scores_val, args)
        sweep_dfs.append(linear_gsoft_sweep)
        pred_linear_soft_test, _ = base.soft_mixture_pred(cand_test, linear_scores_test, best_linear_gsoft['temperature'])
        pred_linear_gsoft_test = pred_global_test + best_linear_gsoft['alpha'] * (pred_linear_soft_test - pred_global_test)
        add_result(rows, sample_dfs, test, pred_linear_gsoft_test, 'linear_global_soft_fusion', {'alpha': best_linear_gsoft['alpha'], 'temperature': best_linear_gsoft['temperature']})
        selected['linear_global_soft_fusion'] = best_linear_gsoft
        Xtr_raw = make_sequence_features(train, args.dt)
        Xva_raw = make_sequence_features(val, args.dt)
        Xte_raw = make_sequence_features(test, args.dt)
        xstd = SeqStandardizer.fit(Xtr_raw)
        Xtr = xstd.transform(Xtr_raw)
        Xva = xstd.transform(Xva_raw)
        Xte = xstd.transform(Xte_raw)
        ytr = np.asarray(train.future_global[:, :, :3], dtype=np.float32)
        yva = np.asarray(val.future_global[:, :, :3], dtype=np.float32)
        Etr = candidate_error_matrix(train, cand_train, args)
        Eve = candidate_error_matrix(val, cand_val, args)
        gate_records = []
        training_histories = []
        best_gate = None
        best_gate_pack = None
        for tau in parse_float_list(args.oracle_tau_grid):
            qtr = soft_targets_from_error(Etr, tau)
            for lam in parse_float_list(args.traj_lambda_grid):
                print(f'\ntraining oracle-distilled GRU gate: tau={tau}, lambda={lam}')
                gate, hist, info = train_one_gate(Xtr, qtr, cand_train, ytr, Xva, cand_val, yva, args.K, args, device, tau, lam)
                training_histories.append(hist)
                pva = predict_gate_probs(gate, Xva, args.batch_size, device)
                pred_neural_val = mixture_prediction(cand_val, pva)
                best_alpha, alpha_sweep = tune_constant_global_fusion(val, pred_global_val, pred_neural_val, args, prefix='neural_global_alpha')
                best_safe, safe_sweep = tune_disagreement_safety_fusion(val, pred_global_val, pred_neural_val, args, prefix='neural_global_safety')
                alpha_sweep['oracle_tau'] = tau
                alpha_sweep['traj_lambda'] = lam
                safe_sweep['oracle_tau'] = tau
                safe_sweep['traj_lambda'] = lam
                sweep_dfs.extend([alpha_sweep, safe_sweep])
                record = {**info, 'safe_val_score': best_safe['score'], 'safe_val_pos_rmse': best_safe['pos_rmse'], 'safe_val_ade': best_safe['ade'], 'safe_val_fde': best_safe['fde'], 'safe_alpha': best_safe['alpha'], 'safe_quantile': best_safe['safety_quantile'], 'safe_tau': best_safe['safety_tau'], 'alpha_val_score': best_alpha['score'], 'alpha': best_alpha['alpha']}
                gate_records.append(record)
                if best_gate_pack is None or best_safe['score'] < best_gate_pack['best_safe']['score']:
                    best_gate = copy.deepcopy(gate).cpu()
                    best_gate_pack = {'info': info, 'best_alpha': best_alpha, 'best_safe': best_safe}
        pd.DataFrame(gate_records).to_csv(out_dir / 'neural_gate_model_selection_val.csv', index=False)
        pd.concat(training_histories, ignore_index=True).to_csv(out_dir / 'neural_gate_training_history.csv', index=False)
        best_gate = best_gate.to(device)
        pval = predict_gate_probs(best_gate, Xva, args.batch_size, device)
        ptest = predict_gate_probs(best_gate, Xte, args.batch_size, device)
        pred_neural_val = mixture_prediction(cand_val, pval)
        pred_neural_test = mixture_prediction(cand_test, ptest)
        best_alpha = best_gate_pack['best_alpha']
        pred_neural_alpha_test = pred_global_test + best_alpha['alpha'] * (pred_neural_test - pred_global_test)
        best_safe = best_gate_pack['best_safe']
        pred_neural_safe_test, effective_alpha_test, disagreement_test = apply_disagreement_safety(pred_global_test, pred_neural_test, best_safe['alpha'], best_safe['safety_tau'])
        add_result(rows, sample_dfs, test, pred_neural_test, 'neural_distilled_soft_mixture', {'oracle_tau': best_gate_pack['info']['oracle_tau'], 'traj_lambda': best_gate_pack['info']['traj_lambda']})
        add_result(rows, sample_dfs, test, pred_neural_alpha_test, 'neural_distilled_global_alpha', {'alpha': best_alpha['alpha'], 'oracle_tau': best_gate_pack['info']['oracle_tau'], 'traj_lambda': best_gate_pack['info']['traj_lambda']})
        add_result(rows, sample_dfs, test, pred_neural_safe_test, 'neural_distilled_global_safety', {'alpha': best_safe['alpha'], 'safety_tau': best_safe['safety_tau'], 'mean_effective_alpha_test': float(effective_alpha_test.mean()), 'oracle_tau': best_gate_pack['info']['oracle_tau'], 'traj_lambda': best_gate_pack['info']['traj_lambda']})
        selected['neural_distilled_gate'] = best_gate_pack
        pd.DataFrame({'sample_idx': np.arange(test.hist_global.shape[0]), 'disagreement': disagreement_test, 'effective_alpha': effective_alpha_test, **{f'prob_mode_{k}': ptest[:, k] for k in range(args.K)}}).to_csv(out_dir / 'neural_gate_test_probabilities_and_safety.csv', index=False)
        init, trans, duration, run_summary = estimate_latent_temporal_stats(train, latent_assign_train, args.K, args)
        pd.DataFrame(trans).to_csv(out_dir / 'latent_transition_matrix_train.csv', index=False)
        pd.DataFrame(duration).to_csv(out_dir / 'latent_duration_distribution_train.csv', index=False)
        run_summary.to_csv(out_dir / 'latent_duration_summary_train.csv', index=False)
        best_hmm, hmm_sweep = tune_temporal_model('hmm', pval, val, cand_val, pred_global_val, init, trans, duration, args)
        best_hsmm, hsmm_sweep = tune_temporal_model('hsmm', pval, val, cand_val, pred_global_val, init, trans, duration, args)
        sweep_dfs.extend([hmm_sweep, hsmm_sweep])
        idx_hmm_test = decode_hmm(ptest, test, init, trans, best_hmm['transition_weight'])
        pred_hmm_hard_test = candidate_by_mode(cand_test, idx_hmm_test)
        pred_hmm_test = pred_global_test + best_hmm['alpha'] * (pred_hmm_hard_test - pred_global_test)
        idx_hsmm_test = decode_hsmm(ptest, test, init, trans, duration, best_hsmm['transition_weight'], best_hsmm['duration_weight'], args.max_duration)
        pred_hsmm_hard_test = candidate_by_mode(cand_test, idx_hsmm_test)
        pred_hsmm_test = pred_global_test + best_hsmm['alpha'] * (pred_hsmm_hard_test - pred_global_test)
        add_result(rows, sample_dfs, test, pred_hmm_hard_test, 'neural_hmm_hard_mode', {'transition_weight': best_hmm['transition_weight']})
        add_result(rows, sample_dfs, test, pred_hmm_test, 'neural_hmm_global_alpha', {'transition_weight': best_hmm['transition_weight'], 'alpha': best_hmm['alpha']})
        add_result(rows, sample_dfs, test, pred_hsmm_hard_test, 'neural_hsmm_hard_mode', {'transition_weight': best_hsmm['transition_weight'], 'duration_weight': best_hsmm['duration_weight']})
        add_result(rows, sample_dfs, test, pred_hsmm_test, 'neural_hsmm_global_alpha', {'transition_weight': best_hsmm['transition_weight'], 'duration_weight': best_hsmm['duration_weight'], 'alpha': best_hsmm['alpha']})
        selected['hmm'] = best_hmm
        selected['hsmm'] = best_hsmm
        pd.DataFrame({'sample_idx': np.arange(test.hist_global.shape[0]), 'hmm_latent_mode': idx_hmm_test, 'hsmm_latent_mode': idx_hsmm_test}).to_csv(out_dir / 'temporal_decoded_modes_test.csv', index=False)
        compare_df = pd.DataFrame(rows)
        sample_df = pd.concat(sample_dfs, ignore_index=True)
        sweep_df = pd.concat(sweep_dfs, ignore_index=True)
        compare_df.to_csv(out_dir / 'compare_neural_hsmm_selector_metrics_test.csv', index=False)
        sample_df.to_csv(out_dir / 'sample_neural_hsmm_selector_metrics_test.csv', index=False)
        sweep_df.to_csv(out_dir / 'selector_validation_sweeps.csv', index=False)
        base.group_metrics(sample_df, 'scene_type').to_csv(out_dir / 'grouped_by_scene_test.csv', index=False)
        base.group_metrics(sample_df, 'dominant_behavior').to_csv(out_dir / 'grouped_by_behavior_posthoc_test.csv', index=False)
        base.group_metrics(sample_df, 'behavior_key').to_csv(out_dir / 'grouped_by_behavior_key_posthoc_test.csv', index=False)
        torch.save({'model_state_dict': best_gate.cpu().state_dict(), 'sequence_feature_mean': xstd.mean.tolist(), 'sequence_feature_std': xstd.std.tolist(), 'selected': best_gate_pack, 'args': vars(args)}, out_dir / 'oracle_distilled_gru_selector.pt')
        report = {'version': 'unsup_koopman_neural_hsmm_selector', 'setting': 'no action labels in training or selector tuning', 'data_root': str(root), 'out_dir': str(out_dir), 'K': args.K, 'ridge_model': args.ridge_model, 'ridge_gate': args.ridge_gate, 'selected': selected, 'important_note': 'Future trajectory targets are used to train and validate forecasting selectors, but behavior_key/action labels are not used. Behavior-wise CSVs are post-hoc analysis only.'}
        with open(out_dir / 'report_unsup_koopman_neural_hsmm_selector.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('\n=== No-action-label Neural Selector + HMM/HSMM experiments finished ===')
        print(compare_df.to_string(index=False))
        print(f'\nOutput directory: {out_dir}')
        print('Main file: compare_neural_hsmm_selector_metrics_test.csv')

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

unsup_koopman_neural_hsmm_selector_v2 = _init_unsup_koopman_neural_hsmm_selector_v2()


# ========================================================================================
# SECTION: DIFFERENTIABLE KOOPMAN MOE
# ========================================================================================
def _init_differentiable_koopman_moe():
    """
    differentiable_koopman_moe.py

    Action-label-free Differentiable Mixture over Koopman Experts (D-KMoE).

    This experiment keeps the unsupervised Switching-Koopman candidate structure and
    replaces the selector by a differentiable GRU mixture trained directly with
    trajectory forecasting loss. Each Koopman candidate has a small learnable,
    regularized calibration around its initialized output.

    No behavior/action labels are used for training or model selection.
    The future trajectory is used only as ordinary forecasting supervision.

    Required companion files (same directory):
      - unsup_switch_selector_enhance_standalone_v2.py
      - unsup_koopman_neural_hsmm_selector_v2.py
      - behavior_mode_endpoint_koopman_v2_hmm_selector.py
      - axis_feature_decoupled_position_behavior_koopman_v5.py

    Fast run:
    python differentiable_koopman_moe.py --data_root DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed --fast --out_dir DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed/differentiable_koopman_moe_fast
    """
    import argparse, copy, json, random
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    base = unsup_switch_selector_enhance_standalone_v2
    prior = unsup_koopman_neural_hsmm_selector_v2
    bk = base.bk

    def parse_float_list(s):
        return [float(x.strip()) for x in str(s).split(',') if x.strip()]

    def ensure_dir(p):
        Path(p).mkdir(parents=True, exist_ok=True)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def score(m, args):
        if args.selection_metric in m:
            return m[args.selection_metric]
        return m['pos_rmse'] + args.combo_ade_weight * m['ade'] + args.combo_fde_weight * m['fde']

    def add_result(rows, sample_dfs, data, pred, name, extra=None):
        met = base.compute_position_metrics(data.future_global[:, :, :3], pred)
        row = {'method': name, 'horizon': data.future_global.shape[1], **met}
        if extra:
            row.update(extra)
        rows.append(row)
        sample_dfs.append(base.make_sample_df(data, pred, name))

    class ForecastDataset(Dataset):

        def __init__(self, X, global_pred, cands, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.g = torch.from_numpy(global_pred.astype(np.float32))
            self.c = torch.from_numpy(cands.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return (self.X[i], self.g[i], self.c[i], self.y[i])

    class DifferentiableKMoE(nn.Module):

        def __init__(self, input_dim, K, H, hidden_dim, layers, dropout, max_scale_dev):
            super().__init__()
            self.K = K
            self.max_scale_dev = max_scale_dev
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, K))
            self.raw_scale = nn.Parameter(torch.zeros(K, H, 3))
            self.bias = nn.Parameter(torch.zeros(K, H, 3))

        def forward(self, X, global_pred, cands):
            _, h = self.gru(X)
            z = torch.cat([h[-2], h[-1]], dim=-1)
            pi = torch.softmax(self.head(z), dim=-1)
            scale = 1.0 + self.max_scale_dev * torch.tanh(self.raw_scale)
            calibrated = global_pred[:, None] + scale[None] * (cands - global_pred[:, None]) + self.bias[None]
            pred = (pi[:, :, None, None] * calibrated).sum(dim=1)
            return (pred, pi, scale)

    def forecast_loss(pred, y, args):
        d = pred - y
        rmse = torch.sqrt(torch.mean(d * d) + 1e-10)
        dist = torch.linalg.vector_norm(d, dim=-1)
        return rmse + args.train_ade_weight * dist.mean() + args.train_fde_weight * dist[:, -1].mean()

    def infer(model, X, g, c, bs, device):
        model.eval()
        preds = []
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).float().to(device)
                gb = torch.from_numpy(g[i:i + bs]).float().to(device)
                cb = torch.from_numpy(c[i:i + bs]).float().to(device)
                p, pi, _ = model(xb, gb, cb)
                preds.append(p.cpu().numpy())
                probs.append(pi.cpu().numpy())
        return (np.concatenate(preds), np.concatenate(probs))

    def train_one(Xtr, gtr, ctr, ytr, Xva, gva, cva, yva, args, device, anchor_lam, balance_lam):
        dl = DataLoader(ForecastDataset(Xtr, gtr, ctr, ytr), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == 'cuda')
        model = DifferentiableKMoE(Xtr.shape[-1], ctr.shape[1], ctr.shape[2], args.hidden_dim, args.gru_layers, args.dropout, args.max_scale_deviation).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = None
        state = None
        bad = 0
        history = []
        best_epoch = -1
        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            n = 0
            for xb, gb, cb, yb in dl:
                xb, gb, cb, yb = (xb.to(device), gb.to(device), cb.to(device), yb.to(device))
                pred, pi, scale = model(xb, gb, cb)
                anchor = ((scale - 1) ** 2).mean() + (model.bias ** 2).mean()
                balance = ((pi.mean(0) - 1.0 / model.K) ** 2).mean()
                loss = forecast_loss(pred, yb, args) + anchor_lam * anchor + balance_lam * balance
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                total += loss.item() * len(xb)
                n += len(xb)
            pv, _ = infer(model, Xva, gva, cva, args.batch_size, device)
            met = base.compute_position_metrics(yva, pv)
            sc = score(met, args)
            history.append({'epoch': ep, 'anchor_lambda': anchor_lam, 'balance_lambda': balance_lam, 'train_loss': total / max(n, 1), 'val_score': sc, **{f'val_{k}': v for k, v in met.items()}})
            if best is None or sc < best:
                best = sc
                state = copy.deepcopy(model.state_dict())
                best_epoch = ep
                bad = 0
            else:
                bad += 1
            if ep == 1 or ep % args.log_every == 0 or bad == 0:
                print(f"[D-KMoE a={anchor_lam:g} b={balance_lam:g}] epoch={ep:03d} loss={total / max(n, 1):.6f} val_score={sc:.6f} ADE={met['ade']:.6f} FDE={met['fde']:.6f}")
            if bad >= args.patience:
                break
        model.load_state_dict(state)
        return (model, pd.DataFrame(history), {'anchor_lambda': anchor_lam, 'balance_lambda': balance_lam, 'best_epoch': best_epoch, 'best_raw_val_score': best})

    def tune_safety(val, global_pred, moe_pred, args):
        dis = np.sqrt(np.mean((moe_pred - global_pred) ** 2, axis=(1, 2)))
        best = None
        rows = []
        for q in parse_float_list(args.safety_quantile_grid):
            tau = float(np.quantile(dis, q))
            for a in parse_float_list(args.alpha_grid):
                eff = a * np.minimum(1.0, tau / np.maximum(dis, 1e-12))
                pred = global_pred + eff[:, None, None] * (moe_pred - global_pred)
                met = base.compute_position_metrics(val.future_global[:, :, :3], pred)
                row = {'alpha': a, 'safety_quantile': q, 'safety_tau': tau, 'mean_effective_alpha': float(eff.mean()), 'score': score(met, args), **met}
                rows.append(row)
                if best is None or row['score'] < best['score']:
                    best = row
        return (best, pd.DataFrame(rows))

    def apply_safety(g, p, a, tau):
        dis = np.sqrt(np.mean((p - g) ** 2, axis=(1, 2)))
        eff = a * np.minimum(1.0, tau / np.maximum(dis, 1e-12))
        return (g + eff[:, None, None] * (p - g), eff, dis)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument('--data_root', required=True)
        ap.add_argument('--out_dir', default=None)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizon', type=int, default=10)
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--K', type=int, default=4)
        ap.add_argument('--ridge_model', type=float, default=1.0)
        ap.add_argument('--ridge_gate', type=float, default=1.0)
        ap.add_argument('--min_cluster_samples', type=int, default=80)
        ap.add_argument('--alt_iter', type=int, default=12)
        ap.add_argument('--kmeans_iter', type=int, default=50)
        ap.add_argument('--global_ridge_xy_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--global_ridge_z_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--hidden_dim', type=int, default=128)
        ap.add_argument('--gru_layers', type=int, default=2)
        ap.add_argument('--dropout', type=float, default=0.2)
        ap.add_argument('--max_scale_deviation', type=float, default=0.25)
        ap.add_argument('--anchor_lambda_grid', default='0,0.01,0.1,1')
        ap.add_argument('--balance_lambda_grid', default='0,0.001')
        ap.add_argument('--epochs', type=int, default=100)
        ap.add_argument('--patience', type=int, default=15)
        ap.add_argument('--batch_size', type=int, default=512)
        ap.add_argument('--lr', type=float, default=0.001)
        ap.add_argument('--weight_decay', type=float, default=0.0001)
        ap.add_argument('--grad_clip', type=float, default=1.0)
        ap.add_argument('--log_every', type=int, default=10)
        ap.add_argument('--train_ade_weight', type=float, default=0.5)
        ap.add_argument('--train_fde_weight', type=float, default=0.25)
        ap.add_argument('--selection_metric', default='combo', choices=['pos_rmse', 'ade', 'fde', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--alpha_grid', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        ap.add_argument('--safety_quantile_grid', default='0.25,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0')
        ap.add_argument('--device', default='auto')
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--fast', action='store_true')
        args = ap.parse_args()
        set_seed(args.seed)
        if args.fast:
            args.alt_iter = min(args.alt_iter, 8)
            args.epochs = min(args.epochs, 50)
            args.patience = min(args.patience, 10)
            args.anchor_lambda_grid = '0,0.01,0.1'
            args.balance_lambda_grid = '0,0.001'
            args.safety_quantile_grid = '0.4,0.6,0.8,0.95,1.0'
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        print('device=', device)
        root = Path(args.data_root)
        out = Path(args.out_dir) if args.out_dir else root / 'differentiable_koopman_moe'
        ensure_dir(out)
        trc = bk.load_split_chains(root, 'train')
        vac = bk.load_split_chains(root, 'val')
        tec = bk.load_split_chains(root, 'test')
        tr = bk.make_split_data(trc, args.input_len, args.horizon, args.stride)
        va = bk.make_split_data(vac, args.input_len, args.horizon, args.stride)
        te = bk.make_split_data(tec, args.input_len, args.horizon, args.stride)
        print(f'samples: train={tr.hist_global.shape[0]}, val={va.hist_global.shape[0]}, test={te.hist_global.shape[0]}')
        print('IMPORTANT: no action labels are used.')
        gp = base.fit_no_action_global(tr, va, args)
        gtr = base.predict_no_action_global(gp, tr, args).astype(np.float32)
        gva = base.predict_no_action_global(gp, va, args).astype(np.float32)
        gte = base.predict_no_action_global(gp, te, args).astype(np.float32)
        latent, _, alt = base.fit_switching_model(tr, args.K, args.ridge_model, args.ridge_gate, args, seed=args.seed + args.K)
        alt.to_csv(out / 'latent_alternating_history.csv', index=False)
        ctr = base.candidates_global(latent, tr).astype(np.float32)
        cva = base.candidates_global(latent, va).astype(np.float32)
        cte = base.candidates_global(latent, te).astype(np.float32)
        oracle, _, _, _ = base.oracle_pred(latent, te)
        Xtr0 = prior.make_sequence_features(tr, args.dt)
        Xva0 = prior.make_sequence_features(va, args.dt)
        Xte0 = prior.make_sequence_features(te, args.dt)
        scl = prior.SeqStandardizer.fit(Xtr0)
        Xtr = scl.transform(Xtr0)
        Xva = scl.transform(Xva0)
        Xte = scl.transform(Xte0)
        ytr = tr.future_global[:, :, :3].astype(np.float32)
        yva = va.future_global[:, :, :3].astype(np.float32)
        best_model = None
        best_pack = None
        histories = []
        sels = []
        sweeps = []
        for a in parse_float_list(args.anchor_lambda_grid):
            for b in parse_float_list(args.balance_lambda_grid):
                print(f'\ntraining D-KMoE: anchor={a}, balance={b}')
                model, h, info = train_one(Xtr, gtr, ctr, ytr, Xva, gva, cva, yva, args, device, a, b)
                histories.append(h)
                pv, _ = infer(model, Xva, gva, cva, args.batch_size, device)
                safe, sw = tune_safety(va, gva, pv, args)
                sw['anchor_lambda'] = a
                sw['balance_lambda'] = b
                sweeps.append(sw)
                rec = {**info, 'safe_val_score': safe['score'], 'safe_alpha': safe['alpha'], 'safe_tau': safe['safety_tau'], 'safe_pos_rmse': safe['pos_rmse'], 'safe_ade': safe['ade'], 'safe_fde': safe['fde']}
                sels.append(rec)
                if best_pack is None or safe['score'] < best_pack['safe']['score']:
                    best_model = copy.deepcopy(model).cpu()
                    best_pack = {'info': info, 'safe': safe}
        pd.concat(histories, ignore_index=True).to_csv(out / 'training_history.csv', index=False)
        pd.DataFrame(sels).to_csv(out / 'model_selection_val.csv', index=False)
        pd.concat(sweeps, ignore_index=True).to_csv(out / 'safety_sweep_val.csv', index=False)
        best_model = best_model.to(device)
        raw, pi = infer(best_model, Xte, gte, cte, args.batch_size, device)
        safe, eff, dis = apply_safety(gte, raw, best_pack['safe']['alpha'], best_pack['safe']['safety_tau'])
        rows = []
        samples = []
        add_result(rows, samples, te, gte, 'v5_no_action_global')
        add_result(rows, samples, te, oracle, 'unsup_switch_oracle')
        add_result(rows, samples, te, raw, 'differentiable_koopman_moe_raw', best_pack['info'])
        add_result(rows, samples, te, safe, 'differentiable_koopman_moe_safety', {**best_pack['info'], 'alpha': best_pack['safe']['alpha'], 'safety_tau': best_pack['safe']['safety_tau'], 'mean_effective_alpha_test': float(eff.mean())})
        compare = pd.DataFrame(rows)
        sample_df = pd.concat(samples, ignore_index=True)
        compare.to_csv(out / 'compare_differentiable_koopman_moe_test.csv', index=False)
        sample_df.to_csv(out / 'sample_differentiable_koopman_moe_test.csv', index=False)
        base.group_metrics(sample_df, 'scene_type').to_csv(out / 'grouped_by_scene_test.csv', index=False)
        base.group_metrics(sample_df, 'behavior_key').to_csv(out / 'grouped_by_behavior_key_posthoc_test.csv', index=False)
        pd.DataFrame({'sample_idx': np.arange(te.hist_global.shape[0]), 'disagreement': dis, 'effective_alpha': eff, **{f'prob_mode_{k}': pi[:, k] for k in range(args.K)}}).to_csv(out / 'test_gate_probabilities.csv', index=False)
        torch.save({'model_state_dict': best_model.cpu().state_dict(), 'feature_mean': scl.mean.tolist(), 'feature_std': scl.std.tolist(), 'best': best_pack, 'args': vars(args)}, out / 'differentiable_koopman_moe.pt')
        with open(out / 'report_differentiable_koopman_moe.json', 'w', encoding='utf-8') as f:
            json.dump({'setting': 'action-label-free trajectory-supervised differentiable mixture over Koopman candidates', 'best': best_pack, 'note': 'behavior_key not used in training; grouped behavior table is post-hoc only.'}, f, ensure_ascii=False, indent=2)
        print('\n=== Differentiable Koopman MoE finished ===')
        print(compare.to_string(index=False))
        print('\nOutput directory:', out)

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

differentiable_koopman_moe = _init_differentiable_koopman_moe()


# ========================================================================================
# SECTION: ENV REWEIGHT DIFFERENTIABLE KOOPMAN MOE E1
# ========================================================================================
def _init_env_reweight_differentiable_koopman_moe_e1():
    """
    env_reweight_differentiable_koopman_moe_e1.py

    E1: Environment-based reweighting for action-label-free Differentiable Koopman MoE.

    核心目标：
        不改变 Koopman experts，不使用动作标签；在 D-KMoE 已生成的 K 条候选未来轨迹上，
        用 AirSim 三维静态 voxel map 计算候选轨迹净空，并在 validation set 上选择环境重评分参数。

    模型结构：
        history trajectory -> D-KMoE candidate trajectories + learned pi
        candidate trajectories -> voxel clearance query -> environment penalty
        pi_env = softmax(log(pi) - beta * penalty)
        pred_env = sum_m pi_env[m] * candidate[m]
        pred_final = pred_raw + env_alpha * (pred_env - pred_raw)

    重要坐标约定（已经在当前项目中验证）：
        1) AirSim voxel linear index:
               idx = x + nx * (z + nz * y)
           解码为：
               occ_xyz = flat.reshape((ny, nz, nx)).transpose(2, 0, 1)
        2) 轨迹坐标为 AirSim NED；voxel k 由 Unreal +Z 构造，因此高度映射需反向：
               z_ned(k) = center_z - (k - nz//2) * resolution
        3) 本脚本中的环境查询始终为严格三维查询，绝不依赖 XY 跨高度投影判断可行性。

    依赖文件（与本脚本同目录）：
        differentiable_koopman_moe.py
        unsup_switch_selector_enhance_standalone_v2.py
        unsup_koopman_neural_hsmm_selector_v2.py
        behavior_mode_endpoint_koopman_v2_hmm_selector.py
        axis_feature_decoupled_position_behavior_koopman_v5.py

    Linux 快速运行示例：
    python env_reweight_differentiable_koopman_moe_e1.py --data_root DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed --binvox_file DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/static_map/raw_binvox/city_trajectory_roi_train_val.binvox --out_dir DATA/UAV_AirSim_City_LiDAR_Agent_V3_3_200_40_40/processed/e1_env_reweight_dkmoe_fast --map_expected_xyz 417 359 62 --map_center_xyz 156.5914516449 0.4333877563 -28.8091073036 --map_resolution 1.0 --fast
    """
    import argparse
    import copy
    import json
    import random
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from scipy.ndimage import distance_transform_edt, map_coordinates
    dk = differentiable_koopman_moe
    base = dk.base
    prior = dk.prior
    bk = dk.bk

    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def parse_float_list(text):
        return [float(x.strip()) for x in str(text).split(',') if x.strip()]

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def metric_score(metrics, args):
        if args.selection_metric in metrics:
            return float(metrics[args.selection_metric])
        return float(metrics['pos_rmse'] + args.combo_ade_weight * metrics['ade'] + args.combo_fde_weight * metrics['fde'])

    def softmax_np(logits, axis=-1):
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - np.max(logits, axis=axis, keepdims=True)
        e = np.exp(logits)
        return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)

    class AirSimNEDClearanceMap:
        """AirSim occupancy and clearance map with validated NED-Z handling."""

        def __init__(self, occ_xyz, center_xyz, resolution, header_dims=None, translate=None, scale=None, axis_mode='swap_yz'):
            self.occ = np.asarray(occ_xyz, dtype=bool)
            self.center = np.asarray(center_xyz, dtype=np.float64)
            self.resolution = float(resolution)
            self.header_dims = header_dims
            self.translate = translate
            self.scale = scale
            self.axis_mode = str(axis_mode or 'swap_yz')
            if self.axis_mode not in {'current_xyz', 'swap_yz'}:
                raise ValueError(f'unsupported map_axis_mode={self.axis_mode}')
            self.clearance = distance_transform_edt(~self.occ).astype(np.float32) * self.resolution
            self.shape = self.occ.shape

        @classmethod
        def from_binvox(cls, path, expected_xyz, center_xyz, resolution, axis_mode='swap_yz'):
            path = Path(path)
            with path.open('rb') as f:
                first = f.readline().strip()
                if not first.startswith(b'#binvox'):
                    raise ValueError(f'{path} 不是有效 .binvox 文件。')
                dims = translate = scale = None
                while True:
                    line = f.readline()
                    if not line:
                        raise ValueError('binvox 文件在 data 段之前意外结束。')
                    line = line.strip()
                    if line.startswith(b'dim'):
                        dims = tuple((int(v) for v in line.split()[1:4]))
                    elif line.startswith(b'translate'):
                        translate = tuple((float(v) for v in line.split()[1:4]))
                    elif line.startswith(b'scale'):
                        scale = float(line.split()[1])
                    elif line == b'data':
                        break
                encoded = np.frombuffer(f.read(), dtype=np.uint8)
            if encoded.size % 2 != 0:
                raise ValueError('binvox RLE payload 异常。')
            flat = np.repeat(encoded[0::2], encoded[1::2].astype(np.int64)).astype(bool)
            nx, ny, nz = (int(v) for v in expected_xyz)
            if flat.size != nx * ny * nz:
                raise ValueError(f'解码体素数 {flat.size} 与 expected_xyz={tuple(expected_xyz)} 不一致；header_dims={dims}。')
            occ_xyz = flat.reshape((ny, nz, nx)).transpose(2, 0, 1)
            if occ_xyz.shape != (nx, ny, nz):
                raise RuntimeError(f'恢复后的 occupancy shape={occ_xyz.shape}，期望 {(nx, ny, nz)}。')
            return cls(occ_xyz, center_xyz, resolution, dims, translate, scale, axis_mode=axis_mode)

        def voxel_centres_ned(self):
            nx, ny, nz = self.shape
            cx, cy, cz = self.center
            r = self.resolution
            xs = cx + (np.arange(nx) - nx // 2) * r
            if self.axis_mode == 'swap_yz':
                # Current AirSim city.binvox: array axis 1 is physical height, array axis 2 is physical NED-Y.
                ys = cy + (np.arange(nz) - nz // 2) * r
                zs = cz - (np.arange(ny) - ny // 2) * r
            else:
                ys = cy + (np.arange(ny) - ny // 2) * r
                zs = cz - (np.arange(nz) - nz // 2) * r
            return (xs, ys, zs)

        def fractional_indices(self, points_xyz):
            points = np.asarray(points_xyz, dtype=np.float64)
            nx, ny, nz = self.shape
            cx, cy, cz = self.center
            r = self.resolution
            if self.axis_mode == 'swap_yz':
                return np.stack([
                    (points[..., 0] - cx) / r + nx // 2,
                    (cz - points[..., 2]) / r + ny // 2,
                    (points[..., 1] - cy) / r + nz // 2,
                ], axis=-1)
            return np.stack([
                (points[..., 0] - cx) / r + nx // 2,
                (points[..., 1] - cy) / r + ny // 2,
                (cz - points[..., 2]) / r + nz // 2,
            ], axis=-1)

        def query_clearance(self, points_xyz):
            """Trilinear clearance query. Points outside the map are treated as zero-clearance/unsafe."""
            points = np.asarray(points_xyz, dtype=np.float64)
            original_shape = points.shape[:-1]
            frac = self.fractional_indices(points).reshape(-1, 3)
            shape = np.asarray(self.shape, dtype=np.float64)
            valid = ((frac >= 0.0) & (frac <= (shape - 1.0)[None, :])).all(axis=1)
            values = map_coordinates(self.clearance, frac.T, order=1, mode='constant', cval=0.0, prefilter=False)
            values[~valid] = 0.0
            return (values.reshape(original_shape), valid.reshape(original_shape))

        def query_occupied_nearest(self, points_xyz):
            """Strict nearest tested-voxel occupancy query used only for diagnostics."""
            points = np.asarray(points_xyz, dtype=np.float64)
            original_shape = points.shape[:-1]
            frac = self.fractional_indices(points).reshape(-1, 3)
            idx = np.floor(frac + 0.5).astype(np.int64)
            shape = np.asarray(self.shape, dtype=np.int64)
            valid = ((idx >= 0) & (idx < shape[None, :])).all(axis=1)
            occupied = np.ones(len(idx), dtype=bool)
            if valid.any():
                vi = idx[valid]
                occupied[valid] = self.occ[vi[:, 0], vi[:, 1], vi[:, 2]]
            return (occupied.reshape(original_shape), valid.reshape(original_shape))

        def metadata(self):
            xs, ys, zs = self.voxel_centres_ned()
            return {'header_dims': list(self.header_dims) if self.header_dims is not None else None, 'decoded_shape_xyz': list(self.shape), 'center_xyz': self.center.tolist(), 'resolution': self.resolution, 'axis_mode': self.axis_mode, 'ned_bounds_x': [float(xs[0]), float(xs[-1])], 'ned_bounds_y': [float(ys[0]), float(ys[-1])], 'ned_bounds_z': [float(zs[0]), float(zs[-1])], 'overall_occupancy_ratio': float(self.occ.mean()), 'decoding_rule': 'occ_xyz = flat.reshape((ny,nz,nx)).transpose(2,0,1)', 'ned_mapping_rule': 'swap_yz: x=cx+(i-nx//2)r, y=cy+(k-nz//2)r, z=cz-(j-ny//2)r; current_xyz kept for legacy'}

    def infer_full(model, X, global_pred, candidates, batch_size, device):
        model.eval()
        preds, probs, calibrated_all = ([], [], [])
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start:start + batch_size]).float().to(device)
                gb = torch.from_numpy(global_pred[start:start + batch_size]).float().to(device)
                cb = torch.from_numpy(candidates[start:start + batch_size]).float().to(device)
                pred, pi, scale = model(xb, gb, cb)
                calibrated = gb[:, None] + scale[None] * (cb - gb[:, None]) + model.bias[None]
                preds.append(pred.cpu().numpy())
                probs.append(pi.cpu().numpy())
                calibrated_all.append(calibrated.cpu().numpy())
        return (np.concatenate(preds), np.concatenate(probs), np.concatenate(calibrated_all))

    def candidate_environment_features(voxel_map, candidate_trajs, terminal_weight):
        """
        candidate_trajs: [N, K, H, 3] global NED positions.
        Returns clearance and compact candidate-level fields.
        """
        clearance, valid = voxel_map.query_clearance(candidate_trajs)
        occupied, _ = voxel_map.query_occupied_nearest(candidate_trajs)
        H = candidate_trajs.shape[2]
        weights = np.linspace(1.0, float(terminal_weight), H, dtype=np.float64)
        weights /= weights.sum()
        return {'clearance': clearance.astype(np.float32), 'valid': valid, 'occupied': occupied, 'min_clearance': clearance.min(axis=2), 'mean_clearance': clearance.mean(axis=2), 'terminal_clearance': clearance[:, :, -1], 'weighted_clearance': (clearance * weights[None, None, :]).sum(axis=2), 'occupied_steps': occupied.sum(axis=2).astype(np.int64), 'outside_steps': (~valid).sum(axis=2).astype(np.int64)}

    def environment_penalty(clearance, safe_distance, terminal_weight):
        """Soft barrier penalty. No hard rejection: robust to remaining voxel boundary uncertainty."""
        H = clearance.shape[2]
        weights = np.linspace(1.0, float(terminal_weight), H, dtype=np.float64)
        weights /= weights.sum()
        shortfall = np.maximum(float(safe_distance) - clearance, 0.0)
        return (shortfall ** 2 * weights[None, None, :]).sum(axis=2)

    def apply_environment_reweight(raw_pred, raw_pi, calibrated_candidates, candidate_clearance, safe_distance, beta, env_alpha, terminal_weight):
        penalty = environment_penalty(candidate_clearance, safe_distance, terminal_weight)
        logits = np.log(np.maximum(raw_pi, 1e-12)) - float(beta) * penalty
        env_pi = softmax_np(logits, axis=1).astype(np.float32)
        env_mix = (env_pi[:, :, None, None] * calibrated_candidates).sum(axis=1)
        final_pred = raw_pred + float(env_alpha) * (env_mix - raw_pred)
        return (final_pred.astype(np.float32), env_pi, penalty.astype(np.float32), env_mix.astype(np.float32))

    def tune_environment_reweight(val_data, raw_pred, raw_pi, calibrated_candidates, cand_clearance, args):
        rows = []
        best = None
        for d_safe in parse_float_list(args.env_safe_distance_grid):
            for beta in parse_float_list(args.env_beta_grid):
                for alpha in parse_float_list(args.env_alpha_grid):
                    pred, env_pi, penalty, _ = apply_environment_reweight(raw_pred, raw_pi, calibrated_candidates, cand_clearance, d_safe, beta, alpha, args.env_terminal_weight)
                    metrics = base.compute_position_metrics(val_data.future_global[:, :, :3], pred)
                    rec = {'env_safe_distance': d_safe, 'env_beta': beta, 'env_alpha': alpha, 'score': metric_score(metrics, args), 'mean_penalty': float(penalty.mean()), 'mean_gate_l1_change': float(np.abs(env_pi - raw_pi).sum(axis=1).mean()), **metrics}
                    rows.append(rec)
                    if best is None or rec['score'] < best['score']:
                        best = rec
        return (best, pd.DataFrame(rows))

    def tune_global_safety_from_pred(val_data, global_pred, proposed_pred, args):
        return dk.tune_safety(val_data, global_pred, proposed_pred, args)

    def prediction_map_diagnostics(voxel_map, pred, unsafe_clearance):
        clearance, valid = voxel_map.query_clearance(pred)
        occupied, _ = voxel_map.query_occupied_nearest(pred)
        min_clearance = clearance.min(axis=1)
        return {'pred_mean_min_clearance': float(min_clearance.mean()), 'pred_median_min_clearance': float(np.median(min_clearance)), 'pred_unsafe_sample_rate': float(np.mean(min_clearance < float(unsafe_clearance))), 'pred_occupied_sample_rate': float(np.mean(occupied.any(axis=1))), 'pred_outside_map_sample_rate': float(np.mean((~valid).any(axis=1)))}

    def add_result(rows, sample_frames, data, pred, name, voxel_map, args, extra=None):
        metrics = base.compute_position_metrics(data.future_global[:, :, :3], pred)
        row = {'method': name, 'horizon': data.future_global.shape[1], **metrics}
        row.update(prediction_map_diagnostics(voxel_map, pred, args.unsafe_clearance))
        if extra:
            row.update(extra)
        rows.append(row)
        sample_frames.append(base.make_sample_df(data, pred, name))

    def subset_metrics(test_data, predictions, voxel_map, candidate_feat, args):
        gt = test_data.future_global[:, :, :3]
        gt_clearance, _ = voxel_map.query_clearance(gt)
        gt_min_clearance = gt_clearance.min(axis=1)
        gt_hit, _ = voxel_map.query_occupied_nearest(gt)
        candidate_gap = candidate_feat['min_clearance'].max(axis=1) - candidate_feat['min_clearance'].min(axis=1)
        masks = {'all': np.ones(len(gt), dtype=bool), f'gt_near_obstacle_le_{args.report_near_clearance:g}m': gt_min_clearance <= args.report_near_clearance, f'gt_open_gt_{args.report_near_clearance:g}m': gt_min_clearance > args.report_near_clearance, f'candidate_clearance_gap_ge_{args.report_candidate_gap:g}m': candidate_gap >= args.report_candidate_gap, 'gt_strict_voxel_hit_diagnostic': gt_hit.any(axis=1)}
        rows = []
        for subset_name, mask in masks.items():
            if int(mask.sum()) == 0:
                continue
            for method, pred in predictions.items():
                met = base.compute_position_metrics(gt[mask], pred[mask])
                diag = prediction_map_diagnostics(voxel_map, pred[mask], args.unsafe_clearance)
                rows.append({'subset': subset_name, 'num_samples': int(mask.sum()), 'method': method, **met, **diag})
        sample_env = pd.DataFrame({'sample_idx': np.arange(len(gt), dtype=int), 'gt_min_clearance': gt_min_clearance, 'gt_has_voxel_hit_diagnostic': gt_hit.any(axis=1).astype(int), 'candidate_min_clearance_min': candidate_feat['min_clearance'].min(axis=1), 'candidate_min_clearance_max': candidate_feat['min_clearance'].max(axis=1), 'candidate_min_clearance_gap': candidate_gap})
        return (pd.DataFrame(rows), sample_env)

    def main():
        ap = argparse.ArgumentParser(description='E1: environment reweighting over Differentiable Koopman MoE candidates.')
        ap.add_argument('--data_root', required=True)
        ap.add_argument('--out_dir', default=None)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizon', type=int, default=10)
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--K', type=int, default=4)
        ap.add_argument('--ridge_model', type=float, default=1.0)
        ap.add_argument('--ridge_gate', type=float, default=1.0)
        ap.add_argument('--min_cluster_samples', type=int, default=80)
        ap.add_argument('--alt_iter', type=int, default=12)
        ap.add_argument('--kmeans_iter', type=int, default=50)
        ap.add_argument('--global_ridge_xy_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--global_ridge_z_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--hidden_dim', type=int, default=128)
        ap.add_argument('--gru_layers', type=int, default=2)
        ap.add_argument('--dropout', type=float, default=0.2)
        ap.add_argument('--max_scale_deviation', type=float, default=0.25)
        ap.add_argument('--anchor_lambda_grid', default='0,0.01,0.1,1')
        ap.add_argument('--balance_lambda_grid', default='0,0.001')
        ap.add_argument('--epochs', type=int, default=100)
        ap.add_argument('--patience', type=int, default=15)
        ap.add_argument('--batch_size', type=int, default=512)
        ap.add_argument('--lr', type=float, default=0.001)
        ap.add_argument('--weight_decay', type=float, default=0.0001)
        ap.add_argument('--grad_clip', type=float, default=1.0)
        ap.add_argument('--log_every', type=int, default=10)
        ap.add_argument('--train_ade_weight', type=float, default=0.5)
        ap.add_argument('--train_fde_weight', type=float, default=0.25)
        ap.add_argument('--selection_metric', default='combo', choices=['pos_rmse', 'ade', 'fde', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--alpha_grid', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        ap.add_argument('--safety_quantile_grid', default='0.25,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0')
        ap.add_argument('--device', default='auto')
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--fast', action='store_true')
        ap.add_argument('--binvox_file', required=True)
        ap.add_argument('--map_expected_xyz', nargs=3, type=int, required=True)
        ap.add_argument('--map_center_xyz', nargs=3, type=float, required=True)
        ap.add_argument('--map_resolution', type=float, default=1.0)
        ap.add_argument('--map_axis_mode', default='swap_yz', choices=['swap_yz', 'current_xyz'])
        p.add_argument('--map_axis_mode', default='swap_yz', choices=['swap_yz', 'current_xyz'])
        ap.add_argument('--env_safe_distance_grid', default='0,0.5,1,2,3,5')
        ap.add_argument('--env_beta_grid', default='0,0.03,0.1,0.3,1,3,10')
        ap.add_argument('--env_alpha_grid', default='0,0.25,0.5,0.75,1')
        ap.add_argument('--env_terminal_weight', type=float, default=2.0)
        ap.add_argument('--unsafe_clearance', type=float, default=1.0)
        ap.add_argument('--report_near_clearance', type=float, default=5.0)
        ap.add_argument('--report_candidate_gap', type=float, default=2.0)
        args = ap.parse_args()
        set_seed(args.seed)
        if args.fast:
            args.alt_iter = min(args.alt_iter, 8)
            args.epochs = min(args.epochs, 50)
            args.patience = min(args.patience, 10)
            args.anchor_lambda_grid = '0,0.01,0.1'
            args.balance_lambda_grid = '0,0.001'
            args.safety_quantile_grid = '0.4,0.6,0.8,0.95,1.0'
            args.env_safe_distance_grid = '0,0.5,1,2,3,5'
            args.env_beta_grid = '0,0.03,0.1,0.3,1,3,10'
            args.env_alpha_grid = '0,0.25,0.5,0.75,1'
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        print('device =', device)
        print('IMPORTANT: no action labels are used in model training or environment reweight tuning.')
        print('IMPORTANT: voxel queries use validated AirSim NED-Z mapping.')
        root = Path(args.data_root)
        out = Path(args.out_dir) if args.out_dir else root / 'e1_env_reweight_dkmoe'
        ensure_dir(out)
        voxel_map = AirSimNEDClearanceMap.from_binvox(args.binvox_file, args.map_expected_xyz, args.map_center_xyz, args.map_resolution, axis_mode=getattr(args, 'map_axis_mode', 'swap_yz'))
        with open(out / 'map_metadata_validated_ned.json', 'w', encoding='utf-8') as f:
            json.dump(voxel_map.metadata(), f, ensure_ascii=False, indent=2)
        print('voxel map metadata:', json.dumps(voxel_map.metadata(), ensure_ascii=False))
        trc = bk.load_split_chains(root, 'train')
        vac = bk.load_split_chains(root, 'val')
        tec = bk.load_split_chains(root, 'test')
        tr = bk.make_split_data(trc, args.input_len, args.horizon, args.stride)
        va = bk.make_split_data(vac, args.input_len, args.horizon, args.stride)
        te = bk.make_split_data(tec, args.input_len, args.horizon, args.stride)
        print(f'samples: train={tr.hist_global.shape[0]}, val={va.hist_global.shape[0]}, test={te.hist_global.shape[0]}')
        global_pack = base.fit_no_action_global(tr, va, args)
        gtr = base.predict_no_action_global(global_pack, tr, args).astype(np.float32)
        gva = base.predict_no_action_global(global_pack, va, args).astype(np.float32)
        gte = base.predict_no_action_global(global_pack, te, args).astype(np.float32)
        latent, _, alt_history = base.fit_switching_model(tr, args.K, args.ridge_model, args.ridge_gate, args, seed=args.seed + args.K)
        alt_history.to_csv(out / 'latent_alternating_history.csv', index=False)
        ctr = base.candidates_global(latent, tr).astype(np.float32)
        cva = base.candidates_global(latent, va).astype(np.float32)
        cte = base.candidates_global(latent, te).astype(np.float32)
        oracle_test, _, _, _ = base.oracle_pred(latent, te)
        Xtr0 = prior.make_sequence_features(tr, args.dt)
        Xva0 = prior.make_sequence_features(va, args.dt)
        Xte0 = prior.make_sequence_features(te, args.dt)
        standardizer = prior.SeqStandardizer.fit(Xtr0)
        Xtr = standardizer.transform(Xtr0)
        Xva = standardizer.transform(Xva0)
        Xte = standardizer.transform(Xte0)
        ytr = tr.future_global[:, :, :3].astype(np.float32)
        yva = va.future_global[:, :, :3].astype(np.float32)
        best_model = None
        best_pack = None
        train_histories = []
        base_model_selection = []
        base_safety_sweeps = []
        for anchor_lambda in parse_float_list(args.anchor_lambda_grid):
            for balance_lambda in parse_float_list(args.balance_lambda_grid):
                print(f'\ntraining baseline D-KMoE: anchor={anchor_lambda}, balance={balance_lambda}')
                model, hist, info = dk.train_one(Xtr, gtr, ctr, ytr, Xva, gva, cva, yva, args, device, anchor_lambda, balance_lambda)
                train_histories.append(hist)
                val_raw, _ = dk.infer(model, Xva, gva, cva, args.batch_size, device)
                val_safe, safety_sweep = dk.tune_safety(va, gva, val_raw, args)
                safety_sweep['anchor_lambda'] = anchor_lambda
                safety_sweep['balance_lambda'] = balance_lambda
                base_safety_sweeps.append(safety_sweep)
                rec = {**info, 'safe_val_score': val_safe['score'], 'safe_alpha': val_safe['alpha'], 'safe_tau': val_safe['safety_tau'], 'safe_pos_rmse': val_safe['pos_rmse'], 'safe_ade': val_safe['ade'], 'safe_fde': val_safe['fde']}
                base_model_selection.append(rec)
                if best_pack is None or val_safe['score'] < best_pack['safety']['score']:
                    best_model = copy.deepcopy(model).cpu()
                    best_pack = {'info': info, 'safety': val_safe}
        pd.concat(train_histories, ignore_index=True).to_csv(out / 'baseline_dkmoe_training_history.csv', index=False)
        pd.DataFrame(base_model_selection).to_csv(out / 'baseline_dkmoe_model_selection_val.csv', index=False)
        pd.concat(base_safety_sweeps, ignore_index=True).to_csv(out / 'baseline_dkmoe_safety_sweep_val.csv', index=False)
        best_model = best_model.to(device)
        va_raw, va_pi, va_calibrated = infer_full(best_model, Xva, gva, cva, args.batch_size, device)
        te_raw, te_pi, te_calibrated = infer_full(best_model, Xte, gte, cte, args.batch_size, device)
        te_baseline_safe, te_eff, _ = dk.apply_safety(gte, te_raw, best_pack['safety']['alpha'], best_pack['safety']['safety_tau'])
        va_env_feat = candidate_environment_features(voxel_map, va_calibrated, args.env_terminal_weight)
        te_env_feat = candidate_environment_features(voxel_map, te_calibrated, args.env_terminal_weight)
        best_env, env_sweep = tune_environment_reweight(va, va_raw, va_pi, va_calibrated, va_env_feat['clearance'], args)
        env_sweep.to_csv(out / 'e1_environment_reweight_sweep_val.csv', index=False)
        print('\nselected E1 environment parameters on validation:')
        print(best_env)
        va_env, va_env_pi, va_penalty, _ = apply_environment_reweight(va_raw, va_pi, va_calibrated, va_env_feat['clearance'], best_env['env_safe_distance'], best_env['env_beta'], best_env['env_alpha'], args.env_terminal_weight)
        te_env, te_env_pi, te_penalty, _ = apply_environment_reweight(te_raw, te_pi, te_calibrated, te_env_feat['clearance'], best_env['env_safe_distance'], best_env['env_beta'], best_env['env_alpha'], args.env_terminal_weight)
        env_safety_best, env_safety_sweep = tune_global_safety_from_pred(va, gva, va_env, args)
        env_safety_sweep.to_csv(out / 'e1_environment_plus_global_safety_sweep_val.csv', index=False)
        te_env_safe, te_env_eff, _ = dk.apply_safety(gte, te_env, env_safety_best['alpha'], env_safety_best['safety_tau'])
        rows = []
        sample_frames = []
        add_result(rows, sample_frames, te, gte, 'v5_no_action_global', voxel_map, args)
        add_result(rows, sample_frames, te, oracle_test, 'unsup_switch_oracle', voxel_map, args)
        add_result(rows, sample_frames, te, te_raw, 'differentiable_koopman_moe_raw', voxel_map, args, best_pack['info'])
        add_result(rows, sample_frames, te, te_baseline_safe, 'differentiable_koopman_moe_safety', voxel_map, args, {**best_pack['info'], 'alpha': best_pack['safety']['alpha'], 'safety_tau': best_pack['safety']['safety_tau'], 'mean_effective_alpha_test': float(te_eff.mean())})
        add_result(rows, sample_frames, te, te_env, 'e1_env_reweight_dkmoe', voxel_map, args, {'env_safe_distance': best_env['env_safe_distance'], 'env_beta': best_env['env_beta'], 'env_alpha': best_env['env_alpha'], 'mean_gate_l1_change_test': float(np.abs(te_env_pi - te_pi).sum(axis=1).mean())})
        add_result(rows, sample_frames, te, te_env_safe, 'e1_env_reweight_plus_global_safety', voxel_map, args, {'env_safe_distance': best_env['env_safe_distance'], 'env_beta': best_env['env_beta'], 'env_alpha': best_env['env_alpha'], 'alpha': env_safety_best['alpha'], 'safety_tau': env_safety_best['safety_tau'], 'mean_effective_alpha_test': float(te_env_eff.mean())})
        comparison = pd.DataFrame(rows)
        comparison.to_csv(out / 'compare_e1_environment_reweight_test.csv', index=False)
        sample_df = pd.concat(sample_frames, ignore_index=True)
        sample_df.to_csv(out / 'sample_e1_environment_reweight_test.csv', index=False)
        base.group_metrics(sample_df, 'scene_type').to_csv(out / 'grouped_by_scene_test.csv', index=False)
        base.group_metrics(sample_df, 'behavior_key').to_csv(out / 'grouped_by_behavior_key_posthoc_test.csv', index=False)
        predictions = {'v5_no_action_global': gte, 'differentiable_koopman_moe_raw': te_raw, 'differentiable_koopman_moe_safety': te_baseline_safe, 'e1_env_reweight_dkmoe': te_env, 'e1_env_reweight_plus_global_safety': te_env_safe}
        subset_df, sample_env_df = subset_metrics(te, predictions, voxel_map, te_env_feat, args)
        subset_df.to_csv(out / 'e1_subset_metrics_test.csv', index=False)
        gate_details = pd.DataFrame({'sample_idx': np.arange(te.hist_global.shape[0]), 'candidate_min_clearance_min': te_env_feat['min_clearance'].min(axis=1), 'candidate_min_clearance_max': te_env_feat['min_clearance'].max(axis=1), 'candidate_min_clearance_gap': te_env_feat['min_clearance'].max(axis=1) - te_env_feat['min_clearance'].min(axis=1), 'candidate_any_occupied': te_env_feat['occupied_steps'].max(axis=1) > 0, 'candidate_any_outside_map': te_env_feat['outside_steps'].max(axis=1) > 0, 'gate_l1_change': np.abs(te_env_pi - te_pi).sum(axis=1)})
        for k in range(args.K):
            gate_details[f'raw_pi_mode_{k}'] = te_pi[:, k]
            gate_details[f'env_pi_mode_{k}'] = te_env_pi[:, k]
            gate_details[f'mode_{k}_min_clearance'] = te_env_feat['min_clearance'][:, k]
            gate_details[f'mode_{k}_mean_clearance'] = te_env_feat['mean_clearance'][:, k]
            gate_details[f'mode_{k}_penalty'] = te_penalty[:, k]
        gate_details = gate_details.merge(sample_env_df, on='sample_idx', how='left')
        gate_details.to_csv(out / 'e1_test_environment_gate_details.csv', index=False)
        torch.save({'model_state_dict': best_model.cpu().state_dict(), 'feature_mean': standardizer.mean.tolist(), 'feature_std': standardizer.std.tolist(), 'baseline_best': best_pack, 'environment_best': best_env, 'environment_plus_global_safety_best': env_safety_best, 'args': vars(args), 'map_metadata': voxel_map.metadata()}, out / 'e1_env_reweight_dkmoe.pt')
        report = {'setting': 'action-label-free D-KMoE with validation-tuned 3D static-environment candidate reweighting', 'environment_usage': 'environment only reweights Koopman candidate mixture weights; it does not modify candidate dynamics or use action labels', 'coordinate_safety_rules': voxel_map.metadata(), 'baseline_dkmoe_best': best_pack, 'e1_environment_best': best_env, 'e1_environment_plus_global_safety_best': env_safety_best, 'test_comparison': comparison.to_dict(orient='records'), 'notes': ['GT future clearance is used only for post-hoc subset reporting, not for prediction or parameter selection.', 'behavior_key grouped output is post-hoc reporting only; action labels are not used during training or tuning.', 'A safe-distance value of 0 or env_alpha=0 selected on validation indicates environment reweighting did not improve the main validation metric in this run.']}
        with open(out / 'report_e1_environment_reweight.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('\n=== E1 Environment Reweighting over D-KMoE finished ===')
        print(comparison.to_string(index=False))
        print('\nSelected E1 env parameters:', best_env)
        print('Selected E1 + global safety parameters:', env_safety_best)
        print('\nKey files:')
        for name in ['compare_e1_environment_reweight_test.csv', 'e1_environment_reweight_sweep_val.csv', 'e1_subset_metrics_test.csv', 'e1_test_environment_gate_details.csv', 'report_e1_environment_reweight.json']:
            print(' ', out / name)

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

env_reweight_differentiable_koopman_moe_e1 = _init_env_reweight_differentiable_koopman_moe_e1()


# ========================================================================================
# SECTION: E2 LEARNED ENVIRONMENT CONDITIONED KOOPMAN SELECTOR
# ========================================================================================
def _init_e2_learned_environment_conditioned_koopman_selector():
    """
    e2_learned_environment_conditioned_koopman_selector.py

    E2: Learned Environment-Conditioned Selector over Differentiable Koopman Candidates
    ======================================================================================

    目标：
        验证静态三维环境信息是否能够被一个小型可训练 selector 真正利用，
        从 K 条 Koopman candidates 中选择更符合真实轨迹的混合结果。

    实验结构：
        1. 不使用动作标签，复现固定 K 的 D-KMoE 候选与基础 gate；
        2. 冻结 Koopman candidates 与基础 D-KMoE；
        3. 训练残差 selector，只修正 candidate mixture weights；
        4. 对照三个版本：
             - e2_noenv_selector: 历史运动 + candidate 动态特征，不提供地图信息；
             - e2_env_selector: 在同样结构上加入 candidate 三维环境特征；
             - e2_env_gated_selector: 环境特征 + 仅在近障碍预测附近激活修正；
        5. 如果 env / env_gated 优于 noenv，才可认为环境信息带来额外预测价值。

    重要实验口径：
        - 不使用行为/动作标签训练或选模；
        - 未来轨迹仅作为正常预测监督；
        - Koopman experts 与基础 D-KMoE 先训练后冻结，E2 只学习 selector 修正；
        - 所有超参数仅在 validation 上选；test 只报告固定变体结果；
        - 地图坐标严格遵守已验证的 AirSim binvox + NED-Z 约定。

    已验证地图约定：
        idx = x + nx * (z + nz * y)
        occ_xyz = flat.reshape((ny, nz, nx)).transpose(2, 0, 1)
        z_ned(k) = center_z - (k - nz//2) * resolution
    """
    import argparse
    import copy
    import json
    import random
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    e1 = env_reweight_differentiable_koopman_moe_e1
    dk = e1.dk
    base = e1.base
    prior = e1.prior
    bk = e1.bk

    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def parse_float_list(text):
        return [float(x.strip()) for x in str(text).split(',') if x.strip()]

    def parse_str_list(text):
        return [x.strip() for x in str(text).split(',') if x.strip()]

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def selection_score(metrics, args):
        return e1.metric_score(metrics, args)

    class FlatStandardizer:

        def __init__(self, mean, std):
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        @classmethod
        def fit(cls, x, eps=1e-06):
            flat = x.reshape(-1, x.shape[-1])
            return cls(flat.mean(axis=0), np.maximum(flat.std(axis=0), eps))

        def transform(self, x):
            return ((x - self.mean) / self.std).astype(np.float32)

    def current_heading_features(data):
        """当前水平运动方向；低速时使用零方向，并提供 valid 标记。"""
        v = np.asarray(data.hist_global[:, -1, 3:6], dtype=np.float32)
        vxy = v[:, :2]
        speed = np.linalg.norm(vxy, axis=1, keepdims=True)
        valid = (speed > 0.0001).astype(np.float32)
        unit = vxy / np.maximum(speed, 0.0001)
        unit *= valid
        return (unit, valid, v)

    def build_dynamic_candidate_features(data, global_pred, raw_pred, calibrated_candidates, raw_pi):
        """
        仅由历史状态、基础预测与 Koopman candidates 构造，不使用地图或动作标签。
        输出 [N,K,F_dyn]。
        """
        n, k, h, _ = calibrated_candidates.shape
        last = np.asarray(data.hist_global[:, -1, :3], dtype=np.float32)
        heading_xy, heading_valid, velocity = current_heading_features(data)
        cand_end_disp = calibrated_candidates[:, :, -1, :] - last[:, None, :]
        cand_mean_disp = calibrated_candidates.mean(axis=2) - last[:, None, :]
        cand_vs_raw_end = calibrated_candidates[:, :, -1, :] - raw_pred[:, None, -1, :]
        cand_vs_raw_mean = calibrated_candidates.mean(axis=2) - raw_pred[:, None].mean(axis=2)
        cand_vs_global_end = calibrated_candidates[:, :, -1, :] - global_pred[:, None, -1, :]
        end_norm = np.linalg.norm(cand_end_disp, axis=2, keepdims=True)
        versus_raw_traj = np.linalg.norm(calibrated_candidates - raw_pred[:, None, :, :], axis=3)
        mean_dist_to_raw = versus_raw_traj.mean(axis=2, keepdims=True)
        terminal_dist_to_raw = versus_raw_traj[:, :, -1, None]
        disp_xy = cand_end_disp[:, :, :2]
        disp_xy_norm = np.linalg.norm(disp_xy, axis=2, keepdims=True)
        disp_xy_unit = disp_xy / np.maximum(disp_xy_norm, 1e-05)
        heading_repeat = heading_xy[:, None, :]
        align = (disp_xy_unit * heading_repeat).sum(axis=2, keepdims=True)
        cross = heading_repeat[:, :, 0:1] * disp_xy_unit[:, :, 1:2] - heading_repeat[:, :, 1:2] * disp_xy_unit[:, :, 0:1]
        heading_valid_repeat = np.repeat(heading_valid[:, None, :], k, axis=1)
        pi = raw_pi[:, :, None]
        log_pi = np.log(np.maximum(pi, 1e-08))
        features = np.concatenate([cand_end_disp, cand_mean_disp, cand_vs_raw_end, cand_vs_raw_mean, cand_vs_global_end, end_norm, mean_dist_to_raw, terminal_dist_to_raw, align, cross, heading_valid_repeat, pi, log_pi], axis=2)
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def build_environment_candidate_features(voxel_map, raw_pred, calibrated_candidates, terminal_weight):
        """
        从正确 NED 映射的三维 voxel clearance map 构造 candidate 环境特征。
        输出 [N,K,F_env]，不含 GT 信息。
        """
        feat = e1.candidate_environment_features(voxel_map, calibrated_candidates, terminal_weight)
        raw_clearance, _ = voxel_map.query_clearance(raw_pred)
        raw_min = raw_clearance.min(axis=1)
        raw_mean = raw_clearance.mean(axis=1)
        raw_terminal = raw_clearance[:, -1]
        h = calibrated_candidates.shape[2]
        clearance = feat['clearance']
        unsafe_1 = (clearance < 1.0).mean(axis=2)
        unsafe_3 = (clearance < 3.0).mean(axis=2)
        unsafe_5 = (clearance < 5.0).mean(axis=2)
        env = np.stack([feat['min_clearance'], feat['mean_clearance'], feat['terminal_clearance'], feat['weighted_clearance'], feat['min_clearance'] - raw_min[:, None], feat['mean_clearance'] - raw_mean[:, None], feat['terminal_clearance'] - raw_terminal[:, None], unsafe_1, unsafe_3, unsafe_5, feat['occupied_steps'].astype(np.float32) / float(h), feat['outside_steps'].astype(np.float32) / float(h), np.repeat(raw_min[:, None], calibrated_candidates.shape[1], axis=1), np.repeat(raw_mean[:, None], calibrated_candidates.shape[1], axis=1)], axis=2).astype(np.float32)
        return (env, feat, raw_min.astype(np.float32))

    def environment_activation(raw_min_clearance, gate_distance, temperature):
        raw_min = np.asarray(raw_min_clearance, dtype=np.float32)
        temperature = max(float(temperature), 1e-06)
        act = 1.0 / (1.0 + np.exp((raw_min - float(gate_distance)) / temperature))
        return act.astype(np.float32)

    def make_environment_penalty(candidate_feat, safe_distance, terminal_weight):
        return e1.environment_penalty(candidate_feat['clearance'], safe_distance, terminal_weight).astype(np.float32)

    def gt_environment_mask(voxel_map, future_gt, ignore_clearance):
        gt_clearance, _ = voxel_map.query_clearance(future_gt)
        gt_min = gt_clearance.min(axis=1)
        mask = (gt_min > float(ignore_clearance)).astype(np.float32)
        return (mask, gt_min.astype(np.float32))

    class E2Dataset(Dataset):

        def __init__(self, X, raw_pred, raw_pi, candidates, cand_features, activation, penalty, gt_mask, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.raw_pred = torch.from_numpy(raw_pred.astype(np.float32))
            self.raw_pi = torch.from_numpy(raw_pi.astype(np.float32))
            self.candidates = torch.from_numpy(candidates.astype(np.float32))
            self.features = torch.from_numpy(cand_features.astype(np.float32))
            self.activation = torch.from_numpy(activation.astype(np.float32))
            self.penalty = torch.from_numpy(penalty.astype(np.float32))
            self.gt_mask = torch.from_numpy(gt_mask.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return (self.X[i], self.raw_pred[i], self.raw_pi[i], self.candidates[i], self.features[i], self.activation[i], self.penalty[i], self.gt_mask[i], self.y[i])

    class LearnedResidualCandidateSelector(nn.Module):
        """
        在冻结的基础 D-KMoE gate 上学习 residual logits，不修改 Koopman candidates。
        variant:
            noenv      : 动态 candidate 特征，不使用环境；
            env        : 动态 + 环境特征；
            env_gated  : 动态 + 环境特征，修正量由近障碍 activation 缩放。
        """

        def __init__(self, seq_dim, cand_dim, hidden_dim, layers, dropout, max_logit_correction, variant):
            super().__init__()
            self.variant = variant
            self.max_logit_correction = float(max_logit_correction)
            self.gru = nn.GRU(seq_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=True)
            self.candidate_encoder = nn.Sequential(nn.Linear(cand_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            self.scorer = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

        def forward(self, X, raw_pi, candidates, cand_features, activation):
            _, h = self.gru(X)
            history = torch.cat([h[-2], h[-1]], dim=-1)
            cand_embed = self.candidate_encoder(cand_features)
            hist_rep = history[:, None, :].expand(-1, cand_embed.shape[1], -1)
            score_input = torch.cat([hist_rep, cand_embed], dim=-1)
            delta = self.max_logit_correction * torch.tanh(self.scorer(score_input).squeeze(-1))
            if self.variant == 'env_gated':
                delta = delta * activation[:, None]
            logits = torch.log(torch.clamp(raw_pi, min=1e-08)) + delta
            pi = torch.softmax(logits, dim=1)
            pred = (pi[:, :, None, None] * candidates).sum(dim=1)
            return (pred, pi, delta)

    def selector_loss(pred, y, pi, raw_pi, penalty, gt_mask, args, anchor_lambda, env_lambda):
        forecast = dk.forecast_loss(pred, y, args)
        anchor = torch.mean((pi - raw_pi) ** 2)
        per_sample_env = torch.sum(pi * penalty, dim=1)
        env = torch.sum(per_sample_env * gt_mask) / torch.clamp(torch.sum(gt_mask), min=1.0)
        total = forecast + float(anchor_lambda) * anchor + float(env_lambda) * env
        return (total, forecast, anchor, env)

    def infer_selector(model, X, raw_pi, candidates, cand_features, activation, batch_size, device):
        model.eval()
        preds, pis, deltas = ([], [], [])
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start:start + batch_size]).float().to(device)
                pb = torch.from_numpy(raw_pi[start:start + batch_size]).float().to(device)
                cb = torch.from_numpy(candidates[start:start + batch_size]).float().to(device)
                fb = torch.from_numpy(cand_features[start:start + batch_size]).float().to(device)
                ab = torch.from_numpy(activation[start:start + batch_size]).float().to(device)
                pred, pi, delta = model(xb, pb, cb, fb, ab)
                preds.append(pred.cpu().numpy())
                pis.append(pi.cpu().numpy())
                deltas.append(delta.cpu().numpy())
        return (np.concatenate(preds), np.concatenate(pis), np.concatenate(deltas))

    def train_one_selector(variant, Xtr, raw_tr, pi_tr, cand_tr, feat_tr, act_tr, penalty_tr, mask_tr, ytr, Xva, raw_va, pi_va, cand_va, feat_va, act_va, penalty_va, mask_va, yva, args, device, anchor_lambda, env_lambda):
        loader = DataLoader(E2Dataset(Xtr, raw_tr, pi_tr, cand_tr, feat_tr, act_tr, penalty_tr, mask_tr, ytr), batch_size=args.e2_batch_size, shuffle=True, pin_memory=device.type == 'cuda')
        model = LearnedResidualCandidateSelector(Xtr.shape[-1], feat_tr.shape[-1], args.e2_hidden_dim, args.e2_gru_layers, args.e2_dropout, args.max_logit_correction, variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.e2_lr, weight_decay=args.e2_weight_decay)
        best_score = None
        best_state = None
        best_epoch = -1
        bad = 0
        history = []
        for epoch in range(1, args.e2_epochs + 1):
            model.train()
            sums = {'loss': 0.0, 'forecast': 0.0, 'anchor': 0.0, 'env': 0.0}
            count = 0
            for batch in loader:
                xb, _, pib, cb, fb, ab, penb, maskb, yb = [x.to(device) for x in batch]
                pred, pi, _ = model(xb, pib, cb, fb, ab)
                loss, forecast, anchor, env = selector_loss(pred, yb, pi, pib, penb, maskb, args, anchor_lambda, env_lambda)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                n = len(xb)
                count += n
                sums['loss'] += float(loss.item()) * n
                sums['forecast'] += float(forecast.item()) * n
                sums['anchor'] += float(anchor.item()) * n
                sums['env'] += float(env.item()) * n
            val_pred, val_pi, _ = infer_selector(model, Xva, pi_va, cand_va, feat_va, act_va, args.e2_batch_size, device)
            val_metrics = base.compute_position_metrics(yva, val_pred)
            val_score = selection_score(val_metrics, args)
            gate_l1 = float(np.abs(val_pi - pi_va).sum(axis=1).mean())
            record = {'variant': variant, 'anchor_lambda': anchor_lambda, 'env_lambda': env_lambda, 'epoch': epoch, 'train_loss': sums['loss'] / max(count, 1), 'train_forecast': sums['forecast'] / max(count, 1), 'train_anchor': sums['anchor'] / max(count, 1), 'train_env': sums['env'] / max(count, 1), 'val_score_raw_selector': val_score, 'val_gate_l1_change': gate_l1, **{f'val_{k}': v for k, v in val_metrics.items()}}
            history.append(record)
            if best_score is None or val_score < best_score:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
            if epoch == 1 or epoch % args.log_every == 0 or bad == 0:
                print(f"[E2 {variant} a={anchor_lambda:g} env={env_lambda:g}] ep={epoch:03d} val_score={val_score:.6f} ADE={val_metrics['ade']:.6f} FDE={val_metrics['fde']:.6f} gateL1={gate_l1:.5f}")
            if bad >= args.e2_patience:
                break
        model.load_state_dict(best_state)
        val_pred, val_pi, _ = infer_selector(model, Xva, pi_va, cand_va, feat_va, act_va, args.e2_batch_size, device)
        safe_best, safe_sweep = dk.tune_safety(va_for_global_safety, global_val_for_safety, val_pred, args)
        val_safe, eff, _ = dk.apply_safety(global_val_for_safety, val_pred, safe_best['alpha'], safe_best['safety_tau'])
        val_safe_metrics = base.compute_position_metrics(yva, val_safe)
        info = {'variant': variant, 'anchor_lambda': float(anchor_lambda), 'env_lambda': float(env_lambda), 'best_epoch': int(best_epoch), 'val_raw_selector_score': float(best_score), 'val_safety_score': float(selection_score(val_safe_metrics, args)), 'safety_alpha': float(safe_best['alpha']), 'safety_tau': float(safe_best['safety_tau']), 'mean_effective_alpha_val': float(eff.mean()), **{f'val_safe_{k}': float(v) for k, v in val_safe_metrics.items()}}
        return (model, pd.DataFrame(history), info, safe_best, safe_sweep)
    va_for_global_safety = None
    global_val_for_safety = None

    def method_row(data, pred, method, voxel_map, args, extra=None):
        row = {'method': method, 'horizon': int(data.future_global.shape[1]), **base.compute_position_metrics(data.future_global[:, :, :3], pred), **e1.prediction_map_diagnostics(voxel_map, pred, args.unsafe_clearance)}
        if extra:
            row.update(extra)
        return row

    def gate_change_summary(data, voxel_map, raw_pi, new_pi, method, near_distance):
        gt = data.future_global[:, :, :3]
        gt_clearance, _ = voxel_map.query_clearance(gt)
        gt_min = gt_clearance.min(axis=1)
        l1 = np.abs(new_pi - raw_pi).sum(axis=1)
        top_changed = (new_pi.argmax(axis=1) != raw_pi.argmax(axis=1)).astype(np.float32)
        rows = []
        for subset, mask in {'all': np.ones(len(gt_min), dtype=bool), f'gt_near_obstacle_le_{near_distance:g}m': gt_min <= near_distance, f'gt_open_gt_{near_distance:g}m': gt_min > near_distance}.items():
            if mask.sum() == 0:
                continue
            rows.append({'method': method, 'subset': subset, 'num_samples': int(mask.sum()), 'mean_gate_l1_change': float(l1[mask].mean()), 'median_gate_l1_change': float(np.median(l1[mask])), 'top1_mode_changed_rate': float(top_changed[mask].mean())})
        return (rows, l1, gt_min)

    def subset_results(data, predictions, voxel_map, candidate_feat, args):
        gt = data.future_global[:, :, :3]
        gt_clearance, _ = voxel_map.query_clearance(gt)
        gt_min = gt_clearance.min(axis=1)
        candidate_gap = candidate_feat['min_clearance'].max(axis=1) - candidate_feat['min_clearance'].min(axis=1)
        masks = {'all': np.ones(len(gt), dtype=bool), f'gt_near_obstacle_le_{args.report_near_clearance:g}m': gt_min <= args.report_near_clearance, f'gt_open_gt_{args.report_near_clearance:g}m': gt_min > args.report_near_clearance, f'candidate_clearance_gap_ge_{args.report_candidate_gap:g}m': candidate_gap >= args.report_candidate_gap}
        rows = []
        for subset, mask in masks.items():
            if mask.sum() == 0:
                continue
            for method, pred in predictions.items():
                rows.append({'subset': subset, 'num_samples': int(mask.sum()), **method_row(data_subset_view(data, mask), pred[mask], method, voxel_map, args)})
        return pd.DataFrame(rows)

    class DataSubsetView:
        pass

    def data_subset_view(data, mask):
        view = DataSubsetView()
        view.future_global = data.future_global[mask]
        return view

    def _pack_standardizer(obj):
        """Convert a mean/std standardizer to a torch.save-friendly plain dictionary."""
        if obj is None:
            return None
        return {
            'mean': np.asarray(obj.mean).copy(),
            'std': np.asarray(obj.std).copy(),
        }

    def _cpu_state_dict(model):
        """Detach a model state_dict and move every tensor to CPU before serialization."""
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    def _pack_global_koopman(global_pack):
        """Serialize the fitted no-action global XY/Z Koopman predictor."""
        return {
            'norm': _pack_standardizer(global_pack['norm']),
            'fstd_xy': _pack_standardizer(global_pack['fstd_xy']),
            'fstd_z': _pack_standardizer(global_pack['fstd_z']),
            'Wxy': np.asarray(global_pack['Wxy']).copy(),
            'Wz': np.asarray(global_pack['Wz']).copy(),
            'best': global_pack.get('best'),
        }

    def _pack_switching_koopman(latent, args):
        """
        Serialize the complete label-free Switching-Koopman candidate frontend.

        The K candidate operators are represented by one global XY/Z operator plus
        K residual XY/Z operators.  No class instance is stored, so the checkpoint
        can be reconstructed without relying on pickle import paths.
        """
        return {
            'format_version': 1,
            'K': int(latent.K),
            'input_len': int(args.input_len),
            'horizon': int(args.horizon),
            'dt': float(latent.dt),
            'norm': _pack_standardizer(latent.norm),
            'fstd_xy': _pack_standardizer(latent.fstd_xy),
            'fstd_z': _pack_standardizer(latent.fstd_z),
            'gate_std': _pack_standardizer(latent.gate_std),
            'Wxy_g': np.asarray(latent.Wxy_g).copy(),
            'Wz_g': np.asarray(latent.Wz_g).copy(),
            'Rxy': [None if value is None else np.asarray(value).copy() for value in latent.Rxy],
            'Rz': [None if value is None else np.asarray(value).copy() for value in latent.Rz],
            'centroids': np.asarray(latent.centroids).copy(),
            'Wgate': np.asarray(latent.Wgate).copy(),
            'fit_config': {
                'ridge_model': float(args.ridge_model),
                'ridge_gate': float(args.ridge_gate),
                'min_cluster_samples': int(args.min_cluster_samples),
                'alt_iter': int(args.alt_iter),
                'kmeans_iter': int(args.kmeans_iter),
                'seed': int(args.seed + args.K),
            },
        }

    def fit_frozen_base_dkmoe(tr, va, te, gtr, gva, gte, Xtr, Xva, Xte, ytr, yva, args, device, out_dir):
        print(f'\n=== Training frozen D-KMoE base at K={args.K} ===')
        latent, _, alt_history = base.fit_switching_model(tr, args.K, args.ridge_model, args.ridge_gate, args, seed=args.seed + args.K)
        alt_history.to_csv(Path(out_dir) / 'latent_alternating_history.csv', index=False)

        # Save the fitted K Switching-Koopman operators immediately.  This is the
        # deployable candidate generator; real-time inference only loads these
        # matrices and performs feature construction + matrix multiplication.
        switching_path = Path(out_dir) / 'label_free_switching_koopman_frontend.pt'
        torch.save(_pack_switching_koopman(latent, args), switching_path)
        print('saved label-free Switching-Koopman frontend:', switching_path)

        ctr = base.candidates_global(latent, tr).astype(np.float32)
        cva = base.candidates_global(latent, va).astype(np.float32)
        cte = base.candidates_global(latent, te).astype(np.float32)
        best_model = None
        best_pack = None
        histories = []
        selections = []
        sweeps = []
        for anchor in parse_float_list(args.anchor_lambda_grid):
            for balance in parse_float_list(args.balance_lambda_grid):
                model, hist, info = dk.train_one(Xtr, gtr, ctr, ytr, Xva, gva, cva, yva, args, device, anchor, balance)
                histories.append(hist)
                val_pred, _ = dk.infer(model, Xva, gva, cva, args.batch_size, device)
                safe, sweep = dk.tune_safety(va, gva, val_pred, args)
                sweep['anchor_lambda'] = anchor
                sweep['balance_lambda'] = balance
                sweeps.append(sweep)
                rec = {**info, 'safe_val_score': safe['score'], 'safe_alpha': safe['alpha'], 'safe_tau': safe['safety_tau'], 'safe_ade': safe['ade'], 'safe_fde': safe['fde']}
                selections.append(rec)
                if best_pack is None or safe['score'] < best_pack['safe']['score']:
                    best_model = copy.deepcopy(model).cpu()
                    best_pack = {'info': info, 'safe': safe}
        pd.concat(histories, ignore_index=True).to_csv(Path(out_dir) / 'base_dkmoe_training_history.csv', index=False)
        pd.DataFrame(selections).to_csv(Path(out_dir) / 'base_dkmoe_selection_val.csv', index=False)
        pd.concat(sweeps, ignore_index=True).to_csv(Path(out_dir) / 'base_dkmoe_safety_sweep_val.csv', index=False)
        # Save the frozen D-KMoE separately as soon as model selection finishes.
        # The sequence standardizer is saved later in the complete E6 deployment bundle.
        dkmoe_path = Path(out_dir) / 'label_free_base_dkmoe.pt'
        torch.save({
            'format_version': 1,
            'model_state_dict': _cpu_state_dict(best_model),
            'model_config': {
                'input_dim': int(Xtr.shape[-1]),
                'K': int(args.K),
                'horizon': int(args.horizon),
                'hidden_dim': int(args.hidden_dim),
                'gru_layers': int(args.gru_layers),
                'dropout': float(args.dropout),
                'max_scale_deviation': float(args.max_scale_deviation),
            },
            'best_pack': best_pack,
            'args': vars(args),
        }, dkmoe_path)
        print('saved frozen label-free D-KMoE:', dkmoe_path)

        model = best_model.to(device)
        tr_raw, tr_pi, tr_cal = e1.infer_full(model, Xtr, gtr, ctr, args.batch_size, device)
        va_raw, va_pi, va_cal = e1.infer_full(model, Xva, gva, cva, args.batch_size, device)
        te_raw, te_pi, te_cal = e1.infer_full(model, Xte, gte, cte, args.batch_size, device)
        va_safe, _, _ = dk.apply_safety(gva, va_raw, best_pack['safe']['alpha'], best_pack['safe']['safety_tau'])
        te_safe, _, _ = dk.apply_safety(gte, te_raw, best_pack['safe']['alpha'], best_pack['safe']['safety_tau'])
        return {'latent': latent, 'model': model, 'best_pack': best_pack, 'train': {'raw': tr_raw, 'pi': tr_pi, 'calibrated': tr_cal}, 'val': {'raw': va_raw, 'pi': va_pi, 'calibrated': va_cal, 'safe': va_safe}, 'test': {'raw': te_raw, 'pi': te_pi, 'calibrated': te_cal, 'safe': te_safe}}

    def main():
        global va_for_global_safety, global_val_for_safety
        ap = argparse.ArgumentParser(description='E2 learned environment-conditioned selector over frozen D-KMoE candidates.')
        ap.add_argument('--data_root', required=True)
        ap.add_argument('--binvox_file', required=True)
        ap.add_argument('--out_dir', required=True)
        ap.add_argument('--input_len', type=int, default=20)
        ap.add_argument('--horizon', type=int, default=10)
        ap.add_argument('--stride', type=int, default=5)
        ap.add_argument('--dt', type=float, default=0.1)
        ap.add_argument('--K', type=int, default=12)
        ap.add_argument('--ridge_model', type=float, default=1.0)
        ap.add_argument('--ridge_gate', type=float, default=1.0)
        ap.add_argument('--min_cluster_samples', type=int, default=80)
        ap.add_argument('--alt_iter', type=int, default=12)
        ap.add_argument('--kmeans_iter', type=int, default=50)
        ap.add_argument('--global_ridge_xy_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--global_ridge_z_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        ap.add_argument('--hidden_dim', type=int, default=128)
        ap.add_argument('--gru_layers', type=int, default=2)
        ap.add_argument('--dropout', type=float, default=0.2)
        ap.add_argument('--max_scale_deviation', type=float, default=0.25)
        ap.add_argument('--anchor_lambda_grid', default='0,0.01,0.1')
        ap.add_argument('--balance_lambda_grid', default='0,0.001')
        ap.add_argument('--epochs', type=int, default=80)
        ap.add_argument('--patience', type=int, default=12)
        ap.add_argument('--batch_size', type=int, default=512)
        ap.add_argument('--lr', type=float, default=0.001)
        ap.add_argument('--weight_decay', type=float, default=0.0001)
        ap.add_argument('--grad_clip', type=float, default=1.0)
        ap.add_argument('--log_every', type=int, default=10)
        ap.add_argument('--train_ade_weight', type=float, default=0.5)
        ap.add_argument('--train_fde_weight', type=float, default=0.25)
        ap.add_argument('--selection_metric', default='combo', choices=['pos_rmse', 'ade', 'fde', 'combo'])
        ap.add_argument('--combo_ade_weight', type=float, default=0.5)
        ap.add_argument('--combo_fde_weight', type=float, default=0.25)
        ap.add_argument('--alpha_grid', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        ap.add_argument('--safety_quantile_grid', default='0.4,0.6,0.8,0.95,1.0')
        ap.add_argument('--map_expected_xyz', nargs=3, type=int, required=True)
        ap.add_argument('--map_center_xyz', nargs=3, type=float, required=True)
        ap.add_argument('--map_resolution', type=float, default=1.0)
        ap.add_argument('--map_axis_mode', default='swap_yz', choices=['swap_yz', 'current_xyz'])
        p.add_argument('--map_axis_mode', default='swap_yz', choices=['swap_yz', 'current_xyz'])
        ap.add_argument('--unsafe_clearance', type=float, default=1.0)
        ap.add_argument('--report_near_clearance', type=float, default=5.0)
        ap.add_argument('--report_candidate_gap', type=float, default=2.0)
        ap.add_argument('--env_terminal_weight', type=float, default=2.0)
        ap.add_argument('--env_safe_distance_grid', default='0,0.5,1,2,3,5')
        ap.add_argument('--env_beta_grid', default='0,0.03,0.1,0.3,1,3,10')
        ap.add_argument('--env_alpha_grid', default='0,0.25,0.5,0.75,1')
        ap.add_argument('--variants', default='noenv,env,env_gated')
        ap.add_argument('--e2_hidden_dim', type=int, default=96)
        ap.add_argument('--e2_gru_layers', type=int, default=1)
        ap.add_argument('--e2_dropout', type=float, default=0.15)
        ap.add_argument('--e2_epochs', type=int, default=100)
        ap.add_argument('--e2_patience', type=int, default=15)
        ap.add_argument('--e2_batch_size', type=int, default=512)
        ap.add_argument('--e2_lr', type=float, default=0.001)
        ap.add_argument('--e2_weight_decay', type=float, default=0.0001)
        ap.add_argument('--max_logit_correction', type=float, default=3.0)
        ap.add_argument('--e2_anchor_lambda_grid', default='0,0.01,0.1')
        ap.add_argument('--e2_env_lambda_grid', default='0,0.001,0.01')
        ap.add_argument('--e2_safe_distance', type=float, default=3.0)
        ap.add_argument('--gt_env_ignore_clearance', type=float, default=1.0)
        ap.add_argument('--env_gate_distance', type=float, default=5.0)
        ap.add_argument('--env_gate_temperature', type=float, default=1.0)
        ap.add_argument('--device', default='auto')
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--fast', action='store_true')
        args = ap.parse_args()
        set_seed(args.seed)
        variants = parse_str_list(args.variants)
        allowed = {'noenv', 'env', 'env_gated'}
        if any((v not in allowed for v in variants)):
            raise ValueError(f'--variants 只能从 {sorted(allowed)} 中选择。')
        if args.fast:
            args.alt_iter = min(args.alt_iter, 8)
            args.epochs = min(args.epochs, 50)
            args.patience = min(args.patience, 10)
            args.anchor_lambda_grid = '0,0.01,0.1'
            args.balance_lambda_grid = '0,0.001'
            args.e2_epochs = min(args.e2_epochs, 60)
            args.e2_patience = min(args.e2_patience, 10)
            args.e2_anchor_lambda_grid = '0,0.01'
            args.e2_env_lambda_grid = '0,0.001'
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        print('device =', device)
        print('K =', args.K)
        print('variants =', variants)
        print('IMPORTANT: no action labels are used.')
        print('IMPORTANT: E2 freezes Koopman candidates and base D-KMoE, and trains only residual selector.')
        print('IMPORTANT: env vs noenv is the primary test of added environment predictive value.')
        print('IMPORTANT: voxel decoding and NED-Z mapping follow the previously verified convention.')
        out_dir = Path(args.out_dir)
        ensure_dir(out_dir)
        voxel_map = e1.AirSimNEDClearanceMap.from_binvox(args.binvox_file, args.map_expected_xyz, args.map_center_xyz, args.map_resolution, axis_mode=getattr(args, 'map_axis_mode', 'swap_yz'))
        (out_dir / 'map_metadata_validated_ned.json').write_text(json.dumps(voxel_map.metadata(), ensure_ascii=False, indent=2), encoding='utf-8')
        root = Path(args.data_root)
        tr = bk.make_split_data(bk.load_split_chains(root, 'train'), args.input_len, args.horizon, args.stride)
        va = bk.make_split_data(bk.load_split_chains(root, 'val'), args.input_len, args.horizon, args.stride)
        te = bk.make_split_data(bk.load_split_chains(root, 'test'), args.input_len, args.horizon, args.stride)
        print(f'samples: train={len(tr.hist_global)}, val={len(va.hist_global)}, test={len(te.hist_global)}')
        global_pack = base.fit_no_action_global(tr, va, args)
        gtr = base.predict_no_action_global(global_pack, tr, args).astype(np.float32)
        gva = base.predict_no_action_global(global_pack, va, args).astype(np.float32)
        gte = base.predict_no_action_global(global_pack, te, args).astype(np.float32)
        Xtr0 = prior.make_sequence_features(tr, args.dt)
        Xva0 = prior.make_sequence_features(va, args.dt)
        Xte0 = prior.make_sequence_features(te, args.dt)
        seq_std = prior.SeqStandardizer.fit(Xtr0)
        Xtr = seq_std.transform(Xtr0)
        Xva = seq_std.transform(Xva0)
        Xte = seq_std.transform(Xte0)
        ytr = tr.future_global[:, :, :3].astype(np.float32)
        yva = va.future_global[:, :, :3].astype(np.float32)
        frozen = fit_frozen_base_dkmoe(tr, va, te, gtr, gva, gte, Xtr, Xva, Xte, ytr, yva, args, device, out_dir)
        va_feat = e1.candidate_environment_features(voxel_map, frozen['val']['calibrated'], args.env_terminal_weight)
        te_feat = e1.candidate_environment_features(voxel_map, frozen['test']['calibrated'], args.env_terminal_weight)
        e1_best, e1_sweep = e1.tune_environment_reweight(va, frozen['val']['raw'], frozen['val']['pi'], frozen['val']['calibrated'], va_feat['clearance'], args)
        e1_sweep.to_csv(out_dir / 'e1_rule_reweight_sweep_val_same_base.csv', index=False)
        e1_val_pred, _, _, _ = e1.apply_environment_reweight(frozen['val']['raw'], frozen['val']['pi'], frozen['val']['calibrated'], va_feat['clearance'], e1_best['env_safe_distance'], e1_best['env_beta'], e1_best['env_alpha'], args.env_terminal_weight)
        e1_safety, e1_safety_sweep = dk.tune_safety(va, gva, e1_val_pred, args)
        e1_safety_sweep.to_csv(out_dir / 'e1_rule_reweight_plus_safety_sweep_val_same_base.csv', index=False)
        dyn_tr = build_dynamic_candidate_features(tr, gtr, frozen['train']['raw'], frozen['train']['calibrated'], frozen['train']['pi'])
        dyn_va = build_dynamic_candidate_features(va, gva, frozen['val']['raw'], frozen['val']['calibrated'], frozen['val']['pi'])
        dyn_te = build_dynamic_candidate_features(te, gte, frozen['test']['raw'], frozen['test']['calibrated'], frozen['test']['pi'])
        env_tr, env_feat_tr, raw_min_tr = build_environment_candidate_features(voxel_map, frozen['train']['raw'], frozen['train']['calibrated'], args.env_terminal_weight)
        env_va, env_feat_va, raw_min_va = build_environment_candidate_features(voxel_map, frozen['val']['raw'], frozen['val']['calibrated'], args.env_terminal_weight)
        env_te, env_feat_te, raw_min_te = build_environment_candidate_features(voxel_map, frozen['test']['raw'], frozen['test']['calibrated'], args.env_terminal_weight)
        dyn_std = FlatStandardizer.fit(dyn_tr)
        env_std = FlatStandardizer.fit(env_tr)
        dyn_tr, dyn_va, dyn_te = (dyn_std.transform(dyn_tr), dyn_std.transform(dyn_va), dyn_std.transform(dyn_te))
        env_tr, env_va, env_te = (env_std.transform(env_tr), env_std.transform(env_va), env_std.transform(env_te))
        act_tr = environment_activation(raw_min_tr, args.env_gate_distance, args.env_gate_temperature)
        act_va = environment_activation(raw_min_va, args.env_gate_distance, args.env_gate_temperature)
        act_te = environment_activation(raw_min_te, args.env_gate_distance, args.env_gate_temperature)
        penalty_tr = make_environment_penalty(env_feat_tr, args.e2_safe_distance, args.env_terminal_weight)
        penalty_va = make_environment_penalty(env_feat_va, args.e2_safe_distance, args.env_terminal_weight)
        penalty_te = make_environment_penalty(env_feat_te, args.e2_safe_distance, args.env_terminal_weight)
        mask_tr, _ = gt_environment_mask(voxel_map, ytr, args.gt_env_ignore_clearance)
        mask_va, _ = gt_environment_mask(voxel_map, yva, args.gt_env_ignore_clearance)
        mask_te, _ = gt_environment_mask(voxel_map, te.future_global[:, :, :3].astype(np.float32), args.gt_env_ignore_clearance)
        va_for_global_safety = va
        global_val_for_safety = gva
        globals()['va_for_global_safety'] = va_for_global_safety
        globals()['global_val_for_safety'] = global_val_for_safety
        variant_feature_arrays = {'noenv': (dyn_tr, dyn_va, dyn_te), 'env': (np.concatenate([dyn_tr, env_tr], axis=2), np.concatenate([dyn_va, env_va], axis=2), np.concatenate([dyn_te, env_te], axis=2)), 'env_gated': (np.concatenate([dyn_tr, env_tr], axis=2), np.concatenate([dyn_va, env_va], axis=2), np.concatenate([dyn_te, env_te], axis=2))}
        variant_best = {}
        all_selection_rows = []
        all_histories = []
        all_safety_sweeps = []
        for variant in variants:
            print(f'\n=== Training E2 variant: {variant} ===')
            ftr, fva, fte = variant_feature_arrays[variant]
            env_lambdas = [0.0] if variant == 'noenv' else parse_float_list(args.e2_env_lambda_grid)
            best = None
            for anchor_lambda in parse_float_list(args.e2_anchor_lambda_grid):
                for env_lambda in env_lambdas:
                    set_seed(args.seed + int(anchor_lambda * 1000) + int(env_lambda * 100000) + len(variant))
                    model, history, info, safety, safety_sweep = train_one_selector(variant, Xtr, frozen['train']['raw'], frozen['train']['pi'], frozen['train']['calibrated'], ftr, act_tr, penalty_tr, mask_tr, ytr, Xva, frozen['val']['raw'], frozen['val']['pi'], frozen['val']['calibrated'], fva, act_va, penalty_va, mask_va, yva, args, device, anchor_lambda, env_lambda)
                    history['variant'] = variant
                    all_histories.append(history)
                    safety_sweep['variant'] = variant
                    safety_sweep['anchor_lambda'] = anchor_lambda
                    safety_sweep['env_lambda'] = env_lambda
                    all_safety_sweeps.append(safety_sweep)
                    all_selection_rows.append(info)
                    if best is None or info['val_safety_score'] < best['info']['val_safety_score']:
                        best = {'model': copy.deepcopy(model).cpu(), 'info': info, 'safety': safety, 'feature_dim': int(ftr.shape[-1])}
            variant_best[variant] = best
        pd.DataFrame(all_selection_rows).to_csv(out_dir / 'e2_variant_selection_val.csv', index=False)
        pd.concat(all_histories, ignore_index=True).to_csv(out_dir / 'e2_training_history.csv', index=False)
        pd.concat(all_safety_sweeps, ignore_index=True).to_csv(out_dir / 'e2_global_safety_sweep_val.csv', index=False)
        e1_test_pred, _, _, _ = e1.apply_environment_reweight(frozen['test']['raw'], frozen['test']['pi'], frozen['test']['calibrated'], te_feat['clearance'], e1_best['env_safe_distance'], e1_best['env_beta'], e1_best['env_alpha'], args.env_terminal_weight)
        e1_test_safe, _, _ = dk.apply_safety(gte, e1_test_pred, e1_safety['alpha'], e1_safety['safety_tau'])
        predictions = {'v5_no_action_global': gte, 'differentiable_koopman_moe_raw': frozen['test']['raw'], 'differentiable_koopman_moe_safety': frozen['test']['safe'], 'e1_env_reweight_dkmoe_same_base': e1_test_pred, 'e1_env_reweight_plus_global_safety_same_base': e1_test_safe}
        rows = [method_row(te, gte, 'v5_no_action_global', voxel_map, args), method_row(te, frozen['test']['raw'], 'differentiable_koopman_moe_raw', voxel_map, args), method_row(te, frozen['test']['safe'], 'differentiable_koopman_moe_safety', voxel_map, args), method_row(te, e1_test_pred, 'e1_env_reweight_dkmoe_same_base', voxel_map, args, e1_best), method_row(te, e1_test_safe, 'e1_env_reweight_plus_global_safety_same_base', voxel_map, args, {**e1_best, 'safety_alpha': e1_safety['alpha'], 'safety_tau': e1_safety['safety_tau']})]
        gate_rows = []
        gate_detail = pd.DataFrame({'sample_idx': np.arange(len(te.hist_global)), 'base_raw_min_clearance': raw_min_te})
        for variant in variants:
            fte = variant_feature_arrays[variant][2]
            best = variant_best[variant]
            model = best['model'].to(device)
            pred_raw, pi_new, _ = infer_selector(model, Xte, frozen['test']['pi'], frozen['test']['calibrated'], fte, act_te, args.e2_batch_size, device)
            safe = best['safety']
            pred_safe, eff, _ = dk.apply_safety(gte, pred_raw, safe['alpha'], safe['safety_tau'])
            name_raw = f'e2_{variant}_selector_raw'
            name_safe = f'e2_{variant}_selector_plus_global_safety'
            predictions[name_raw] = pred_raw
            predictions[name_safe] = pred_safe
            rows.append(method_row(te, pred_raw, name_raw, voxel_map, args, best['info']))
            rows.append(method_row(te, pred_safe, name_safe, voxel_map, args, {**best['info'], 'test_mean_effective_global_safety_alpha': float(eff.mean())}))
            summary_rows, l1, gt_min = gate_change_summary(te, voxel_map, frozen['test']['pi'], pi_new, name_raw, args.report_near_clearance)
            gate_rows.extend(summary_rows)
            gate_detail[f'{variant}_gate_l1_change'] = l1
            gate_detail[f'{variant}_top_mode_changed'] = (pi_new.argmax(axis=1) != frozen['test']['pi'].argmax(axis=1)).astype(int)
            gate_detail['gt_min_clearance'] = gt_min
        compare = pd.DataFrame(rows)
        compare.to_csv(out_dir / 'compare_e2_learned_environment_selector_test.csv', index=False)
        pd.DataFrame(gate_rows).to_csv(out_dir / 'e2_gate_change_summary_test.csv', index=False)
        gate_detail.to_csv(out_dir / 'e2_gate_change_details_test.csv', index=False)
        subset = subset_results(te, predictions, voxel_map, env_feat_te, args)
        subset.to_csv(out_dir / 'e2_subset_metrics_test.csv', index=False)
        report = {'experiment': 'E2 learned environment-conditioned residual selector over frozen D-KMoE candidates', 'K': args.K, 'action_labels_used': False, 'key_comparison': 'Compare e2_env_selector / e2_env_gated_selector against e2_noenv_selector. Only this isolates added predictive value from environment features.', 'map_metadata': voxel_map.metadata(), 'selected_validation_hyperparameters': {v: variant_best[v]['info'] for v in variants}, 'test_results': compare.to_dict(orient='records'), 'gate_change_summary_test': pd.DataFrame(gate_rows).to_dict(orient='records'), 'args': vars(args)}
        (out_dir / 'report_e2_learned_environment_selector.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        for variant in variants:
            torch.save({'variant': variant, 'K': args.K, 'model_state_dict': variant_best[variant]['model'].state_dict(), 'selection_info': variant_best[variant]['info'], 'global_safety': variant_best[variant]['safety'], 'seq_feature_mean': seq_std.mean.tolist(), 'seq_feature_std': seq_std.std.tolist(), 'dynamic_feature_mean': dyn_std.mean.tolist(), 'dynamic_feature_std': dyn_std.std.tolist(), 'environment_feature_mean': env_std.mean.tolist(), 'environment_feature_std': env_std.std.tolist(), 'map_metadata': voxel_map.metadata(), 'args': vars(args)}, out_dir / f'e2_{variant}_selector.pt')
        print('\n=== E2 Learned Environment-Conditioned Selector finished ===')
        print(compare.to_string(index=False))
        print('\nEnvironment contribution should be judged by:')
        print('  e2_env_selector / e2_env_gated_selector  versus  e2_noenv_selector')
        print('\nKey files:')
        for name in ['compare_e2_learned_environment_selector_test.csv', 'e2_variant_selection_val.csv', 'e2_subset_metrics_test.csv', 'e2_gate_change_summary_test.csv', 'report_e2_learned_environment_selector.json']:
            print(' ', out_dir / name)

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

e2_learned_environment_conditioned_koopman_selector = _init_e2_learned_environment_conditioned_koopman_selector()


# ========================================================================================
# SECTION: E4 MAP USAGE AB EXPERIMENT V2
# ========================================================================================
def _init_e4_map_usage_ab_experiment_v2():
    """
    e4_map_usage_ab_experiment.py

    E4: A / B / A+B Environment-Usage Experiment for Koopman UAV Trajectory Prediction
    ====================================================================================

    本脚本同时测试三种“改变地图使用方式”的环境增强模型：

    A. Candidate-Centric Corridor Token Selector
       - 保留冻结的 D-KMoE / Koopman candidates；
       - 不再使用当前无人机周围规则立方体网格作为地图 token；
       - 对每条 candidate 沿途构建三维管廊 token，只用于修正候选混合权重。

    B. Map-Conditioned Motion-Query Candidate Generator
       - 使用当前 UAV 局部坐标下的可行终点/自由空间 motion-query tokens；
       - 在冻结 Koopman candidate 周围生成有界轨迹残差，即地图开始影响候选生成；
       - 同时学习候选权重。

    AB. Corridor Token + Motion-Query Joint Model
       - candidate 沿途管廊 token 提供路径相关建筑几何；
       - 自由空间 motion-query tokens 提供局部可飞方向；
       - 二者联合用于 candidate residual generation 与 candidate selection。

    严格实验规则：
        - 不使用动作/行为标签；
        - 复用并冻结 D-KMoE 的 latent Koopman candidates 与基础 gate；
        - train/val 的未来轨迹仅作为正常预测监督与 candidate-ranking 监督；
        - 每种变体的参数仅由 validation 选择；
        - test 对每种预先定义的变体报告一次；
        - 对 A/B/AB 均做 CorrectMap / ShuffledMap / ZeroSemanticMap 反事实测试。

    对 B 的说明：
        本实验是“map-conditioned candidate generation around frozen Koopman anchors”
        的可运行验证版：地图生成对 Koopman candidates 的有界残差修正；
        它不是重新学习一个完全地图条件化的 Koopman 算子。如果 B/AB 成功，
        再将该 residual generator 升级为真正的 map-conditioned Koopman motion query。

    已验证且必须保持的 AirSim 地图约定：
        idx = x + nx * (z + nz * y)
        occ_xyz = flat.reshape((ny, nz, nx)).transpose(2, 0, 1)
        z_ned(k) = center_z - (k - nz//2) * resolution
        所有环境特征均通过严格三维 NED 查询产生。
    """
    import argparse
    import copy
    import json
    import math
    import random
    from pathlib import Path
    from typing import Dict, Tuple
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    e1 = env_reweight_differentiable_koopman_moe_e1
    e2 = e2_learned_environment_conditioned_koopman_selector
    dk = e1.dk
    base = e1.base
    prior = e1.prior
    bk = e1.bk

    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def parse_float_list(text):
        return [float(x.strip()) for x in str(text).split(',') if x.strip()]

    def parse_int_list(text):
        return [int(x.strip()) for x in str(text).split(',') if x.strip()]

    def parse_str_list(text):
        return [x.strip().upper() for x in str(text).split(',') if x.strip()]

    def parse_range(text):
        values = parse_float_list(text)
        if len(values) != 3:
            raise ValueError(f'range 应采用 start,end,step 格式，当前为 {text}')
        start, end, step = values
        if step <= 0 or end < start:
            raise ValueError(f'range 不合法：{text}')
        arr = np.arange(start, end + step * 0.25, step, dtype=np.float32)
        return arr[arr <= end + 1e-05]

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def score_metrics(metrics, args):
        return e1.metric_score(metrics, args)

    class LastDimStandardizer:

        def __init__(self, mean, std):
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)

        @classmethod
        def fit(cls, x, eps=1e-06):
            arr = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
            return cls(arr.mean(axis=0), np.maximum(arr.std(axis=0), eps))

        def transform(self, x):
            return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    class DataView:
        pass

    def subset_view(data, mask):
        out = DataView()
        out.future_global = data.future_global[mask]
        return out

    def local_basis_from_history(data):
        """
        局部坐标：
            +forward = 水平当前运动方向
            +left    = 左侧
            +up      = AirSim NED 中的真实向上方向 [0, 0, -1]
        """
        hist = np.asarray(data.hist_global, dtype=np.float32)
        pos = hist[:, :, :3]
        direction_xy = hist[:, -1, 3:5].copy()
        norm = np.linalg.norm(direction_xy, axis=1)
        slow = norm < 0.0001
        history_disp = pos[:, -1, :2] - pos[:, 0, :2]
        direction_xy[slow] = history_disp[slow]
        norm = np.linalg.norm(direction_xy, axis=1)
        invalid = norm < 0.0001
        direction_xy[invalid] = np.array([1.0, 0.0], dtype=np.float32)
        norm = np.linalg.norm(direction_xy, axis=1, keepdims=True)
        unit = direction_xy / np.maximum(norm, 1e-06)
        forward = np.concatenate([unit, np.zeros((len(unit), 1), dtype=np.float32)], axis=1)
        left = np.stack([-unit[:, 1], unit[:, 0], np.zeros(len(unit), dtype=np.float32)], axis=1)
        up = np.repeat(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), len(unit), axis=0)
        valid = (~invalid).astype(np.float32)
        origin = pos[:, -1, :].astype(np.float32)
        return (origin, forward, left, up, valid)

    def world_vector_to_local(vectors, forward, left, up):
        """
        vectors shape: [N, ..., 3]
        return local [forward, left, up].
        """
        extra = (None,) * (vectors.ndim - 2)
        f = forward[(slice(None),) + extra + (slice(None),)]
        l = left[(slice(None),) + extra + (slice(None),)]
        u = up[(slice(None),) + extra + (slice(None),)]
        return np.stack([np.sum(vectors * f, axis=-1), np.sum(vectors * l, axis=-1), np.sum(vectors * u, axis=-1)], axis=-1).astype(np.float32)

    def local_vector_to_world(local, forward, left, up):
        """
        local shape: [B, K, H, 3] torch tensor;
        basis tensors: [B, 3].
        """
        return local[..., 0:1] * forward[:, None, None, :] + local[..., 1:2] * left[:, None, None, :] + local[..., 2:3] * up[:, None, None, :]

    def fourier_xyz(xyz, scales=(0.1, 0.25, 0.5)):
        xyz = np.asarray(xyz, dtype=np.float32)
        pieces = [xyz / 30.0]
        for scale in scales:
            pieces.append(np.sin(xyz * scale))
            pieces.append(np.cos(xyz * scale))
        return np.concatenate(pieces, axis=-1).astype(np.float32)

    def build_candidate_queries(data, raw_pred, candidates, raw_pi):
        n, k, h, _ = candidates.shape
        origin, forward, left, up, heading_valid = local_basis_from_history(data)
        cand_local = world_vector_to_local(candidates - origin[:, None, None, :], forward, left, up)
        raw_local = world_vector_to_local(raw_pred - origin[:, None, :], forward, left, up)
        inc_world = np.diff(candidates, axis=2, prepend=candidates[:, :, 0:1, :])
        inc_local = world_vector_to_local(inc_world, forward, left, up)
        diff_raw = cand_local - raw_local[:, None, :, :]
        t = np.linspace(1.0 / h, 1.0, h, dtype=np.float32)[None, None, :, None]
        t = np.broadcast_to(t, (n, k, h, 1)).copy()
        pi = np.broadcast_to(raw_pi[:, :, None, None], (n, k, h, 1)).astype(np.float32)
        log_pi = np.log(np.maximum(pi, 1e-08))
        hv = np.broadcast_to(heading_valid[:, None, None, None], (n, k, h, 1)).copy()
        query = np.concatenate([fourier_xyz(cand_local), inc_local / 10.0, diff_raw / 10.0, t, pi, log_pi, hv], axis=-1).astype(np.float32)
        summary = np.concatenate([cand_local[:, :, -1, :] / 30.0, cand_local.mean(axis=2) / 30.0, diff_raw[:, :, -1, :] / 20.0, np.linalg.norm(diff_raw, axis=-1).mean(axis=2, keepdims=True) / 20.0, raw_pi[:, :, None], np.log(np.maximum(raw_pi[:, :, None], 1e-08))], axis=-1).astype(np.float32)
        basis = np.stack([forward, left, up], axis=1).astype(np.float32)
        return (query, summary, basis)

    def _map_semantics(voxel_map, world_points, args, query_gradient=False):
        clearance, valid = voxel_map.query_clearance(world_points)
        occupied, _ = voxel_map.query_occupied_nearest(world_points)
        parts = [np.clip(clearance[..., None], 0.0, args.map_clearance_clip) / args.map_clearance_clip, np.exp(-np.clip(clearance[..., None], 0.0, args.map_clearance_clip) / args.boundary_scale), (clearance < 1.0)[..., None].astype(np.float32), (clearance < 3.0)[..., None].astype(np.float32), (clearance < 5.0)[..., None].astype(np.float32), occupied[..., None].astype(np.float32), valid[..., None].astype(np.float32)]
        if query_gradient:
            raise RuntimeError('内部函数不直接计算梯度，请在专用 builder 中提供 basis。')
        return (np.concatenate(parts, axis=-1).astype(np.float32), clearance, valid)

    def build_corridor_tokens(data, candidates, voxel_map, args):
        """
        A / AB 使用的 candidate-specific corridor tokens。
        每条 candidate 沿其未来路径的若干时间锚点提取横截面环境 token，
        再根据边界相关性选出固定数量 token。
        输出：
            tokens [N,K,M,F]
            geometry_dim
            semantic_dim
        """
        n, k, h, _ = candidates.shape
        anchor_indices = [min(max(i, 0), h - 1) for i in parse_int_list(args.corridor_anchor_indices)]
        anchor_indices = sorted(set(anchor_indices))
        lateral = parse_float_list(args.corridor_lateral_offsets)
        vertical = parse_float_list(args.corridor_up_offsets)
        if not anchor_indices or not lateral or (not vertical):
            raise ValueError('corridor anchors / offsets 不能为空。')
        origin, default_forward, _, up_global, heading_valid = local_basis_from_history(data)
        pool_count = len(anchor_indices) * len(lateral) * len(vertical)
        if int(args.corridor_token_topk) > 0:
            raise ValueError('当前正式反事实实验要求 --corridor_token_topk 设为 0：若按真实 clearance 选择 top-k token，即使将 semantic 清零，被选中的几何位置仍泄露地图信息，ZeroSemanticMap 对照将不再严格。')
        keep = pool_count
        geometry_dim = fourier_xyz(np.zeros((1, 3), dtype=np.float32)).shape[-1] + 3
        semantic_dim = 7
        out = np.empty((n, k, keep, geometry_dim + semantic_dim), dtype=np.float32)
        chunk = int(args.map_query_chunk)
        for mode in range(k):
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                b = end - start
                cand = candidates[start:end, mode]
                records = []
                local_geo = []
                for ai, t_idx in enumerate(anchor_indices):
                    center = cand[:, t_idx, :]
                    t0 = max(0, t_idx - 1)
                    t1 = min(h - 1, t_idx + 1)
                    tangent = cand[:, t1, :2] - cand[:, t0, :2]
                    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
                    invalid = tangent_norm[:, 0] < 0.0001
                    tangent[invalid] = default_forward[start:end][invalid, :2]
                    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
                    tangent = tangent / np.maximum(tangent_norm, 1e-06)
                    cand_forward = np.concatenate([tangent, np.zeros((b, 1), dtype=np.float32)], axis=1)
                    cand_left = np.stack([-tangent[:, 1], tangent[:, 0], np.zeros(b, dtype=np.float32)], axis=1)
                    cand_up = up_global[start:end]
                    for lat in lateral:
                        for vert in vertical:
                            world = center + float(lat) * cand_left + float(vert) * cand_up
                            records.append(world[:, None, :])
                            geom = np.array([float(t_idx + 1) / h, float(lat) / 15.0, float(vert) / 10.0], dtype=np.float32)
                            local_coord = np.repeat(np.array([[float(t_idx + 1) / h * 30.0, float(lat), float(vert)]], dtype=np.float32), b, axis=0)
                            g = np.concatenate([fourier_xyz(local_coord), np.repeat(geom[None, :], b, axis=0)], axis=1)
                            local_geo.append(g[:, None, :])
                world_pool = np.concatenate(records, axis=1)
                geo_pool = np.concatenate(local_geo, axis=1)
                sem_pool, _, _ = _map_semantics(voxel_map, world_pool, args)
                out[start:end, mode] = np.concatenate([geo_pool, sem_pool], axis=2)
        return (out, geometry_dim, semantic_dim)

    def build_motion_query_tokens(data, voxel_map, args):
        """
        B / AB 使用的 map-conditioned free-space motion-query tokens。
        token 是局部可达终点及原点到终点射线的三维净空描述。
        """
        forward_offsets = parse_float_list(args.motion_forward_offsets)
        left_offsets = parse_float_list(args.motion_left_offsets)
        up_offsets = parse_float_list(args.motion_up_offsets)
        ray_fractions = parse_float_list(args.motion_ray_fractions)
        endpoints = np.array([[f, l, u] for f in forward_offsets for l in left_offsets for u in up_offsets], dtype=np.float32)
        if len(endpoints) == 0 or len(ray_fractions) == 0:
            raise ValueError('motion query offsets / ray fractions 不能为空。')
        origin, forward, left, up, heading_valid = local_basis_from_history(data)
        n, m = (len(origin), len(endpoints))
        geom = np.concatenate([fourier_xyz(endpoints), np.linalg.norm(endpoints, axis=1, keepdims=True) / 30.0], axis=1).astype(np.float32)
        geom = np.repeat(geom[None, :, :], n, axis=0)
        geometry_dim = geom.shape[-1]
        semantics = []
        chunk = int(args.map_query_chunk)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            b = end - start
            ep = origin[start:end, None, :] + endpoints[None, :, 0:1] * forward[start:end, None, :] + endpoints[None, :, 1:2] * left[start:end, None, :] + endpoints[None, :, 2:3] * up[start:end, None, :]
            ray_world = []
            for frac in ray_fractions:
                ray_world.append(origin[start:end, None, :] + float(frac) * (ep - origin[start:end, None, :]))
            ray_world = np.stack(ray_world, axis=2)
            clear, valid = voxel_map.query_clearance(ray_world)
            occ, _ = voxel_map.query_occupied_nearest(ray_world)
            min_clear = clear.min(axis=2)
            mean_clear = clear.mean(axis=2)
            term_clear = clear[:, :, -1]
            valid_ratio = valid.astype(np.float32).mean(axis=2)
            occ_ratio = occ.astype(np.float32).mean(axis=2)
            sem = np.stack([np.clip(min_clear, 0.0, args.map_clearance_clip) / args.map_clearance_clip, np.clip(mean_clear, 0.0, args.map_clearance_clip) / args.map_clearance_clip, np.clip(term_clear, 0.0, args.map_clearance_clip) / args.map_clearance_clip, np.exp(-np.clip(min_clear, 0.0, args.map_clearance_clip) / args.boundary_scale), (min_clear < 1.0).astype(np.float32), (min_clear < 3.0).astype(np.float32), (min_clear < 5.0).astype(np.float32), occ_ratio, valid_ratio, np.repeat(heading_valid[start:end, None], m, axis=1)], axis=2).astype(np.float32)
            semantics.append(sem)
        semantics = np.concatenate(semantics, axis=0)
        return (np.concatenate([geom, semantics], axis=2).astype(np.float32), geometry_dim, semantics.shape[-1])

    def zero_semantic(raw_tokens, geometry_dim, std):
        x = np.asarray(raw_tokens, dtype=np.float32).copy()
        x[..., geometry_dim:] = 0.0
        return std.transform(x)

    class E4Dataset(Dataset):

        def __init__(self, X, raw_pi, candidates, query, dyn, basis, corridor, intention, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.raw_pi = torch.from_numpy(raw_pi.astype(np.float32))
            self.candidates = torch.from_numpy(candidates.astype(np.float32))
            self.query = torch.from_numpy(query.astype(np.float32))
            self.dyn = torch.from_numpy(dyn.astype(np.float32))
            self.basis = torch.from_numpy(basis.astype(np.float32))
            self.corridor = torch.from_numpy(corridor.astype(np.float32))
            self.intention = torch.from_numpy(intention.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return (self.X[i], self.raw_pi[i], self.candidates[i], self.query[i], self.dyn[i], self.basis[i], self.corridor[i], self.intention[i], self.y[i])

    class E4ABModel(nn.Module):
        """
        A  : candidate-centric corridor attention -> gate correction only.
        B  : free-space map motion-query attention -> bounded candidate residual + gate.
        AB : corridor + motion-query dual context -> bounded candidate residual + gate.
        """

        def __init__(self, variant, seq_dim, query_dim, dyn_dim, corridor_dim, intention_dim, args):
            super().__init__()
            self.variant = variant
            self.d_model = int(args.d_model)
            self.max_logit_correction = float(args.max_logit_correction)
            self.max_candidate_residual = float(args.max_candidate_residual)
            self.history_gru = nn.GRU(seq_dim, args.e4_hidden_dim, batch_first=True, bidirectional=True)
            self.query_encoder = nn.Sequential(nn.Linear(query_dim, args.d_model), nn.GELU(), nn.LayerNorm(args.d_model))
            self.dyn_encoder = nn.Sequential(nn.Linear(dyn_dim, args.d_model), nn.GELU(), nn.LayerNorm(args.d_model))
            self.corridor_encoder = nn.Sequential(nn.Linear(corridor_dim, args.d_model), nn.GELU(), nn.LayerNorm(args.d_model))
            self.intention_encoder = nn.Sequential(nn.Linear(intention_dim, args.d_model), nn.GELU(), nn.LayerNorm(args.d_model))
            self.corridor_attn = nn.MultiheadAttention(args.d_model, args.attn_heads, dropout=args.e4_dropout, batch_first=True)
            self.intention_attn = nn.MultiheadAttention(args.d_model, args.attn_heads, dropout=args.e4_dropout, batch_first=True)
            self.norm_corridor = nn.LayerNorm(args.d_model)
            self.norm_intention = nn.LayerNorm(args.d_model)
            self.dropout = nn.Dropout(args.e4_dropout)
            context_count = 1 if variant in {'A', 'B'} else 2
            scorer_input = 2 * args.e4_hidden_dim + args.d_model * (2 + context_count)
            self.scorer = nn.Sequential(nn.Linear(scorer_input, args.e4_hidden_dim), nn.GELU(), nn.Dropout(args.e4_dropout), nn.Linear(args.e4_hidden_dim, 1))
            if variant in {'B', 'AB'}:
                residual_context_count = 1 if variant == 'B' else 2
                residual_input = args.d_model * (1 + residual_context_count) + 2 * args.e4_hidden_dim
                self.residual_head = nn.Sequential(nn.Linear(residual_input, args.e4_hidden_dim), nn.GELU(), nn.Dropout(args.e4_dropout), nn.Linear(args.e4_hidden_dim, 3))
            else:
                self.residual_head = None

        def forward(self, X, raw_pi, candidates, query, dyn, basis, corridor, intention):
            b, k, h, _ = query.shape
            _, state = self.history_gru(X)
            hist = torch.cat([state[-2], state[-1]], dim=-1)
            hist_mode = hist[:, None, :].expand(-1, k, -1)
            hist_step = hist[:, None, None, :].expand(-1, k, h, -1)
            q = self.query_encoder(query)
            q_pool = q.mean(dim=2)
            dyn_emb = self.dyn_encoder(dyn)
            contexts = []
            step_contexts = []
            if self.variant in {'A', 'AB'}:
                cq = q.reshape(b * k, h, -1)
                cm = self.corridor_encoder(corridor).reshape(b * k, corridor.shape[2], -1)
                ca, _ = self.corridor_attn(cq, cm, cm, need_weights=False)
                ca = self.norm_corridor(cq + self.dropout(ca)).reshape(b, k, h, -1)
                step_contexts.append(ca)
                contexts.append(ca.mean(dim=2))
            if self.variant in {'B', 'AB'}:
                iq = q.reshape(b, k * h, -1)
                im = self.intention_encoder(intention)
                ia, _ = self.intention_attn(iq, im, im, need_weights=False)
                ia = self.norm_intention(iq + self.dropout(ia)).reshape(b, k, h, -1)
                step_contexts.append(ia)
                contexts.append(ia.mean(dim=2))
            refined = candidates
            residual_local = torch.zeros_like(candidates)
            if self.variant in {'B', 'AB'}:
                residual_input = torch.cat([q, *step_contexts, hist_step], dim=-1)
                residual_local = self.max_candidate_residual * torch.tanh(self.residual_head(residual_input))
                time_scale = torch.linspace(1.0 / h, 1.0, h, device=X.device)[None, None, :, None]
                residual_local = residual_local * time_scale
                forward, left, up = (basis[:, 0], basis[:, 1], basis[:, 2])
                residual_world = local_vector_to_world(residual_local, forward, left, up)
                refined = candidates + residual_world
            score_input = torch.cat([hist_mode, q_pool, dyn_emb, *contexts], dim=-1)
            delta_logit = self.max_logit_correction * torch.tanh(self.scorer(score_input).squeeze(-1))
            logits = torch.log(torch.clamp(raw_pi, min=1e-08)) + delta_logit
            pi = torch.softmax(logits, dim=1)
            pred = torch.sum(pi[:, :, None, None] * refined, dim=1)
            return (pred, pi, refined, residual_local, delta_logit)

    def candidate_errors(candidates, y, fde_weight):
        dist = torch.linalg.norm(candidates - y[:, None, :, :], dim=-1)
        return dist.mean(dim=2) + float(fde_weight) * dist[:, :, -1]

    def e4_loss(pred, pi, refined, original, residual_local, raw_pi, y, args, rank_lambda, gate_anchor_lambda, residual_anchor_lambda, bestofk_lambda):
        forecast = dk.forecast_loss(pred, y, args)
        err = candidate_errors(refined, y, args.rank_target_fde_weight)
        target = torch.softmax(-err.detach() / max(args.rank_target_temperature, 1e-06), dim=1)
        rank = torch.mean(torch.sum(target * (torch.log(torch.clamp(target, min=1e-08)) - torch.log(torch.clamp(pi, min=1e-08))), dim=1))
        gate_anchor = torch.mean((pi - raw_pi) ** 2)
        residual_anchor = torch.mean(residual_local ** 2)
        bestofk = torch.mean(torch.min(err, dim=1).values)
        total = forecast + float(rank_lambda) * rank + float(gate_anchor_lambda) * gate_anchor + float(residual_anchor_lambda) * residual_anchor + float(bestofk_lambda) * bestofk
        return (total, forecast, rank, gate_anchor, residual_anchor, bestofk)

    def infer_model(model, pack, corridor, intention, batch_size, device):
        model.eval()
        preds, pis, refined_all, residual_all = ([], [], [], [])
        with torch.no_grad():
            for start in range(0, len(pack['X']), batch_size):
                sl = slice(start, min(len(pack['X']), start + batch_size))
                tensors = [torch.from_numpy(arr[sl]).float().to(device) for arr in [pack['X'], pack['raw_pi'], pack['candidates'], pack['query'], pack['dyn'], pack['basis'], corridor, intention]]
                pred, pi, refined, residual, _ = model(*tensors)
                preds.append(pred.cpu().numpy())
                pis.append(pi.cpu().numpy())
                refined_all.append(refined.cpu().numpy())
                residual_all.append(residual.cpu().numpy())
        return (np.concatenate(preds), np.concatenate(pis), np.concatenate(refined_all), np.concatenate(residual_all))

    def train_one(variant, arrays, tokens, data_pack, args, device, rank_lambda, gate_anchor_lambda, residual_anchor_lambda, bestofk_lambda):
        tr = arrays['train']
        va = arrays['val']
        loader = DataLoader(E4Dataset(tr['X'], tr['raw_pi'], tr['candidates'], tr['query'], tr['dyn'], tr['basis'], tokens['train']['corridor'], tokens['train']['intention'], tr['y']), batch_size=args.e4_batch_size, shuffle=True, pin_memory=device.type == 'cuda')
        model = E4ABModel(variant, tr['X'].shape[-1], tr['query'].shape[-1], tr['dyn'].shape[-1], tokens['train']['corridor'].shape[-1], tokens['train']['intention'].shape[-1], args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.e4_lr, weight_decay=args.e4_weight_decay)
        best_score = None
        best_state = None
        best_epoch = -1
        bad = 0
        history = []
        for epoch in range(1, args.e4_epochs + 1):
            model.train()
            sums = {'loss': 0.0, 'forecast': 0.0, 'rank': 0.0, 'gate_anchor': 0.0, 'residual_anchor': 0.0, 'bestofk': 0.0, 'count': 0}
            for batch in loader:
                xb, pib, cb, qb, db, bb, cob, inb, yb = [x.to(device) for x in batch]
                pred, pi, refined, residual, _ = model(xb, pib, cb, qb, db, bb, cob, inb)
                loss_values = e4_loss(pred, pi, refined, cb, residual, pib, yb, args, rank_lambda, gate_anchor_lambda, residual_anchor_lambda, bestofk_lambda)
                loss, forecast, rank, gate_anchor, residual_anchor, bestofk = loss_values
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                bn = len(xb)
                sums['count'] += bn
                for key, value in zip(['loss', 'forecast', 'rank', 'gate_anchor', 'residual_anchor', 'bestofk'], loss_values):
                    sums[key] += float(value.item()) * bn
            val_pred, val_pi, val_refined, val_residual = infer_model(model, va, tokens['val']['corridor'], tokens['val']['intention'], args.e4_batch_size, device)
            met = base.compute_position_metrics(va['y'], val_pred)
            score = score_metrics(met, args)
            gate_l1 = float(np.abs(val_pi - va['raw_pi']).sum(axis=1).mean())
            mean_residual = float(np.linalg.norm(val_residual, axis=-1).mean())
            row = {'variant': variant, 'rank_lambda': rank_lambda, 'gate_anchor_lambda': gate_anchor_lambda, 'residual_anchor_lambda': residual_anchor_lambda, 'bestofk_lambda': bestofk_lambda, 'epoch': epoch, **{f'train_{key}': sums[key] / max(1, sums['count']) for key in ['loss', 'forecast', 'rank', 'gate_anchor', 'residual_anchor', 'bestofk']}, 'val_score_raw': score, 'val_gate_l1_change': gate_l1, 'val_mean_residual_local_norm': mean_residual, **{f'val_{key}': value for key, value in met.items()}}
            history.append(row)
            if best_score is None or score < best_score:
                best_score, best_state, best_epoch, bad = (score, copy.deepcopy(model.state_dict()), epoch, 0)
            else:
                bad += 1
            if epoch == 1 or epoch % args.log_every == 0 or bad == 0:
                print(f"[E4-{variant} rank={rank_lambda:g} gate={gate_anchor_lambda:g} res={residual_anchor_lambda:g} bok={bestofk_lambda:g}] ep={epoch:03d} score={score:.6f} ADE={met['ade']:.6f} FDE={met['fde']:.6f} gateL1={gate_l1:.4f} residual={mean_residual:.4f}")
            if bad >= args.e4_patience:
                break
        model.load_state_dict(best_state)
        val_pred, val_pi, val_refined, val_residual = infer_model(model, va, tokens['val']['corridor'], tokens['val']['intention'], args.e4_batch_size, device)
        safety, safety_sweep = dk.tune_safety(data_pack['val_data'], data_pack['val_global'], val_pred, args)
        val_safe, eff, _ = dk.apply_safety(data_pack['val_global'], val_pred, safety['alpha'], safety['safety_tau'])
        val_safe_metrics = base.compute_position_metrics(va['y'], val_safe)
        refined_err = np.linalg.norm(val_refined - va['y'][:, None], axis=-1)
        refined_oracle_ade = float(refined_err.mean(axis=2).min(axis=1).mean())
        info = {'variant': variant, 'rank_lambda': float(rank_lambda), 'gate_anchor_lambda': float(gate_anchor_lambda), 'residual_anchor_lambda': float(residual_anchor_lambda), 'bestofk_lambda': float(bestofk_lambda), 'best_epoch': int(best_epoch), 'val_raw_score': float(best_score), 'val_safety_score': float(score_metrics(val_safe_metrics, args)), 'safety_alpha': float(safety['alpha']), 'safety_tau': float(safety['safety_tau']), 'mean_effective_safety_alpha_val': float(eff.mean()), 'val_gate_l1_change': float(np.abs(val_pi - va['raw_pi']).sum(axis=1).mean()), 'val_mean_residual_local_norm': float(np.linalg.norm(val_residual, axis=-1).mean()), 'val_refined_candidate_oracle_ade': refined_oracle_ade, **{f'val_safe_{key}': float(value) for key, value in val_safe_metrics.items()}}
        return (model, pd.DataFrame(history), info, safety, safety_sweep)

    def method_row(data, pred, method, voxel_map, args, extra=None):
        row = {'method': method, 'horizon': int(data.future_global.shape[1]), **base.compute_position_metrics(data.future_global[:, :, :3], pred), **e1.prediction_map_diagnostics(voxel_map, pred, args.unsafe_clearance)}
        if extra:
            row.update(extra)
        return row

    def candidate_diagnostics(pi, refined, y, residual):
        dist = np.linalg.norm(refined - y[:, None], axis=-1)
        loss = dist.mean(axis=2) + 0.25 * dist[:, :, -1]
        oracle = loss.argmin(axis=1)
        selected = pi.argmax(axis=1)
        oracle_pred = refined[np.arange(len(y)), oracle]
        oracle_metrics = base.compute_position_metrics(y, oracle_pred)
        return {'selector_top1_equals_refined_oracle_rate': float(np.mean(selected == oracle)), 'mean_selected_refined_candidate_regret': float(np.mean(loss[np.arange(len(y)), selected] - loss[np.arange(len(y)), oracle])), 'refined_candidate_oracle_ade': float(oracle_metrics['ade']), 'refined_candidate_oracle_fde': float(oracle_metrics['fde']), 'mean_candidate_residual_local_norm': float(np.linalg.norm(residual, axis=-1).mean())}

    def subset_metrics(data, predictions, voxel_map, base_candidates, args):
        gt = data.future_global[:, :, :3]
        gt_clearance, _ = voxel_map.query_clearance(gt)
        gt_min = gt_clearance.min(axis=1)
        cand_feat = e1.candidate_environment_features(voxel_map, base_candidates, 2.0)
        gap = cand_feat['min_clearance'].max(axis=1) - cand_feat['min_clearance'].min(axis=1)
        masks = {'all': np.ones(len(gt), dtype=bool), f'gt_near_obstacle_le_{args.report_near_clearance:g}m': gt_min <= args.report_near_clearance, f'gt_open_gt_{args.report_near_clearance:g}m': gt_min > args.report_near_clearance, f'base_candidate_clearance_gap_ge_{args.report_candidate_gap:g}m': gap >= args.report_candidate_gap}
        rows = []
        for subset, mask in masks.items():
            if not mask.any():
                continue
            for method, pred in predictions.items():
                rows.append({'subset': subset, 'num_samples': int(mask.sum()), **method_row(subset_view(data, mask), pred[mask], method, voxel_map, args)})
        return pd.DataFrame(rows)

    def main():
        p = argparse.ArgumentParser(description='E4 A/B/A+B map-usage experiment over frozen D-KMoE candidates.')
        p.add_argument('--data_root', required=True)
        p.add_argument('--binvox_file', required=True)
        p.add_argument('--out_dir', required=True)
        p.add_argument('--input_len', type=int, default=20)
        p.add_argument('--horizon', type=int, default=10)
        p.add_argument('--stride', type=int, default=5)
        p.add_argument('--dt', type=float, default=0.1)
        p.add_argument('--K', type=int, default=12)
        p.add_argument('--ridge_model', type=float, default=1.0)
        p.add_argument('--ridge_gate', type=float, default=1.0)
        p.add_argument('--min_cluster_samples', type=int, default=80)
        p.add_argument('--alt_iter', type=int, default=12)
        p.add_argument('--kmeans_iter', type=int, default=50)
        p.add_argument('--global_ridge_xy_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        p.add_argument('--global_ridge_z_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        p.add_argument('--hidden_dim', type=int, default=128)
        p.add_argument('--gru_layers', type=int, default=2)
        p.add_argument('--dropout', type=float, default=0.2)
        p.add_argument('--max_scale_deviation', type=float, default=0.25)
        p.add_argument('--anchor_lambda_grid', default='0,0.01,0.1')
        p.add_argument('--balance_lambda_grid', default='0,0.001')
        p.add_argument('--epochs', type=int, default=80)
        p.add_argument('--patience', type=int, default=12)
        p.add_argument('--batch_size', type=int, default=512)
        p.add_argument('--lr', type=float, default=0.001)
        p.add_argument('--weight_decay', type=float, default=0.0001)
        p.add_argument('--grad_clip', type=float, default=1.0)
        p.add_argument('--log_every', type=int, default=10)
        p.add_argument('--train_ade_weight', type=float, default=0.5)
        p.add_argument('--train_fde_weight', type=float, default=0.25)
        p.add_argument('--selection_metric', default='combo', choices=['pos_rmse', 'ade', 'fde', 'combo'])
        p.add_argument('--combo_ade_weight', type=float, default=0.5)
        p.add_argument('--combo_fde_weight', type=float, default=0.25)
        p.add_argument('--alpha_grid', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        p.add_argument('--safety_quantile_grid', default='0.4,0.6,0.8,0.95,1.0')
        p.add_argument('--map_expected_xyz', nargs=3, type=int, required=True)
        p.add_argument('--map_center_xyz', nargs=3, type=float, required=True)
        p.add_argument('--map_resolution', type=float, default=1.0)
        p.add_argument('--map_axis_mode', default='swap_yz', choices=['swap_yz', 'current_xyz'])
        p.add_argument('--unsafe_clearance', type=float, default=1.0)
        p.add_argument('--report_near_clearance', type=float, default=5.0)
        p.add_argument('--report_candidate_gap', type=float, default=2.0)
        p.add_argument('--map_clearance_clip', type=float, default=20.0)
        p.add_argument('--boundary_scale', type=float, default=5.0)
        p.add_argument('--map_query_chunk', type=int, default=384)
        p.add_argument('--corridor_anchor_indices', default='2,5,9')
        p.add_argument('--corridor_lateral_offsets', default='-8,0,8')
        p.add_argument('--corridor_up_offsets', default='-6,0,6')
        p.add_argument('--corridor_token_topk', type=int, default=0, help='正式反事实实验必须为 0；保留固定 token 网格，避免选择位置泄露地图语义。')
        p.add_argument('--motion_forward_offsets', default='8,16,24')
        p.add_argument('--motion_left_offsets', default='-12,-6,0,6,12')
        p.add_argument('--motion_up_offsets', default='-6,0,6')
        p.add_argument('--motion_ray_fractions', default='0.25,0.5,0.75,1')
        p.add_argument('--variants', default='A,B,AB')
        p.add_argument('--d_model', type=int, default=64)
        p.add_argument('--e4_hidden_dim', type=int, default=96)
        p.add_argument('--attn_heads', type=int, default=4)
        p.add_argument('--e4_dropout', type=float, default=0.15)
        p.add_argument('--max_logit_correction', type=float, default=2.0)
        p.add_argument('--max_candidate_residual', type=float, default=2.0)
        p.add_argument('--rank_target_temperature', type=float, default=0.05)
        p.add_argument('--rank_target_fde_weight', type=float, default=0.25)
        p.add_argument('--rank_lambda_grid', default='0.1,0.5')
        p.add_argument('--gate_anchor_lambda_grid', default='0.01,0.1')
        p.add_argument('--residual_anchor_lambda_grid', default='0.01,0.1')
        p.add_argument('--bestofk_lambda_grid', default='0.05,0.1')
        p.add_argument('--e4_epochs', type=int, default=70)
        p.add_argument('--e4_patience', type=int, default=10)
        p.add_argument('--e4_batch_size', type=int, default=96)
        p.add_argument('--e4_lr', type=float, default=0.001)
        p.add_argument('--e4_weight_decay', type=float, default=0.0001)
        p.add_argument('--device', default='auto')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--fast', action='store_true')
        args = p.parse_args()
        variants = parse_str_list(args.variants)
        if any((v not in {'A', 'B', 'AB'} for v in variants)):
            raise ValueError('--variants 只能为 A,B,AB 的组合。')
        if args.d_model % args.attn_heads != 0:
            raise ValueError('--d_model 必须能被 --attn_heads 整除。')
        if args.fast:
            args.alt_iter = min(args.alt_iter, 8)
            args.epochs = min(args.epochs, 50)
            args.patience = min(args.patience, 10)
            args.anchor_lambda_grid = '0,0.01,0.1'
            args.balance_lambda_grid = '0,0.001'
            args.e4_epochs = min(args.e4_epochs, 45)
            args.e4_patience = min(args.e4_patience, 8)
            args.rank_lambda_grid = '0.5'
            args.gate_anchor_lambda_grid = '0.01'
            args.residual_anchor_lambda_grid = '0.05'
            args.bestofk_lambda_grid = '0.05'
        set_seed(args.seed)
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        out_dir = Path(args.out_dir)
        ensure_dir(out_dir)
        print('device =', device)
        print('variants =', variants, '| K =', args.K)
        print('IMPORTANT: no action labels are used.')
        print('IMPORTANT: A selects frozen Koopman candidates using corridor tokens.')
        print('IMPORTANT: B/AB produce bounded map-conditioned residuals around frozen Koopman candidates.')
        print('IMPORTANT: map decoding and NED-Z mapping use the previously validated convention.')
        voxel_map = e1.AirSimNEDClearanceMap.from_binvox(args.binvox_file, args.map_expected_xyz, args.map_center_xyz, args.map_resolution, axis_mode=getattr(args, 'map_axis_mode', 'swap_yz'))
        (out_dir / 'map_metadata_validated_ned.json').write_text(json.dumps(voxel_map.metadata(), ensure_ascii=False, indent=2), encoding='utf-8')
        root = Path(args.data_root)
        tr = bk.make_split_data(bk.load_split_chains(root, 'train'), args.input_len, args.horizon, args.stride)
        va = bk.make_split_data(bk.load_split_chains(root, 'val'), args.input_len, args.horizon, args.stride)
        te = bk.make_split_data(bk.load_split_chains(root, 'test'), args.input_len, args.horizon, args.stride)
        print(f'samples: train={len(tr.hist_global)}, val={len(va.hist_global)}, test={len(te.hist_global)}')
        global_pack = base.fit_no_action_global(tr, va, args)
        gtr = base.predict_no_action_global(global_pack, tr, args).astype(np.float32)
        gva = base.predict_no_action_global(global_pack, va, args).astype(np.float32)
        gte = base.predict_no_action_global(global_pack, te, args).astype(np.float32)
        Xtr0, Xva0, Xte0 = (prior.make_sequence_features(tr, args.dt), prior.make_sequence_features(va, args.dt), prior.make_sequence_features(te, args.dt))
        seq_std = prior.SeqStandardizer.fit(Xtr0)
        Xtr, Xva, Xte = (seq_std.transform(Xtr0), seq_std.transform(Xva0), seq_std.transform(Xte0))
        ytr = tr.future_global[:, :, :3].astype(np.float32)
        yva = va.future_global[:, :, :3].astype(np.float32)
        yte = te.future_global[:, :, :3].astype(np.float32)
        frozen = e2.fit_frozen_base_dkmoe(tr, va, te, gtr, gva, gte, Xtr, Xva, Xte, ytr, yva, args, device, out_dir)
        qtr, dtr, btr = build_candidate_queries(tr, frozen['train']['raw'], frozen['train']['calibrated'], frozen['train']['pi'])
        qva, dva, bva = build_candidate_queries(va, frozen['val']['raw'], frozen['val']['calibrated'], frozen['val']['pi'])
        qte, dte, bte = build_candidate_queries(te, frozen['test']['raw'], frozen['test']['calibrated'], frozen['test']['pi'])
        q_std, d_std = (LastDimStandardizer.fit(qtr), LastDimStandardizer.fit(dtr))
        qtr, qva, qte = (q_std.transform(qtr), q_std.transform(qva), q_std.transform(qte))
        dtr, dva, dte = (d_std.transform(dtr), d_std.transform(dva), d_std.transform(dte))
        print('building candidate-centric corridor tokens (A/AB)...')
        ctr_raw, cgeom_dim, csem_dim = build_corridor_tokens(tr, frozen['train']['calibrated'], voxel_map, args)
        cva_raw, _, _ = build_corridor_tokens(va, frozen['val']['calibrated'], voxel_map, args)
        cte_raw, _, _ = build_corridor_tokens(te, frozen['test']['calibrated'], voxel_map, args)
        corridor_std = LastDimStandardizer.fit(ctr_raw)
        ctr, cva, cte = (corridor_std.transform(ctr_raw), corridor_std.transform(cva_raw), corridor_std.transform(cte_raw))
        ctr_zero = zero_semantic(ctr_raw, cgeom_dim, corridor_std)
        cva_zero = zero_semantic(cva_raw, cgeom_dim, corridor_std)
        cte_zero = zero_semantic(cte_raw, cgeom_dim, corridor_std)
        print('building map-conditioned motion-query tokens (B/AB)...')
        itr_raw, igeom_dim, isem_dim = build_motion_query_tokens(tr, voxel_map, args)
        iva_raw, _, _ = build_motion_query_tokens(va, voxel_map, args)
        ite_raw, _, _ = build_motion_query_tokens(te, voxel_map, args)
        intention_std = LastDimStandardizer.fit(itr_raw)
        itr, iva, ite = (intention_std.transform(itr_raw), intention_std.transform(iva_raw), intention_std.transform(ite_raw))
        itr_zero = zero_semantic(itr_raw, igeom_dim, intention_std)
        iva_zero = zero_semantic(iva_raw, igeom_dim, intention_std)
        ite_zero = zero_semantic(ite_raw, igeom_dim, intention_std)
        arrays = {'train': {'X': Xtr, 'raw_pi': frozen['train']['pi'], 'candidates': frozen['train']['calibrated'], 'query': qtr, 'dyn': dtr, 'basis': btr, 'y': ytr}, 'val': {'X': Xva, 'raw_pi': frozen['val']['pi'], 'candidates': frozen['val']['calibrated'], 'query': qva, 'dyn': dva, 'basis': bva, 'y': yva}, 'test': {'X': Xte, 'raw_pi': frozen['test']['pi'], 'candidates': frozen['test']['calibrated'], 'query': qte, 'dyn': dte, 'basis': bte, 'y': yte}}
        tokens = {'train': {'corridor': ctr, 'intention': itr}, 'val': {'corridor': cva, 'intention': iva}, 'test': {'corridor': cte, 'intention': ite}}
        data_pack = {'val_data': va, 'val_global': gva}
        selected = {}
        histories, selection_rows, safety_sweeps = ([], [], [])
        for variant in variants:
            print(f'\n========== Training E4 variant {variant} ==========')
            best = None
            residual_grid = [0.0] if variant == 'A' else parse_float_list(args.residual_anchor_lambda_grid)
            bok_grid = [0.0] if variant == 'A' else parse_float_list(args.bestofk_lambda_grid)
            for rank_lambda in parse_float_list(args.rank_lambda_grid):
                for gate_lambda in parse_float_list(args.gate_anchor_lambda_grid):
                    for residual_lambda in residual_grid:
                        for bok_lambda in bok_grid:
                            set_seed(args.seed + len(variant) * 31 + int(rank_lambda * 100) + int(gate_lambda * 1000))
                            model, hist, info, safety, sweep = train_one(variant, arrays, tokens, data_pack, args, device, rank_lambda, gate_lambda, residual_lambda, bok_lambda)
                            histories.append(hist)
                            sweep['variant'] = variant
                            sweep['rank_lambda'] = rank_lambda
                            sweep['gate_anchor_lambda'] = gate_lambda
                            sweep['residual_anchor_lambda'] = residual_lambda
                            sweep['bestofk_lambda'] = bok_lambda
                            safety_sweeps.append(sweep)
                            selection_rows.append(info)
                            if best is None or info['val_safety_score'] < best['info']['val_safety_score']:
                                best = {'model': copy.deepcopy(model).cpu(), 'info': info, 'safety': safety}
            selected[variant] = best
        pd.concat(histories, ignore_index=True).to_csv(out_dir / 'e4_training_history.csv', index=False)
        pd.DataFrame(selection_rows).to_csv(out_dir / 'e4_variant_selection_val.csv', index=False)
        pd.concat(safety_sweeps, ignore_index=True).to_csv(out_dir / 'e4_global_safety_sweep_val.csv', index=False)
        predictions = {'v5_no_action_global': gte, 'differentiable_koopman_moe_raw': frozen['test']['raw'], 'differentiable_koopman_moe_safety': frozen['test']['safe']}
        rows = [method_row(te, gte, 'v5_no_action_global', voxel_map, args), method_row(te, frozen['test']['raw'], 'differentiable_koopman_moe_raw', voxel_map, args), method_row(te, frozen['test']['safe'], 'differentiable_koopman_moe_safety', voxel_map, args)]
        counterfactual_rows = []
        rng = np.random.default_rng(args.seed + 2026)
        shuffled_idx = rng.permutation(len(te.hist_global))
        for variant in variants:
            model = selected[variant]['model'].to(device)
            pred, pi, refined, residual = infer_model(model, arrays['test'], cte, ite, args.e4_batch_size, device)
            safe = selected[variant]['safety']
            pred_safe, eff, _ = dk.apply_safety(gte, pred, safe['alpha'], safe['safety_tau'])
            name = f'e4_{variant.lower()}_correct_map_plus_global_safety'
            predictions[name] = pred_safe
            extra = {**selected[variant]['info'], **candidate_diagnostics(pi, refined, yte, residual), 'test_mean_effective_safety_alpha': float(eff.mean())}
            rows.append(method_row(te, pred_safe, name, voxel_map, args, extra))
            conditions = {'correct_map': (cte, ite), 'shuffled_map': (cte[shuffled_idx], ite[shuffled_idx]), 'zero_semantic_map': (cte_zero, ite_zero)}
            for condition, (corr_input, int_input) in conditions.items():
                cpred, cpi, crefined, cresidual = infer_model(model, arrays['test'], corr_input, int_input, args.e4_batch_size, device)
                cpred_safe, _, _ = dk.apply_safety(gte, cpred, safe['alpha'], safe['safety_tau'])
                counterfactual_rows.append(method_row(te, cpred_safe, f'e4_{variant.lower()}_{condition}_plus_global_safety', voxel_map, args, {'trained_variant': variant, 'map_condition': condition, **candidate_diagnostics(cpi, crefined, yte, cresidual)}))
        compare_df = pd.DataFrame(rows)
        counter_df = pd.DataFrame(counterfactual_rows)
        compare_df.to_csv(out_dir / 'compare_e4_ab_test.csv', index=False)
        counter_df.to_csv(out_dir / 'e4_map_counterfactual_test.csv', index=False)
        subset_df = subset_metrics(te, predictions, voxel_map, frozen['test']['calibrated'], args)
        subset_df.to_csv(out_dir / 'e4_subset_metrics_test.csv', index=False)
        counter_predictions = {}
        for variant in variants:
            model = selected[variant]['model'].to(device)
            safe = selected[variant]['safety']
            for condition, (corr_input, int_input) in {'correct_map': (cte, ite), 'shuffled_map': (cte[shuffled_idx], ite[shuffled_idx]), 'zero_semantic_map': (cte_zero, ite_zero)}.items():
                pred, _, _, _ = infer_model(model, arrays['test'], corr_input, int_input, args.e4_batch_size, device)
                pred, _, _ = dk.apply_safety(gte, pred, safe['alpha'], safe['safety_tau'])
                counter_predictions[f'e4_{variant.lower()}_{condition}'] = pred
        counter_subset_df = subset_metrics(te, counter_predictions, voxel_map, frozen['test']['calibrated'], args)
        counter_subset_df.to_csv(out_dir / 'e4_map_counterfactual_subset_test.csv', index=False)
        for variant, best in selected.items():
            torch.save({'variant': variant, 'K': args.K, 'model_state_dict': best['model'].state_dict(), 'selection_info': best['info'], 'global_safety': best['safety'], 'corridor_geometry_dim': int(cgeom_dim), 'corridor_semantic_dim': int(csem_dim), 'intention_geometry_dim': int(igeom_dim), 'intention_semantic_dim': int(isem_dim), 'map_metadata': voxel_map.metadata(), 'args': vars(args)}, out_dir / f'e4_{variant.lower()}_model.pt')
        report = {'experiment': 'E4 A/B/A+B map-usage experiment over frozen D-KMoE candidates', 'interpretation': {'A': 'Candidate-centric corridor map tokens modify selector weights only.', 'B': 'Map-conditioned free-space motion queries generate bounded residual candidates around frozen Koopman anchors.', 'AB': 'Corridor tokens and motion-query tokens jointly affect candidate generation and selection.', 'success_criterion': 'A/B/AB must outperform D-KMoE Safety and degrade under shuffled/zero maps to claim environment-driven gain.'}, 'strict_protocol': {'action_labels_used': False, 'base_koopman_candidates_frozen': True, 'map_coordinate_convention': 'validated AirSim binvox decode with reversed NED-Z mapping'}, 'selected_validation_hyperparameters': {k: v['info'] for k, v in selected.items()}, 'test_results': compare_df.to_dict(orient='records'), 'counterfactual_test_results': counter_df.to_dict(orient='records'), 'args': vars(args)}
        (out_dir / 'report_e4_ab_experiment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('\n=== E4 A/B/A+B test performance ===')
        print(compare_df.to_string(index=False))
        print('\n=== E4 counterfactual map tests ===')
        print(counter_df.to_string(index=False))
        print('\nKey files:')
        for fname in ['compare_e4_ab_test.csv', 'e4_variant_selection_val.csv', 'e4_subset_metrics_test.csv', 'e4_map_counterfactual_test.csv', 'e4_map_counterfactual_subset_test.csv', 'report_e4_ab_experiment.json']:
            print(' ', out_dir / fname)

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

e4_map_usage_ab_experiment_v2 = _init_e4_map_usage_ab_experiment_v2()


# ========================================================================================
# SECTION: TOPOKOOP FORMAL
# ========================================================================================
def _init_topokoop_formal():
    """
    topokoop_formal.py

    TopoKoop (formal refactor of E6-C+AB): Global Topological Scene-Conditioned Candidate Generation
            followed by Local Fine Motion-Query + Corridor Refinement
    =============================================================================

    目标
    ----
    在 E5 已证明“Global-Sparse + Local-Fine”有效的基础上，进一步把全局环境
    从并行 residual 分支前移到 candidate generation 阶段：

        Stage C:
            全局稀疏可飞空间子图 -> mode intention context
            -> 生成 global-conditioned Koopman candidate anchors

        Stage AB:
            围绕 Stage C 已修正的 candidate anchors 重新查询局部精细地图
            -> local motion-query refinement + corridor-aware candidate selection

    与 E5 的关键区别
    -----------------
    E5:
        原始 Koopman candidates 同时接收 global/local residual。

    E6:
        1) Global topology 先生成新的 candidate anchors；
        2) Local fine tokens 必须围绕这些新 anchors 重新构造；
        3) Local AB 再进行第二阶段精修。

    这是一版严格可运行的两阶段 C+AB 验证实现。Stage C 仍以冻结 Koopman
    candidates 为动力学锚点，学习有界环境条件化 adapter，而不是直接重估 Koopman
    算子矩阵。若该版本成立，后续再升级为可微三维 SDF 动态查询的一体化端到端模型。

    严格协议
    --------
    - 默认 label_free 模式不使用动作/行为标签；action_label 模式仅替换候选生成器，不改动后续环境模块；
    - 基础 D-KMoE / Koopman candidates 冻结；
    - train/val 的未来轨迹仅用作常规预测监督和 candidate-ranking supervision；
    - Stage C 的超参数只由 validation 选择；
    - Stage AB 的超参数只由 validation 选择；
    - test 只报告选定模型结果；
    - 同轮训练一个 Local-AB baseline，与 C+AB 公平比较；
    - 对 C+AB 进行分解反事实测试：
        correct_graph + correct_local
        shuffled_graph + correct_local
        zero_attr_same_graph + correct_local
        null_uniform_graph + correct_local
        correct_graph + shuffled_local
        correct_graph + zero_local
        null_uniform_graph + zero_local

    地图约定（保持前序验证结果）
    --------------------------
    AirSim binvox:
        idx = x + nx * (z + nz * y)
        occ_xyz = flat.reshape((ny, nz, nx)).transpose(2, 0, 1)

    AirSim NED:
        z_ned(k) = center_z - (k - nz//2) * resolution

    所有轨迹/地图交互采用严格三维查询，不使用跨高度投影图判断碰撞。

    依赖文件（与本文件同目录）
    ------------------------
    env_reweight_differentiable_koopman_moe_e1.py
    e2_learned_environment_conditioned_koopman_selector.py
    e4_map_usage_ab_experiment_v2.py
    differentiable_koopman_moe.py
    unsup_switch_selector_enhance_standalone_v2.py
    unsup_koopman_neural_hsmm_selector_v2.py
    behavior_mode_endpoint_koopman_v2_hmm_selector.py
    axis_feature_decoupled_position_behavior_koopman_v5.py
    """
    import argparse
    import copy
    import json
    import random
    from collections import deque
    from pathlib import Path
    from typing import Dict, Tuple
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    e1 = env_reweight_differentiable_koopman_moe_e1
    e2 = e2_learned_environment_conditioned_koopman_selector
    e4 = e4_map_usage_ab_experiment_v2
    v5 = axis_feature_decoupled_position_behavior_koopman_v5
    dk = e1.dk
    base = e1.base
    prior = e1.prior
    bk = e1.bk

    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def parse_float_list(text):
        return [float(x.strip()) for x in str(text).split(',') if x.strip()]

    def parse_str_list(text):
        return [x.strip() for x in str(text).split(',') if x.strip()]

    def parse_range(text):
        values = parse_float_list(text)
        if len(values) != 3:
            raise ValueError(f'range 参数必须是 start,end,step，当前为：{text}')
        start, end, step = values
        if step <= 0 or end < start:
            raise ValueError(f'range 参数不合法：{text}')
        arr = np.arange(start, end + 0.25 * step, step, dtype=np.float32)
        return arr[arr <= end + 1e-05]

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def metric_score(metrics, args):
        return e1.metric_score(metrics, args)

    class MaskedStandardizer:

        def __init__(self, mean, std):
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)

        @classmethod
        def fit(cls, x, mask=None, eps=1e-06):
            arr = np.asarray(x, dtype=np.float32)
            if mask is None:
                flat = arr.reshape(-1, arr.shape[-1])
            else:
                flat = arr[np.asarray(mask, dtype=bool)]
            return cls(flat.mean(axis=0), np.maximum(flat.std(axis=0), eps))

        def transform(self, x):
            return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    class DataView:
        pass

    def subset_view(data, mask):
        out = DataView()
        out.future_global = data.future_global[mask]
        return out

    def method_row(data, pred, method, voxel_map, args, extra=None):
        row = {'method': method, 'horizon': int(data.future_global.shape[1]), **base.compute_position_metrics(data.future_global[:, :, :3], pred), **e1.prediction_map_diagnostics(voxel_map, pred, args.unsafe_clearance)}
        if extra:
            row.update(extra)
        return row

    def candidate_diagnostics(pi, candidates, y, residual=None):
        dist = np.linalg.norm(candidates - y[:, None, :, :], axis=-1)
        err = dist.mean(axis=2) + 0.25 * dist[:, :, -1]
        oracle = err.argmin(axis=1)
        selected = pi.argmax(axis=1)
        oracle_pred = candidates[np.arange(len(y)), oracle]
        met = base.compute_position_metrics(y, oracle_pred)
        result = {'selector_top1_equals_refined_oracle_rate': float(np.mean(selected == oracle)), 'mean_selected_refined_candidate_regret': float(np.mean(err[np.arange(len(y)), selected] - err[np.arange(len(y)), oracle])), 'refined_candidate_oracle_ade': float(met['ade']), 'refined_candidate_oracle_fde': float(met['fde'])}
        if residual is not None:
            result['mean_candidate_residual_local_norm'] = float(np.linalg.norm(residual, axis=-1).mean())
        return result

    def make_coarse_offsets(args):
        f = parse_range(args.graph_forward_range)
        l = parse_range(args.graph_left_range)
        u = parse_range(args.graph_up_range)
        ff, ll, uu = np.meshgrid(f, l, u, indexing='ij')
        offsets = np.stack([ff.ravel(), ll.ravel(), uu.ravel()], axis=1).astype(np.float32)
        shape = (len(f), len(l), len(u))
        return (offsets, shape)

    def _adjacency_features(free_grid):
        """
        基于粗节点占用状态构造六方向邻接特征。
        这是可飞空间粗拓扑近似；真实碰撞判定仍使用严格三维地图查询。
        """
        nf, nl, nu = free_grid.shape
        neighbours = []
        for axis, shift in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
            shifted = np.zeros_like(free_grid, dtype=bool)
            if axis == 0 and shift == 1:
                shifted[:-1] = free_grid[1:]
            elif axis == 0 and shift == -1:
                shifted[1:] = free_grid[:-1]
            elif axis == 1 and shift == 1:
                shifted[:, :-1] = free_grid[:, 1:]
            elif axis == 1 and shift == -1:
                shifted[:, 1:] = free_grid[:, :-1]
            elif axis == 2 and shift == 1:
                shifted[:, :, :-1] = free_grid[:, :, 1:]
            elif axis == 2 and shift == -1:
                shifted[:, :, 1:] = free_grid[:, :, :-1]
            neighbours.append((free_grid & shifted).astype(np.float32))
        edge = np.stack(neighbours, axis=-1)
        degree = edge.sum(axis=-1, keepdims=True) / 6.0
        direction = np.stack([edge[..., 0] - edge[..., 1], edge[..., 2] - edge[..., 3], edge[..., 4] - edge[..., 5]], axis=-1)
        return (edge, degree, direction)

    def _reachable_component(free_grid, offsets_grid):
        """从距当前点最近的自由粗节点出发提取连通分量。"""
        reachable = np.zeros_like(free_grid, dtype=bool)
        free_indices = np.argwhere(free_grid)
        if len(free_indices) == 0:
            return reachable
        positions = offsets_grid[free_grid]
        start = tuple(free_indices[np.argmin(np.linalg.norm(positions, axis=1))])
        q = deque([start])
        reachable[start] = True
        nf, nl, nu = free_grid.shape
        while q:
            i, j, k = q.popleft()
            for di, dj, dk_ in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                ni, nj, nk = (i + di, j + dj, k + dk_)
                if 0 <= ni < nf and 0 <= nj < nl and (0 <= nk < nu):
                    if free_grid[ni, nj, nk] and (not reachable[ni, nj, nk]):
                        reachable[ni, nj, nk] = True
                        q.append((ni, nj, nk))
        return reachable

    def build_topological_graph_tokens(data, voxel_map, offsets, shape, args):
        """
        从当前 UAV 局部大范围空间中构建稀疏可飞空间子图 token。

        token 位置本身包含真实拓扑（仅保留可达/自由的 coarse nodes）。
        因此反事实中：
            - zero_attr_same_graph 只消融节点属性，仍保留拓扑；
            - null_uniform_graph 才用于替换真实拓扑。
        """
        origin, forward, left, up, heading_valid = e4.local_basis_from_history(data)
        n = len(origin)
        total_nodes = len(offsets)
        max_nodes = int(args.graph_max_nodes)
        nf, nl, nu = shape
        offset_grid = offsets.reshape(nf, nl, nu, 3)
        geometry_all = np.concatenate([e4.fourier_xyz(offsets), offsets / float(args.graph_coord_scale), np.linalg.norm(offsets, axis=1, keepdims=True) / float(args.graph_coord_scale)], axis=1).astype(np.float32)
        geom_dim = geometry_all.shape[-1]
        sem_dim = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 + 3 + 1
        out = np.zeros((n, max_nodes, geom_dim + sem_dim), dtype=np.float32)
        mask = np.zeros((n, max_nodes), dtype=bool)
        chunk = int(args.map_query_chunk)
        clip = float(args.map_clearance_clip)
        threshold = float(args.graph_free_clearance)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            world = origin[start:end, None, :] + offsets[None, :, 0:1] * forward[start:end, None, :] + offsets[None, :, 1:2] * left[start:end, None, :] + offsets[None, :, 2:3] * up[start:end, None, :]
            clearance, valid = voxel_map.query_clearance(world)
            occupied, _ = voxel_map.query_occupied_nearest(world)
            for local_i in range(end - start):
                sample_i = start + local_i
                c = clearance[local_i]
                v = valid[local_i].astype(bool)
                occ = occupied[local_i].astype(bool)
                free = (c >= threshold) & v & ~occ
                free_grid = free.reshape(shape)
                reach_grid = _reachable_component(free_grid, offset_grid)
                edge, degree, direction = _adjacency_features(free_grid)
                reach = reach_grid.ravel()
                edge_flat = edge.reshape(total_nodes, 6)
                degree_flat = degree.reshape(total_nodes, 1)
                direction_flat = direction.reshape(total_nodes, 3)
                semantic = np.concatenate([np.clip(c[:, None], 0.0, clip) / clip, np.exp(-np.clip(c[:, None], 0.0, clip) / float(args.boundary_scale)), free[:, None].astype(np.float32), occ[:, None].astype(np.float32), v[:, None].astype(np.float32), reach[:, None].astype(np.float32), degree_flat.astype(np.float32), edge_flat.astype(np.float32), direction_flat.astype(np.float32), np.repeat(heading_valid[sample_i:sample_i + 1, None], total_nodes, axis=0).astype(np.float32)], axis=1)
                eligible = np.where(reach)[0]
                if len(eligible) == 0:
                    eligible = np.where(free)[0]
                if len(eligible) == 0:
                    eligible = np.argsort(-c)[:1]
                score = 2.0 * degree_flat[:, 0] + 0.03 * offsets[:, 0] - 0.01 * np.linalg.norm(offsets, axis=1) + 0.02 * np.clip(c, 0.0, clip)
                selected = eligible[np.argsort(-score[eligible])[:max_nodes]]
                count = len(selected)
                out[sample_i, :count] = np.concatenate([geometry_all[selected], semantic[selected]], axis=1)
                mask[sample_i, :count] = True
        return (out, mask, geom_dim, sem_dim)

    def make_zero_attr_same_graph(raw_tokens, geometry_dim, graph_std):
        tokens = np.asarray(raw_tokens, dtype=np.float32).copy()
        tokens[:, :, geometry_dim:] = 0.0
        return graph_std.transform(tokens)

    def make_null_uniform_graph(n, offsets, graph_max_nodes, geom_dim, sem_dim, graph_std, args):
        """
        与真实地图无关的固定均匀节点集合：不保留真实 sparse topology，也不保留地图语义。
        """
        score = 0.03 * offsets[:, 0] - 0.01 * np.linalg.norm(offsets, axis=1) - 0.002 * np.abs(offsets[:, 1])
        chosen = np.argsort(-score)[:graph_max_nodes]
        geom = np.concatenate([e4.fourier_xyz(offsets[chosen]), offsets[chosen] / float(args.graph_coord_scale), np.linalg.norm(offsets[chosen], axis=1, keepdims=True) / float(args.graph_coord_scale)], axis=1).astype(np.float32)
        raw = np.zeros((n, graph_max_nodes, geom_dim + sem_dim), dtype=np.float32)
        raw[:, :len(chosen), :geom_dim] = geom[None, :, :]
        mask = np.zeros((n, graph_max_nodes), dtype=bool)
        mask[:, :len(chosen)] = True
        return (graph_std.transform(raw), mask)

    class CDataset(Dataset):

        def __init__(self, arrays, graph_tokens, graph_mask):
            self.X = torch.from_numpy(arrays['X'].astype(np.float32))
            self.raw_pi = torch.from_numpy(arrays['raw_pi'].astype(np.float32))
            self.candidates = torch.from_numpy(arrays['candidates'].astype(np.float32))
            self.query = torch.from_numpy(arrays['query'].astype(np.float32))
            self.dyn = torch.from_numpy(arrays['dyn'].astype(np.float32))
            self.basis = torch.from_numpy(arrays['basis'].astype(np.float32))
            self.graph = torch.from_numpy(graph_tokens.astype(np.float32))
            self.graph_mask = torch.from_numpy(graph_mask.astype(bool))
            self.y = torch.from_numpy(arrays['y'].astype(np.float32))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return (self.X[idx], self.raw_pi[idx], self.candidates[idx], self.query[idx], self.dyn[idx], self.basis[idx], self.graph[idx], self.graph_mask[idx], self.y[idx])

    class GlobalTopologyConditionedGenerator(nn.Module):

        def __init__(self, seq_dim, query_dim, dyn_dim, graph_dim, K, args):
            super().__init__()
            self.K = int(K)
            self.max_residual = float(args.c_max_candidate_residual)
            self.max_logit_correction = float(args.c_max_logit_correction)
            d = int(args.c_d_model)
            h = int(args.c_hidden_dim)
            self.history_gru = nn.GRU(seq_dim, h, batch_first=True, bidirectional=True)
            self.history_to_query = nn.Linear(2 * h, d)
            self.query_encoder = nn.Sequential(nn.Linear(query_dim, d), nn.GELU(), nn.LayerNorm(d))
            self.dyn_encoder = nn.Sequential(nn.Linear(dyn_dim, d), nn.GELU(), nn.LayerNorm(d))
            self.graph_input = nn.Sequential(nn.Linear(graph_dim, d), nn.GELU(), nn.LayerNorm(d))
            enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=args.c_attn_heads, dim_feedforward=2 * d, dropout=args.c_dropout, activation='gelu', batch_first=True, norm_first=True)
            self.graph_encoder = nn.TransformerEncoder(enc_layer, num_layers=args.c_graph_layers)
            self.mode_queries = nn.Parameter(torch.randn(1, K, d) * 0.02)
            self.mode_cross_attn = nn.MultiheadAttention(d, args.c_attn_heads, dropout=args.c_dropout, batch_first=True)
            self.mode_norm = nn.LayerNorm(d)
            self.dropout = nn.Dropout(args.c_dropout)
            self.residual_head = nn.Sequential(nn.Linear(2 * h + 2 * d, h), nn.GELU(), nn.Dropout(args.c_dropout), nn.Linear(h, 3))
            self.scorer = nn.Sequential(nn.Linear(2 * h + 3 * d, h), nn.GELU(), nn.Dropout(args.c_dropout), nn.Linear(h, 1))

        def forward(self, X, raw_pi, candidates, query, dyn, basis, graph, graph_mask):
            b, k, horizon, _ = query.shape
            _, hs = self.history_gru(X)
            hist = torch.cat([hs[-2], hs[-1]], dim=-1)
            hist_mode = hist[:, None, :].expand(-1, k, -1)
            hist_step = hist[:, None, None, :].expand(-1, k, horizon, -1)
            q = self.query_encoder(query)
            q_pool = q.mean(dim=2)
            dyn_emb = self.dyn_encoder(dyn)
            memory = self.graph_input(graph)
            padding = ~graph_mask
            memory = self.graph_encoder(memory, src_key_padding_mask=padding)
            mode_q = q_pool + self.mode_queries[:, :k, :] + self.history_to_query(hist)[:, None, :]
            mode_ctx, _ = self.mode_cross_attn(mode_q, memory, memory, key_padding_mask=padding, need_weights=False)
            mode_ctx = self.mode_norm(mode_q + self.dropout(mode_ctx))
            ctx_step = mode_ctx[:, :, None, :].expand(-1, -1, horizon, -1)
            residual_input = torch.cat([hist_step, q, ctx_step], dim=-1)
            residual_local = self.max_residual * torch.tanh(self.residual_head(residual_input))
            t_scale = torch.linspace(1.0 / horizon, 1.0, horizon, device=X.device)[None, None, :, None]
            residual_local = residual_local * t_scale
            forward, left, up = (basis[:, 0], basis[:, 1], basis[:, 2])
            residual_world = e4.local_vector_to_world(residual_local, forward, left, up)
            refined = candidates + residual_world
            score_input = torch.cat([hist_mode, q_pool, dyn_emb, mode_ctx], dim=-1)
            delta_logit = self.max_logit_correction * torch.tanh(self.scorer(score_input).squeeze(-1))
            pi = torch.softmax(torch.log(torch.clamp(raw_pi, min=1e-08)) + delta_logit, dim=1)
            pred = torch.sum(pi[:, :, None, None] * refined, dim=1)
            return (pred, pi, refined, residual_local)

    def candidate_errors_torch(candidates, y, fde_weight):
        dist = torch.linalg.norm(candidates - y[:, None, :, :], dim=-1)
        return dist.mean(dim=2) + float(fde_weight) * dist[:, :, -1]

    def c_loss(pred, pi, refined, residual, raw_pi, y, args, rank_lambda, gate_lambda, residual_lambda, bok_lambda):
        forecast = dk.forecast_loss(pred, y, args)
        err = candidate_errors_torch(refined, y, args.rank_target_fde_weight)
        target = torch.softmax(-err.detach() / max(float(args.rank_target_temperature), 1e-06), dim=1)
        rank = torch.mean(torch.sum(target * (torch.log(torch.clamp(target, min=1e-08)) - torch.log(torch.clamp(pi, min=1e-08))), dim=1))
        gate_anchor = torch.mean((pi - raw_pi) ** 2)
        residual_reg = torch.mean(residual ** 2)
        bestofk = torch.mean(torch.min(err, dim=1).values)
        total = forecast + rank_lambda * rank + gate_lambda * gate_anchor + residual_lambda * residual_reg + bok_lambda * bestofk
        return (total, forecast, rank, gate_anchor, residual_reg, bestofk)

    def infer_c(model, arrays, graph_tokens, graph_mask, batch_size, device):
        model.eval()
        preds, pis, refined, residuals = ([], [], [], [])
        with torch.no_grad():
            for start in range(0, len(arrays['X']), batch_size):
                sl = slice(start, min(len(arrays['X']), start + batch_size))
                inputs = [torch.from_numpy(arr[sl]).float().to(device) for arr in [arrays['X'], arrays['raw_pi'], arrays['candidates'], arrays['query'], arrays['dyn'], arrays['basis'], graph_tokens]]
                mask = torch.from_numpy(graph_mask[sl]).bool().to(device)
                pred, pi, cand, residual = model(*inputs, mask)
                preds.append(pred.cpu().numpy())
                pis.append(pi.cpu().numpy())
                refined.append(cand.cpu().numpy())
                residuals.append(residual.cpu().numpy())
        return (np.concatenate(preds), np.concatenate(pis), np.concatenate(refined), np.concatenate(residuals))

    def train_c(arrays, graph_tokens, graph_masks, val_data, val_global, args, device, rank_lambda, gate_lambda, residual_lambda, bok_lambda):
        tr, va = (arrays['train'], arrays['val'])
        loader = DataLoader(CDataset(tr, graph_tokens['train'], graph_masks['train']), batch_size=args.c_batch_size, shuffle=True, pin_memory=device.type == 'cuda')
        model = GlobalTopologyConditionedGenerator(tr['X'].shape[-1], tr['query'].shape[-1], tr['dyn'].shape[-1], graph_tokens['train'].shape[-1], args.K, args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.c_lr, weight_decay=args.c_weight_decay)
        best_state, best_score, best_epoch, bad = (None, None, -1, 0)
        records = []
        for epoch in range(1, args.c_epochs + 1):
            model.train()
            sums = {k: 0.0 for k in ['loss', 'forecast', 'rank', 'gate', 'residual', 'bestofk']}
            count = 0
            for batch in loader:
                xb, pib, cb, qb, db, bb, gb, mb, yb = [x.to(device) for x in batch]
                pred, pi, refined, residual = model(xb, pib, cb, qb, db, bb, gb, mb)
                vals = c_loss(pred, pi, refined, residual, pib, yb, args, rank_lambda, gate_lambda, residual_lambda, bok_lambda)
                optimizer.zero_grad(set_to_none=True)
                vals[0].backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                bn = len(xb)
                count += bn
                for key, value in zip(['loss', 'forecast', 'rank', 'gate', 'residual', 'bestofk'], vals):
                    sums[key] += float(value.item()) * bn
            vpred, vpi, vcand, vres = infer_c(model, va, graph_tokens['val'], graph_masks['val'], args.c_batch_size, device)
            metrics = base.compute_position_metrics(va['y'], vpred)
            score = metric_score(metrics, args)
            row = {'rank_lambda': rank_lambda, 'gate_lambda': gate_lambda, 'residual_lambda': residual_lambda, 'bestofk_lambda': bok_lambda, 'epoch': epoch, **{f'train_{k}': sums[k] / max(count, 1) for k in sums}, 'val_raw_score': score, 'val_gate_l1_change': float(np.abs(vpi - va['raw_pi']).sum(axis=1).mean()), 'val_residual_norm': float(np.linalg.norm(vres, axis=-1).mean()), **{f'val_{k}': v for k, v in metrics.items()}}
            records.append(row)
            if best_score is None or score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                bad = 0
            else:
                bad += 1
            if epoch == 1 or epoch % args.log_every == 0 or bad == 0:
                print(f"[Stage-C rank={rank_lambda:g} gate={gate_lambda:g} res={residual_lambda:g} bok={bok_lambda:g}] ep={epoch:03d} score={score:.6f} ADE={metrics['ade']:.6f} FDE={metrics['fde']:.6f} resNorm={row['val_residual_norm']:.4f}")
            if bad >= args.c_patience:
                break
        model.load_state_dict(best_state)
        vpred, vpi, vcand, vres = infer_c(model, va, graph_tokens['val'], graph_masks['val'], args.c_batch_size, device)
        safety, sweep = dk.tune_safety(val_data, val_global, vpred, args)
        vsafe, eff, _ = dk.apply_safety(val_global, vpred, safety['alpha'], safety['safety_tau'])
        smet = base.compute_position_metrics(va['y'], vsafe)
        info = {'rank_lambda': float(rank_lambda), 'gate_lambda': float(gate_lambda), 'residual_lambda': float(residual_lambda), 'bestofk_lambda': float(bok_lambda), 'best_epoch': int(best_epoch), 'val_raw_score': float(best_score), 'val_safety_score': float(metric_score(smet, args)), 'safety_alpha': float(safety['alpha']), 'safety_tau': float(safety['safety_tau']), 'mean_effective_safety_alpha_val': float(eff.mean()), **candidate_diagnostics(vpi, vcand, va['y'], vres), **{f'val_safe_{k}': float(v) for k, v in smet.items()}}
        return (model, pd.DataFrame(records), info, safety, sweep)

    def replace_semantics_with_permutation(raw_tokens, geom_dim, permutation, standardizer):
        out = np.asarray(raw_tokens, dtype=np.float32).copy()
        out[..., geom_dim:] = raw_tokens[permutation, ..., geom_dim:]
        return standardizer.transform(out)

    def build_local_pack(data, c_pred, c_pi, c_candidates, voxel_map, args, query_std=None, dyn_std=None, corridor_std=None, intention_std=None, fit=False):
        """
        Stage AB 的关键：corridor tokens 围绕 Stage-C corrected candidates 查询，
        而不是围绕原始 Koopman candidates 查询。
        """
        query_raw, dyn_raw, basis = e4.build_candidate_queries(data, c_pred, c_candidates, c_pi)
        corridor_raw, corridor_geom_dim, corridor_sem_dim = e4.build_corridor_tokens(data, c_candidates, voxel_map, args)
        intention_raw, intention_geom_dim, intention_sem_dim = e4.build_motion_query_tokens(data, voxel_map, args)
        if fit:
            query_std = MaskedStandardizer.fit(query_raw)
            dyn_std = MaskedStandardizer.fit(dyn_raw)
            corridor_std = MaskedStandardizer.fit(corridor_raw)
            intention_std = MaskedStandardizer.fit(intention_raw)
        arrays = {'X': None, 'raw_pi': c_pi.astype(np.float32), 'candidates': c_candidates.astype(np.float32), 'query': query_std.transform(query_raw), 'dyn': dyn_std.transform(dyn_raw), 'basis': basis.astype(np.float32), 'y': data.future_global[:, :, :3].astype(np.float32)}
        tokens = {'corridor': corridor_std.transform(corridor_raw), 'intention': intention_std.transform(intention_raw)}
        raw = {'corridor': corridor_raw, 'intention': intention_raw, 'corridor_geom_dim': corridor_geom_dim, 'intention_geom_dim': intention_geom_dim}
        standards = {'query': query_std, 'dyn': dyn_std, 'corridor': corridor_std, 'intention': intention_std}
        return (arrays, tokens, raw, standards)

    def set_X(local_arrays, X):
        local_arrays['X'] = X.astype(np.float32)
        return local_arrays

    def train_ab(arrays, tokens, val_data, val_global, args, device, prefix):
        histories, selection, sweeps = ([], [], [])
        best = None
        for rank_lambda in parse_float_list(args.ab_rank_lambda_grid):
            for gate_lambda in parse_float_list(args.ab_gate_anchor_lambda_grid):
                for residual_lambda in parse_float_list(args.ab_residual_anchor_lambda_grid):
                    for bok_lambda in parse_float_list(args.ab_bestofk_lambda_grid):
                        set_seed(args.seed + int(rank_lambda * 1000) + int(gate_lambda * 10000) + int(residual_lambda * 100000) + len(prefix))
                        model, history, info, safety, sweep = e4.train_one('AB', arrays, tokens, {'val_data': val_data, 'val_global': val_global}, args, device, rank_lambda, gate_lambda, residual_lambda, bok_lambda)
                        history['experiment'] = prefix
                        histories.append(history)
                        info['experiment'] = prefix
                        selection.append(info)
                        sweep['experiment'] = prefix
                        sweeps.append(sweep)
                        if best is None or info['val_safety_score'] < best['info']['val_safety_score']:
                            best = {'model': copy.deepcopy(model).cpu(), 'info': info, 'safety': safety}
        return (best, pd.concat(histories, ignore_index=True), pd.DataFrame(selection), pd.concat(sweeps, ignore_index=True))

    def subset_metrics(data, predictions, voxel_map, base_candidates, args):
        gt = data.future_global[:, :, :3]
        gt_clearance, _ = voxel_map.query_clearance(gt)
        gt_min = gt_clearance.min(axis=1)
        base_feat = e1.candidate_environment_features(voxel_map, base_candidates, 2.0)
        gap = base_feat['min_clearance'].max(axis=1) - base_feat['min_clearance'].min(axis=1)
        masks = {'all': np.ones(len(gt), dtype=bool), f'gt_near_obstacle_le_{args.report_near_clearance:g}m': gt_min <= args.report_near_clearance, f'gt_open_gt_{args.report_near_clearance:g}m': gt_min > args.report_near_clearance, f'base_candidate_clearance_gap_ge_{args.report_candidate_gap:g}m': gap >= args.report_candidate_gap}
        rows = []
        for subset, mask in masks.items():
            if mask.sum() == 0:
                continue
            for name, pred in predictions.items():
                rows.append({'subset': subset, 'num_samples': int(mask.sum()), **method_row(subset_view(data, mask), pred[mask], name, voxel_map, args)})
        return pd.DataFrame(rows)

    def horizon_method_row(data, pred, method, voxel_map, args, horizon, extra=None):
        """Evaluate a prediction prefix at a given report horizon without changing training."""
        horizon = int(horizon)
        if horizon < 1 or horizon > data.future_global.shape[1]:
            raise ValueError(f'report horizon {horizon} is outside prediction horizon {data.future_global.shape[1]}')
        view = DataView()
        view.future_global = data.future_global[:, :horizon, :3]
        return method_row(view, pred[:, :horizon, :3], method, voxel_map, args, extra=extra)

    def make_multi_horizon_table(data, prediction_dict, voxel_map, args, horizons):
        """Create rows for horizon=1/3/5/10 style reporting from one H=10 prediction."""
        rows = []
        for h in horizons:
            if h > data.future_global.shape[1]:
                continue
            for method, pred in prediction_dict.items():
                rows.append(horizon_method_row(data, pred, method, voxel_map, args, h))
        return pd.DataFrame(rows)

    def parse_report_horizons(text, max_horizon):
        horizons = []
        for x in str(text).split(','):
            x = x.strip()
            if not x:
                continue
            h = int(x)
            if 1 <= h <= max_horizon and h not in horizons:
                horizons.append(h)
        if max_horizon not in horizons:
            horizons.append(max_horizon)
        return sorted(horizons)

    def _action_onehot_indices(data, modes):
        """Return current action indices. Unknown/global/transition labels are -1, not forced to class 0."""
        mode_to_idx = {m: i for i, m in enumerate(modes)}
        idx = np.full(len(data.behavior_key), -1, dtype=np.int64)
        for i, b in enumerate(data.behavior_key):
            idx[i] = mode_to_idx.get(str(b), -1)
        return idx

    def _future_behavior_labels_from_chains(chains, input_len, horizon, stride, modes=None, target='dominant'):
        """
        Build the supervised target label for the prediction horizon.

        This is intentionally separate from data.behavior_key, which denotes the last
        observed/current behavior in the history window.  For an MoE gate that is used
        in forecasting, the supervised target should be a future behavior label, e.g.
        the dominant behavior over the H-step prediction interval.
        """
        labels = []
        for df in chains:
            max_start = len(df) - int(input_len) - int(horizon)
            if max_start < 0:
                continue
            for s0 in range(0, max_start + 1, int(stride)):
                fut = df.iloc[s0 + int(input_len):s0 + int(input_len) + int(horizon)]
                raw = [str(bk.row_behavior_key(row)) for _, row in fut.iterrows()]
                valid = [x for x in raw if x != 'global']
                if target == 'first':
                    lab = valid[0] if valid else raw[0] if raw else 'global'
                elif target == 'last':
                    lab = valid[-1] if valid else raw[-1] if raw else 'global'
                elif target == 'dominant':
                    seq = valid if valid else raw
                    lab = str(pd.Series(seq).mode().iloc[0]) if seq else 'global'
                else:
                    raise ValueError(f'Unknown future behavior target: {target}')
                if modes is not None and lab not in modes:
                    lab = 'global'
                labels.append(lab)
        return labels

    def _clone_data_with_behavior_key(data, labels):
        if len(labels) != len(data.behavior_key):
            raise RuntimeError(f'future/current behavior label length mismatch: labels={len(labels)}, samples={len(data.behavior_key)}')
        out = copy.copy(data)
        out.behavior_key = list(labels)
        return out

    def _subset_split_data(data, mask):
        """Subset a SplitData-like object while preserving local/global frames and metadata."""
        mask = np.asarray(mask, dtype=bool)
        out = copy.copy(data)
        for name in ['hist_global', 'future_global', 'hist_local', 'future_local']:
            if hasattr(data, name):
                setattr(out, name, getattr(data, name)[mask])
        if hasattr(data, 'frames'):
            out.frames = bk.LocalFrameBatch(
                origins=data.frames.origins[mask],
                c=data.frames.c[mask],
                s=data.frames.s[mask],
            )
        for name in ['behavior_key', 'dominant_behavior', 'scene_type', 'chain_file']:
            if hasattr(data, name):
                vals = list(getattr(data, name))
                setattr(out, name, [v for v, keep in zip(vals, mask) if keep])
        if hasattr(data, 'horizon'):
            out.horizon = data.horizon
        return out

    def _estimate_action_transition_excluding_global(chains, modes, smooth=1.0):
        """Estimate transitions only among real action modes; global/transition frames are skipped."""
        m2i = {m: i for i, m in enumerate(modes)}
        K = len(modes)
        counts = np.ones((K, K), dtype=np.float64) * float(smooth)
        for df in chains:
            labs = []
            for _, row in df.iterrows():
                b = str(bk.row_behavior_key(row))
                labs.append(b if b in m2i else None)
            prev = None
            for cur in labs:
                if cur is not None and prev is not None:
                    counts[m2i[prev], m2i[cur]] += 1.0
                if cur is not None:
                    prev = cur
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

    def _labels_to_indices(labels, modes):
        mode_to_idx = {m: i for i, m in enumerate(modes)}
        missing = sorted({str(x) for x in labels if str(x) not in mode_to_idx})
        if missing:
            raise ValueError(f'labels contain values outside real action modes: {missing[:10]}')
        return np.asarray([mode_to_idx[str(x)] for x in labels], dtype=np.int64)

    class _ActionMoEGate(nn.Module):
        """Small supervised MoE gate: history features + current behavior prior -> future behavior probabilities."""

        def __init__(self, in_dim, hidden_dim, num_modes, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_modes))

        def forward(self, x):
            return self.net(x)

    def _action_gate_features(data, norm, fstd_xy, fstd_z, modes, args):
        """Features for the future-behavior MoE gate. Does not alter LF branch."""
        Fxy = fstd_xy.transform(v5.make_axis_features(data, norm, 'xy', args.dt))
        Fz = fstd_z.transform(v5.make_axis_features(data, norm, 'z', args.dt))
        parts = [Fxy.astype(np.float32), Fz.astype(np.float32)]
        if bool(args.action_gate_use_current_behavior):
            idx = _action_onehot_indices(data, modes)
            oh = np.zeros((len(idx), len(modes)), dtype=np.float32)
            valid = idx >= 0
            if valid.any():
                oh[np.arange(len(idx))[valid], idx[valid]] = 1.0
            # global/transition/unknown current behavior is encoded as an all-zero prior.
            parts.append(oh)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _classification_summary(y_true, prob, modes):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.argmax(prob, axis=1).astype(np.int64)
        K = len(modes)
        cm = np.zeros((K, K), dtype=np.int64)
        for a, b in zip(y_true, y_pred):
            if 0 <= a < K and 0 <= b < K:
                cm[a, b] += 1
        rows = []
        f1s = []
        for i, m in enumerate(modes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            f1s.append(f1)
            rows.append({'behavior': m, 'precision': prec, 'recall': rec, 'f1': f1, 'support': int(cm[i, :].sum())})
        metrics = {'accuracy': float((y_pred == y_true).mean()) if len(y_true) else 0.0, 'macro_f1': float(np.mean(f1s)) if f1s else 0.0, 'num_samples': int(len(y_true))}
        return (metrics, pd.DataFrame(rows), pd.DataFrame(cm, index=modes, columns=modes))

    def _train_action_moe_gate(Xtr, ytr, Xva, yva, modes, args, out_dir):
        device = torch.device('cuda' if str(args.device) != 'cpu' and torch.cuda.is_available() else 'cpu')
        model = _ActionMoEGate(Xtr.shape[1], args.action_gate_hidden_dim, len(modes), dropout=args.action_gate_dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.action_gate_lr, weight_decay=args.action_gate_weight_decay)
        if bool(args.action_gate_class_weight):
            counts = np.bincount(ytr, minlength=len(modes)).astype(np.float32)
            nz = counts[counts > 0]
            base_w = float(np.median(nz)) if len(nz) else 1.0
            weights = base_w / np.maximum(counts, 1.0)
            weights = weights / max(weights.mean(), 1e-06)
            ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        else:
            ce = nn.CrossEntropyLoss()
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
        yva_t = torch.tensor(yva, dtype=torch.long, device=device)
        best_loss, best_state, bad = (None, None, 0)
        hist = []
        rng = np.random.default_rng(args.seed + 20260604)
        for ep in range(1, args.action_gate_epochs + 1):
            model.train()
            order = rng.permutation(len(Xtr))
            losses = []
            for start in range(0, len(order), args.action_gate_batch_size):
                idx = order[start:start + args.action_gate_batch_size]
                xb = Xtr_t[idx].to(device)
                yb = ytr_t[idx].to(device)
                loss = ce(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                logits = model(Xva_t)
                val_loss = float(ce(logits, yva_t).detach().cpu())
                pred = torch.argmax(logits, dim=1)
                val_acc = float((pred == yva_t).float().mean().detach().cpu())
            row = {'epoch': ep, 'train_ce': float(np.mean(losses)), 'val_ce': val_loss, 'val_acc': val_acc}
            hist.append(row)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if ep == 1 or ep % args.log_every == 0 or bad == 0:
                print(f"[Action-MoE gate] ep={ep:03d} trainCE={row['train_ce']:.4f} valCE={val_loss:.4f} valAcc={val_acc:.4f}")
            if bad >= args.action_gate_patience:
                break
        model.load_state_dict(best_state)
        model.to(device)
        pd.DataFrame(hist).to_csv(Path(out_dir) / 'action_moe_gate_history_val.csv', index=False)
        torch.save({'model_state_dict': model.state_dict(), 'modes': modes, 'input_dim': int(Xtr.shape[1]), 'best_val_ce': best_loss, 'args': {k: getattr(args, k) for k in ['action_gate_hidden_dim', 'action_gate_dropout', 'action_gate_lr', 'action_gate_weight_decay'] if hasattr(args, k)}}, Path(out_dir) / 'action_moe_gate.pt')
        return (model, device)

    def _predict_action_moe_gate(model, X, device, batch_size=2048):
        model.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
                outs.append(torch.softmax(model(xb), dim=1).detach().cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    def action_mode_probabilities(data, modes, T_behavior=None, steps=1, probability_mode='current_onehot', gate_probs=None, future_labels=None):
        """
        Return action/mode probabilities used as the initial MoE candidate weights.

        - current_onehot: hard prior from the last observed behavior label. It is a prior, not a future action prediction.
        - hmm: propagates current behavior through a train-estimated transition matrix.
        - moe_gate: supervised MoE gate prediction for future behavior distribution.
        - future_onehot: diagnostic/oracle-like future-label one-hot; not for deployable comparison.
        """
        if probability_mode == 'moe_gate':
            if gate_probs is None:
                raise ValueError("gate_probs is required when probability_mode='moe_gate'.")
            pi = np.asarray(gate_probs, dtype=np.float64)
        elif probability_mode == 'future_onehot':
            if future_labels is None:
                raise ValueError("future_labels is required when probability_mode='future_onehot'.")
            idx = _labels_to_indices(future_labels, modes)
            pi = np.zeros((len(idx), len(modes)), dtype=np.float64)
            pi[np.arange(len(idx)), idx] = 1.0
        else:
            idx = _action_onehot_indices(data, modes)
            pi = np.ones((len(idx), len(modes)), dtype=np.float64) / max(len(modes), 1)
            valid = idx >= 0
            if valid.any():
                pi[valid, :] = 0.0
                pi[np.arange(len(idx))[valid], idx[valid]] = 1.0
            # Unknown/global/transition current behavior falls back to uniform over real action modes.
            if probability_mode == 'hmm':
                if T_behavior is None:
                    raise ValueError("T_behavior is required when probability_mode='hmm'.")
                pi = pi @ bk.transition_power(T_behavior, steps)
            elif probability_mode == 'current_onehot':
                pass
            else:
                raise ValueError(f'Unknown action_probability_mode={probability_mode}')
        pi = pi / np.maximum(pi.sum(axis=1, keepdims=True), 1e-12)
        return pi.astype(np.float32)

    def save_action_probabilities(path, data, modes, pi, future_labels=None):
        rows = {'sample_idx': np.arange(len(pi), dtype=int), 'behavior_key_observed': [str(x) for x in getattr(data, 'behavior_key', [''] * len(pi))], 'dominant_behavior': [str(x) for x in getattr(data, 'dominant_behavior', [''] * len(pi))], 'future_behavior_target': [str(x) for x in (future_labels if future_labels is not None else [''] * len(pi))], 'pred_action': [modes[int(i)] for i in np.argmax(pi, axis=1)]}
        for j, m in enumerate(modes):
            rows[f'p_{m}'] = pi[:, j]
        pd.DataFrame(rows).to_csv(path, index=False)

    def fit_action_supervised_frozen_candidates(train_chains, val_chains, test_chains, tr, va, te, gtr, gva, gte, args, out_dir):
        """
        Build action-supervised Koopman candidates by reusing the original V5
        axis-feature-decoupled behavior Koopman implementation.

        Correction relative to the hard-prior prototype:
            - behavior experts are fitted using future-horizon behavior targets
              (default: dominant future behavior over H), not merely the last history label;
            - action_probability_mode='moe_gate' trains a supervised MoE gate to output
              P(future_behavior | history, current_behavior_prior);
            - label_free/unsupervised branch is untouched.
        """
        print('\n=== Building action-supervised future-behavior Koopman-MoE candidates (MapKoop-Act-AD-MoE) ===')
        norm = bk.fit_state_normalizer(tr)
        Fxy_tr = v5.make_axis_features(tr, norm, 'xy', args.dt)
        Fz_tr = v5.make_axis_features(tr, norm, 'z', args.dt)
        fstd_xy = bk.FeatureStandardizer.fit(Fxy_tr)
        fstd_z = bk.FeatureStandardizer.fit(Fz_tr)
        future_tr_all = _future_behavior_labels_from_chains(train_chains, args.input_len, args.horizon, args.stride, modes=None, target=args.future_behavior_target)
        counts = pd.Series(future_tr_all).value_counts()
        eligible = [b for b, c in counts.items() if b != 'global' and c >= args.min_behavior_samples]
        modes = sorted(eligible)
        if len(modes) < 1:
            raise RuntimeError(f'Action-supervised mode requested, but no eligible non-global future behavior labels were found. Counts: {counts.to_dict()}')
        future_tr = _future_behavior_labels_from_chains(train_chains, args.input_len, args.horizon, args.stride, modes=modes, target=args.future_behavior_target)
        future_va = _future_behavior_labels_from_chains(val_chains, args.input_len, args.horizon, args.stride, modes=modes, target=args.future_behavior_target)
        future_te = _future_behavior_labels_from_chains(test_chains, args.input_len, args.horizon, args.stride, modes=modes, target=args.future_behavior_target)
        valid_tr = np.asarray([x in modes for x in future_tr], dtype=bool)
        valid_va = np.asarray([x in modes for x in future_va], dtype=bool)
        valid_te = np.asarray([x in modes for x in future_te], dtype=bool)
        print('action-supervised real-action modes:', modes)
        print('future behavior counts before non-action masking:\n' + pd.Series(future_tr).value_counts().to_string())
        print(f'non-action/global future windows excluded from action gate/expert fitting: train={int((~valid_tr).sum())}/{len(valid_tr)}, val={int((~valid_va).sum())}/{len(valid_va)}, test={int((~valid_te).sum())}/{len(valid_te)}')
        tr_expert = _clone_data_with_behavior_key(_subset_split_data(tr, valid_tr), [x for x, keep in zip(future_tr, valid_tr) if keep])
        va_expert = _clone_data_with_behavior_key(_subset_split_data(va, valid_va), [x for x, keep in zip(future_va, valid_va) if keep])
        Wxy, Wz, best_global, global_sweep = v5.fit_global(tr, va, norm, fstd_xy, fstd_z, parse_float_list(args.global_ridge_xy_grid), parse_float_list(args.global_ridge_z_grid), args.dt, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight)
        model, best_behavior, behavior_sweep = v5.fit_behavior_model(tr_expert, va_expert, norm, fstd_xy, fstd_z, Wxy, Wz, modes, parse_float_list(args.behavior_ridge_xy_grid), parse_float_list(args.behavior_ridge_z_grid), args.dt, args.min_behavior_samples, args.action_candidate_selection, args.selection_metric, args.combo_ade_weight, args.combo_fde_weight, operator_mode=getattr(args, 'condensed_operator_mode', 'shared_residual'))
        pd.DataFrame(global_sweep).to_csv(Path(out_dir) / 'action_global_koopman_sweep_val.csv', index=False)
        pd.DataFrame(behavior_sweep).to_csv(Path(out_dir) / 'action_behavior_koopman_sweep_val.csv', index=False)
        op_pack = {
            'operator_mode': np.asarray([getattr(args, 'condensed_operator_mode', 'shared_residual')]),
            'modes': np.asarray(modes, dtype=object),
            'Wxy_global': np.asarray(model.W_xy),
            'Wz_global': np.asarray(model.W_z),
        }
        for mode in modes:
            op_pack[f'Wxy_{mode}'] = np.asarray(model.W_xy if model.R_xy.get(mode) is None else model.W_xy + model.R_xy[mode])
            op_pack[f'Wz_{mode}'] = np.asarray(model.W_z if model.R_z.get(mode) is None else model.W_z + model.R_z[mode])
        np.savez_compressed(Path(out_dir) / 'action_condensed_mode_operators.npz', **op_pack)
        T_behavior = _estimate_action_transition_excluding_global(train_chains, modes, smooth=args.action_hmm_smooth)
        pd.DataFrame(T_behavior, index=modes, columns=modes).to_csv(Path(out_dir) / 'action_behavior_transition_matrix.csv')
        gate_model, gate_device = (None, None)
        gate_probs = {}
        y_indices = {
            'train': _labels_to_indices([x for x, keep in zip(future_tr, valid_tr) if keep], modes),
            'val': _labels_to_indices([x for x, keep in zip(future_va, valid_va) if keep], modes),
            'test': _labels_to_indices([x for x, keep in zip(future_te, valid_te) if keep], modes),
        }
        gate_features_all = {'train': _action_gate_features(tr, norm, fstd_xy, fstd_z, modes, args), 'val': _action_gate_features(va, norm, fstd_xy, fstd_z, modes, args), 'test': _action_gate_features(te, norm, fstd_xy, fstd_z, modes, args)}
        gate_features_fit = {'train': gate_features_all['train'][valid_tr], 'val': gate_features_all['val'][valid_va], 'test': gate_features_all['test'][valid_te]}
        if args.action_probability_mode == 'moe_gate':
            gate_model, gate_device = _train_action_moe_gate(gate_features_fit['train'], y_indices['train'], gate_features_fit['val'], y_indices['val'], modes, args, out_dir)
            for split_name in ['train', 'val', 'test']:
                gate_probs[split_name] = _predict_action_moe_gate(gate_model, gate_features_all[split_name], gate_device, batch_size=args.action_gate_eval_batch_size)
            for split_name, labels, mask in [('train', future_tr, valid_tr), ('val', future_va, valid_va), ('test', future_te, valid_te)]:
                prob_eval = gate_probs[split_name][mask]
                metrics, per_class, cm = _classification_summary(y_indices[split_name], prob_eval, modes)
                pd.DataFrame([metrics]).to_csv(Path(out_dir) / f'future_behavior_moe_gate_metrics_{split_name}.csv', index=False)
                per_class.to_csv(Path(out_dir) / f'future_behavior_moe_gate_per_class_{split_name}.csv', index=False)
                cm.to_csv(Path(out_dir) / f'future_behavior_moe_gate_confusion_{split_name}.csv')
                print(f"[Action-MoE gate real-action only] {split_name}: acc={metrics['accuracy']:.4f}, macroF1={metrics['macro_f1']:.4f}, evaluated={int(mask.sum())}/{len(mask)}")
        original_K = int(args.K)
        args.K = len(modes)
        if original_K != args.K:
            print(f'MapKoop-Act: overriding K from {original_K} to number of future-behavior modes {args.K}.')
        future_labels_by_split = {'train': future_tr, 'val': future_va, 'test': future_te}

        def pack_split(data, global_pred, split_name):
            cand_std = model.stack_candidates_pos_std(data).astype(np.float32)
            candidates = np.stack([v5.pos_std_to_global(cand_std[:, k], data, norm)[:, :, :3] for k in range(len(modes))], axis=1).astype(np.float32)
            pi = action_mode_probabilities(data, modes, T_behavior=T_behavior, steps=args.horizon, probability_mode=args.action_probability_mode, gate_probs=gate_probs.get(split_name), future_labels=future_labels_by_split[split_name])
            raw = np.sum(pi[:, :, None, None] * candidates, axis=1).astype(np.float32)
            save_action_probabilities(Path(out_dir) / f'action_probabilities_{split_name}.csv', data, modes, pi, future_labels=future_labels_by_split[split_name])
            np.save(Path(out_dir) / f'action_probabilities_{split_name}.npy', pi.astype(np.float32))
            return {'raw': raw, 'pi': pi.astype(np.float32), 'calibrated': candidates.astype(np.float32)}
        train_pack = pack_split(tr, gtr, 'train')
        val_pack = pack_split(va, gva, 'val')
        test_pack = pack_split(te, gte, 'test')
        val_safe_pack, val_sweep = dk.tune_safety(va, gva, val_pack['raw'], args)
        val_sweep['mode_source'] = 'action_label'
        val_sweep['action_probability_mode'] = args.action_probability_mode
        val_sweep.to_csv(Path(out_dir) / 'action_supervised_safety_sweep_val.csv', index=False)
        val_safe, _, _ = dk.apply_safety(gva, val_pack['raw'], val_safe_pack['alpha'], val_safe_pack['safety_tau'])
        test_safe, _, _ = dk.apply_safety(gte, test_pack['raw'], val_safe_pack['alpha'], val_safe_pack['safety_tau'])
        val_pack['safe'] = val_safe.astype(np.float32)
        test_pack['safe'] = test_safe.astype(np.float32)
        report = {'mode_source': 'action_label', 'expert_label_source': 'future_behavior', 'future_behavior_target': args.future_behavior_target, 'modes': modes, 'future_behavior_counts': pd.Series(future_tr).value_counts().to_dict(), 'best_global': best_global, 'best_behavior': best_behavior, 'action_probability_mode': args.action_probability_mode, 'action_hmm_smooth': args.action_hmm_smooth, 'safety': val_safe_pack}
        (Path(out_dir) / 'action_supervised_candidate_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'mode_source': 'action_label', 'action_model': model, 'action_gate_model': gate_model, 'modes': modes, 'transition_matrix': T_behavior, 'future_behavior_labels': future_labels_by_split, 'best_pack': {'safe': val_safe_pack, 'best_behavior': best_behavior}, 'train': train_pack, 'val': val_pack, 'test': test_pack}

    def _rebuild_split_from_noisy_history(data, sigma_values, sigma_probs, seed, split_name='train'):
        """
        Apply fixed per-window i.i.d. zero-mean Gaussian position noise in physical meters.

        Semantics intentionally match the previous label-free robustness training:
          - perturb historical x/y/z only;
          - keep historical vx/vy/vz clean;
          - keep future global GT clean;
          - rebuild the local frame from the noisy historical state;
          - re-express BOTH noisy history and clean future GT in that rebuilt local frame;
          - keep behavior/action metadata unchanged.

        One sigma is sampled for each history window from the specified categorical mixture.
        """
        sigmas = np.asarray(sigma_values, dtype=np.float64)
        probs = np.asarray(sigma_probs, dtype=np.float64)
        if sigmas.ndim != 1 or probs.ndim != 1 or len(sigmas) != len(probs) or len(sigmas) == 0:
            raise ValueError('noise sigmas/probs must be non-empty 1-D arrays of the same length')
        if np.any(sigmas < 0):
            raise ValueError('position noise sigma must be >= 0')
        psum = float(probs.sum())
        if psum <= 0:
            raise ValueError('noise probabilities must sum to > 0')
        probs = probs / psum

        rng = np.random.default_rng(int(seed))
        out = copy.copy(data)
        hg = np.asarray(data.hist_global, dtype=np.float64).copy()
        fg = np.asarray(data.future_global, dtype=np.float64).copy()  # clean GT
        n, L, _ = hg.shape
        sampled_sigma = rng.choice(sigmas, size=n, replace=True, p=probs)
        eps = rng.normal(0.0, 1.0, size=(n, L, 3)) * sampled_sigma[:, None, None]
        hg[:, :, :3] += eps

        hl = np.empty_like(hg)
        fl = np.empty_like(fg)
        origins = np.empty((n, 3), dtype=np.float64)
        cs = np.empty(n, dtype=np.float64)
        ss = np.empty(n, dtype=np.float64)
        for i in range(n):
            origin, c, st = bk.make_one_frame(hg[i, -1], hg[i, -2] if L > 1 else None)
            origins[i] = origin
            cs[i] = c
            ss[i] = st
            hl[i] = bk.global_to_local_one(hg[i], origin, c, st)
            fl[i] = bk.global_to_local_one(fg[i], origin, c, st)

        out.hist_global = hg
        out.future_global = fg
        out.hist_local = hl
        out.future_local = fl
        out.frames = bk.LocalFrameBatch(origins=origins, c=cs, s=ss)
        out.noise_sigma_per_sample = sampled_sigma

        unique, counts = np.unique(sampled_sigma, return_counts=True)
        hist = {float(k): int(v) for k, v in zip(unique, counts)}
        print(f'[POSITION-NOISE-AUG] split={split_name} seed={seed} N={n} sigma_counts={hist}')
        return out

    def _pack_action_koopman(model):
        return {
            'format_version': 1,
            'H': int(model.H),
            'dt': float(model.dt),
            'modes': list(model.modes),
            'norm': e2._pack_standardizer(model.norm),
            'fstd_xy': e2._pack_standardizer(model.fstd_xy),
            'fstd_z': e2._pack_standardizer(model.fstd_z),
            'W_xy': np.asarray(model.W_xy).copy(),
            'W_z': np.asarray(model.W_z).copy(),
            'R_xy': {str(k): (None if v is None else np.asarray(v).copy()) for k, v in model.R_xy.items()},
            'R_z': {str(k): (None if v is None else np.asarray(v).copy()) for k, v in model.R_z.items()},
        }

    def main():
        p = argparse.ArgumentParser(description='TopoKoop formal no-loss refactor runner.')
        p.add_argument('--data_root', required=True)
        p.add_argument('--binvox_file', required=True)
        p.add_argument('--out_dir', required=True)
        p.add_argument('--input_len', type=int, default=20)
        p.add_argument('--horizon', type=int, default=10)
        p.add_argument('--report_horizons', default='1,3,5,10', help='Comma-separated test horizons to report from the same trained prediction, e.g. 1,3,5,10.')
        p.add_argument('--stride', type=int, default=5)
        p.add_argument('--dt', type=float, default=0.1)
        p.add_argument('--K', type=int, default=12)
        p.add_argument('--mode_source', default='label_free', choices=['label_free', 'action_label', 'auto'], help='label_free keeps the original unsupervised D-KMoE candidate generator; action_label uses supervised behavior/action labels to build Koopman candidates; auto uses action_label if behavior labels are present.')
        p.add_argument('--condensed_operator_mode', default='independent', choices=['shared_residual', 'independent'], help='Mode-dependent direct multi-step/condensed operator parameterization. independent fits a complete finite-horizon operator per mode; shared_residual reproduces the original global-plus-residual fitting.')
        p.add_argument('--action_probability_mode', default='moe_gate', choices=['moe_gate', 'current_onehot', 'hmm', 'future_onehot'], help='Action-branch candidate weights: moe_gate predicts future behavior probabilities; current_onehot uses the last observed behavior only as a hard prior; hmm propagates the current label; future_onehot is a diagnostic future-label upper-bound.')
        p.add_argument('--future_behavior_target', default='dominant', choices=['dominant', 'first', 'last'], help='Future behavior label used to supervise action experts and the MoE gate.')
        p.add_argument('--action_gate_hidden_dim', type=int, default=128)
        p.add_argument('--action_gate_dropout', type=float, default=0.15)
        p.add_argument('--action_gate_epochs', type=int, default=80)
        p.add_argument('--action_gate_patience', type=int, default=12)
        p.add_argument('--action_gate_batch_size', type=int, default=512)
        p.add_argument('--action_gate_eval_batch_size', type=int, default=2048)
        p.add_argument('--action_gate_lr', type=float, default=0.001)
        p.add_argument('--action_gate_weight_decay', type=float, default=0.0001)
        p.add_argument('--action_gate_class_weight', action='store_true', help='Use inverse-frequency weights for future-behavior gate CE loss.')
        p.add_argument('--action_gate_use_current_behavior', action='store_true', default=True, help='Append last-observed behavior one-hot as a prior input to the future-behavior MoE gate.')
        p.add_argument('--action_candidate_selection', default='hard', choices=['hard', 'oracle'], help='Validation selection mode for fitting the action-supervised Koopman residuals. Use hard for deployable results; oracle only for upper-bound diagnostics.')
        p.add_argument('--behavior_ridge_xy_grid', default='1e-3,1e-2,1e-1,1,3')
        p.add_argument('--behavior_ridge_z_grid', default='1e-3,1e-2,1e-1,1,3')
        p.add_argument('--min_behavior_samples', type=int, default=40)
        p.add_argument('--action_hmm_smooth', type=float, default=1.0)
        p.add_argument('--ridge_model', type=float, default=1.0)
        p.add_argument('--ridge_gate', type=float, default=1.0)
        p.add_argument('--min_cluster_samples', type=int, default=80)
        p.add_argument('--alt_iter', type=int, default=12)
        p.add_argument('--kmeans_iter', type=int, default=50)
        p.add_argument('--global_ridge_xy_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        p.add_argument('--global_ridge_z_grid', default='1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
        p.add_argument('--hidden_dim', type=int, default=128)
        p.add_argument('--gru_layers', type=int, default=2)
        p.add_argument('--dropout', type=float, default=0.2)
        p.add_argument('--max_scale_deviation', type=float, default=0.25)
        p.add_argument('--anchor_lambda_grid', default='0,0.01,0.1')
        p.add_argument('--balance_lambda_grid', default='0,0.001')
        p.add_argument('--epochs', type=int, default=80)
        p.add_argument('--patience', type=int, default=12)
        p.add_argument('--batch_size', type=int, default=512)
        p.add_argument('--lr', type=float, default=0.001)
        p.add_argument('--weight_decay', type=float, default=0.0001)
        p.add_argument('--grad_clip', type=float, default=1.0)
        p.add_argument('--log_every', type=int, default=10)
        p.add_argument('--train_ade_weight', type=float, default=0.5)
        p.add_argument('--train_fde_weight', type=float, default=0.25)
        p.add_argument('--selection_metric', default='combo', choices=['pos_rmse', 'ade', 'fde', 'combo'])
        p.add_argument('--combo_ade_weight', type=float, default=0.5)
        p.add_argument('--combo_fde_weight', type=float, default=0.25)
        p.add_argument('--alpha_grid', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
        p.add_argument('--safety_quantile_grid', default='0.4,0.6,0.8,0.95,1.0')
        p.add_argument('--map_expected_xyz', nargs=3, type=int, required=True)
        p.add_argument('--map_center_xyz', nargs=3, type=float, required=True)
        p.add_argument('--map_resolution', type=float, default=1.0)
        p.add_argument('--map_axis_mode', default='swap_yz', choices=['swap_yz', 'current_xyz'])
        p.add_argument('--unsafe_clearance', type=float, default=1.0)
        p.add_argument('--report_near_clearance', type=float, default=5.0)
        p.add_argument('--report_candidate_gap', type=float, default=2.0)
        p.add_argument('--map_clearance_clip', type=float, default=20.0)
        p.add_argument('--boundary_scale', type=float, default=5.0)
        p.add_argument('--map_query_chunk', type=int, default=256)
        p.add_argument('--graph_forward_range', default='-8,64,8')
        p.add_argument('--graph_left_range', default='-40,40,8')
        p.add_argument('--graph_up_range', default='-16,16,8')
        p.add_argument('--graph_free_clearance', type=float, default=2.0)
        p.add_argument('--graph_max_nodes', type=int, default=96)
        p.add_argument('--graph_coord_scale', type=float, default=70.0)
        p.add_argument('--c_d_model', type=int, default=64)
        p.add_argument('--c_hidden_dim', type=int, default=96)
        p.add_argument('--c_attn_heads', type=int, default=4)
        p.add_argument('--c_graph_layers', type=int, default=2)
        p.add_argument('--c_dropout', type=float, default=0.15)
        p.add_argument('--c_max_candidate_residual', type=float, default=2.0)
        p.add_argument('--c_max_logit_correction', type=float, default=2.0)
        p.add_argument('--rank_target_temperature', type=float, default=0.05)
        p.add_argument('--rank_target_fde_weight', type=float, default=0.25)
        p.add_argument('--c_rank_lambda_grid', default='0.1,0.5', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--c_gate_lambda_grid', default='0.01,0.1', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--c_residual_lambda_grid', default='0.05,0.1', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--c_bestofk_lambda_grid', default='0.05,0.1', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--env_hparam_mode', default='fixed', choices=['fixed', 'grid'],
                       help='fixed: formal repeated-run protocol, train one Stage-C and one Stage-AB setting; grid: development-only hyperparameter sweep.')
        p.add_argument('--c_rank_lambda', type=float, default=0.1, help='Formal fixed Stage-C rank loss weight.')
        p.add_argument('--c_gate_lambda', type=float, default=0.01, help='Formal fixed Stage-C gate-anchor loss weight.')
        p.add_argument('--c_residual_lambda', type=float, default=0.05, help='Formal fixed Stage-C residual-anchor loss weight.')
        p.add_argument('--c_bestofk_lambda', type=float, default=0.05, help='Formal fixed Stage-C best-of-K loss weight.')
        p.add_argument('--c_epochs', type=int, default=70)
        p.add_argument('--c_patience', type=int, default=10)
        p.add_argument('--c_batch_size', type=int, default=96)
        p.add_argument('--c_lr', type=float, default=0.001)
        p.add_argument('--c_weight_decay', type=float, default=0.0001)
        p.add_argument('--corridor_anchor_indices', default='2,5,9')
        p.add_argument('--corridor_lateral_offsets', default='-8,0,8')
        p.add_argument('--corridor_up_offsets', default='-6,0,6')
        p.add_argument('--corridor_token_topk', type=int, default=0)
        p.add_argument('--motion_forward_offsets', default='8,16,24')
        p.add_argument('--motion_left_offsets', default='-12,-6,0,6,12')
        p.add_argument('--motion_up_offsets', default='-6,0,6')
        p.add_argument('--motion_ray_fractions', default='0.25,0.5,0.75,1')
        p.add_argument('--d_model', type=int, default=64)
        p.add_argument('--e4_hidden_dim', type=int, default=96)
        p.add_argument('--attn_heads', type=int, default=4)
        p.add_argument('--e4_dropout', type=float, default=0.15)
        p.add_argument('--max_logit_correction', type=float, default=2.0)
        p.add_argument('--max_candidate_residual', type=float, default=2.0)
        p.add_argument('--ab_rank_lambda_grid', default='0.1,0.5', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--ab_gate_anchor_lambda_grid', default='0.01,0.1', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--ab_residual_anchor_lambda_grid', default='0.05,0.1', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--ab_bestofk_lambda_grid', default='0.05,0.1', help='Development/tuning grid; ignored when --env_hparam_mode=fixed.')
        p.add_argument('--ab_rank_lambda', type=float, default=0.1, help='Formal fixed Stage-AB rank loss weight.')
        p.add_argument('--ab_gate_anchor_lambda', type=float, default=0.01, help='Formal fixed Stage-AB gate-anchor loss weight.')
        p.add_argument('--ab_residual_anchor_lambda', type=float, default=0.05, help='Formal fixed Stage-AB residual-anchor loss weight.')
        p.add_argument('--ab_bestofk_lambda', type=float, default=0.05, help='Formal fixed Stage-AB best-of-K loss weight.')
        p.add_argument('--run_ab_baseline', action=argparse.BooleanOptionalAction, default=False,
                       help='Train the same-run Local-AB-on-original-anchors ablation. Disabled by default for formal repeated runs.')
        p.add_argument('--run_counterfactuals', action=argparse.BooleanOptionalAction, default=False,
                       help='Run graph/local counterfactual diagnostics after training. Disabled by default for multi-seed formal runs.')
        p.add_argument('--e4_epochs', type=int, default=70)
        p.add_argument('--e4_patience', type=int, default=10)
        p.add_argument('--e4_batch_size', type=int, default=96)
        p.add_argument('--e4_lr', type=float, default=0.001)
        p.add_argument('--e4_weight_decay', type=float, default=0.0001)
        p.add_argument('--device', default='auto')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--fast', action='store_true')
        p.add_argument('--position_noise_aug', action='store_true', help='Enable fixed per-window Gaussian position-noise augmentation for train/val history.')
        p.add_argument('--noise_aug_sigmas', default='0,0.005,0.01,0.02,0.03,0.05', help='Position sigma values in meters.')
        p.add_argument('--noise_aug_probs', default='0.4,0.2,0.2,0.1,0.07,0.03', help='Categorical probabilities for noise_aug_sigmas.')
        p.add_argument('--noise_aug_seed', type=int, default=20260806, help='Base seed controlling fixed per-window noise realization.')
        p.add_argument('--noise_aug_val', action=argparse.BooleanOptionalAction, default=True, help='Apply the same mixture to validation windows. Test windows always remain clean during training.')
        args = p.parse_args()
        if args.c_d_model % args.c_attn_heads != 0:
            raise ValueError('--c_d_model 必须被 --c_attn_heads 整除。')
        if args.d_model % args.attn_heads != 0:
            raise ValueError('--d_model 必须被 --attn_heads 整除。')
        if args.corridor_token_topk != 0:
            raise ValueError('正式反事实实验要求 --corridor_token_topk=0，避免 token 位置泄露地图语义。')
        if args.fast:
            args.alt_iter = min(args.alt_iter, 8)
            args.epochs = min(args.epochs, 50)
            args.patience = min(args.patience, 10)
            args.anchor_lambda_grid = '0,0.01,0.1'
            args.balance_lambda_grid = '0,0.001'
            args.c_epochs = min(args.c_epochs, 45)
            args.c_patience = min(args.c_patience, 8)
            args.c_rank_lambda_grid = '0.5'
            args.c_gate_lambda_grid = '0.01'
            args.c_residual_lambda_grid = '0.05'
            args.c_bestofk_lambda_grid = '0.05'
            args.e4_epochs = min(args.e4_epochs, 45)
            args.e4_patience = min(args.e4_patience, 8)
            args.ab_rank_lambda_grid = '0.5'
            args.ab_gate_anchor_lambda_grid = '0.01'
            args.ab_residual_anchor_lambda_grid = '0.05'
            args.ab_bestofk_lambda_grid = '0.05'
        if args.env_hparam_mode == 'fixed':
            # Formal repeated-run protocol: hyperparameters are frozen across seeds,
            # operator variants, clean/noise-trained runs, and label regimes.
            args.c_rank_lambda_grid = str(args.c_rank_lambda)
            args.c_gate_lambda_grid = str(args.c_gate_lambda)
            args.c_residual_lambda_grid = str(args.c_residual_lambda)
            args.c_bestofk_lambda_grid = str(args.c_bestofk_lambda)
            args.ab_rank_lambda_grid = str(args.ab_rank_lambda)
            args.ab_gate_anchor_lambda_grid = str(args.ab_gate_anchor_lambda)
            args.ab_residual_anchor_lambda_grid = str(args.ab_residual_anchor_lambda)
            args.ab_bestofk_lambda_grid = str(args.ab_bestofk_lambda)
        set_seed(args.seed)
        device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
        out_dir = Path(args.out_dir)
        ensure_dir(out_dir)
        formal_cfg = {
            'env_hparam_mode': args.env_hparam_mode,
            'stage_c': {
                'rank_lambda': parse_float_list(args.c_rank_lambda_grid),
                'gate_lambda': parse_float_list(args.c_gate_lambda_grid),
                'residual_lambda': parse_float_list(args.c_residual_lambda_grid),
                'bestofk_lambda': parse_float_list(args.c_bestofk_lambda_grid),
            },
            'stage_ab': {
                'rank_lambda': parse_float_list(args.ab_rank_lambda_grid),
                'gate_anchor_lambda': parse_float_list(args.ab_gate_anchor_lambda_grid),
                'residual_anchor_lambda': parse_float_list(args.ab_residual_anchor_lambda_grid),
                'bestofk_lambda': parse_float_list(args.ab_bestofk_lambda_grid),
            },
            'run_ab_baseline': bool(args.run_ab_baseline),
            'run_counterfactuals': bool(args.run_counterfactuals),
            'note': 'Formal repeated runs freeze environment-module hyperparameters. Grid mode is development-only.',
        }
        (out_dir / 'formal_training_config.json').write_text(json.dumps(formal_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        print('condensed_operator_mode =', getattr(args, 'condensed_operator_mode', 'shared_residual'))
        print('env_hparam_mode =', args.env_hparam_mode)
        print('Stage-C loss weights =', args.c_rank_lambda_grid, args.c_gate_lambda_grid, args.c_residual_lambda_grid, args.c_bestofk_lambda_grid)
        print('Stage-AB loss weights =', args.ab_rank_lambda_grid, args.ab_gate_anchor_lambda_grid, args.ab_residual_anchor_lambda_grid, args.ab_bestofk_lambda_grid)
        print('run_ab_baseline =', args.run_ab_baseline, '| run_counterfactuals =', args.run_counterfactuals)
        print('device =', device)
        print('K =', args.K)
        print('mode_source =', args.mode_source)
        if args.mode_source == 'label_free':
            print('IMPORTANT: no action labels are used; using label-free latent D-KMoE candidates.')
        else:
            print('IMPORTANT: action/behavior labels are used for supervised Koopman candidate generation.')
        print('IMPORTANT: E6 is sequential: Global Topological C first, Local AB re-queries corrected candidates second.')
        print('IMPORTANT: map uses validated AirSim binvox decoding and NED-Z conversion.')
        voxel_map = e1.AirSimNEDClearanceMap.from_binvox(args.binvox_file, args.map_expected_xyz, args.map_center_xyz, args.map_resolution, axis_mode=getattr(args, 'map_axis_mode', 'swap_yz'))
        (out_dir / 'map_metadata_validated_ned.json').write_text(json.dumps(voxel_map.metadata(), ensure_ascii=False, indent=2), encoding='utf-8')
        root = Path(args.data_root)
        train_chains = bk.load_split_chains(root, 'train')
        val_chains = bk.load_split_chains(root, 'val')
        test_chains = bk.load_split_chains(root, 'test')
        tr = bk.make_split_data(train_chains, args.input_len, args.horizon, args.stride)
        va = bk.make_split_data(val_chains, args.input_len, args.horizon, args.stride)
        te = bk.make_split_data(test_chains, args.input_len, args.horizon, args.stride)
        print(f'samples: train={len(tr.hist_global)}, val={len(va.hist_global)}, test={len(te.hist_global)}')
        if args.position_noise_aug:
            noise_sigmas = parse_float_list(args.noise_aug_sigmas)
            noise_probs = parse_float_list(args.noise_aug_probs)
            if len(noise_sigmas) != len(noise_probs):
                raise ValueError('--noise_aug_sigmas and --noise_aug_probs must have the same length')
            tr = _rebuild_split_from_noisy_history(
                tr, noise_sigmas, noise_probs,
                seed=int(args.noise_aug_seed), split_name='train'
            )
            if bool(args.noise_aug_val):
                va = _rebuild_split_from_noisy_history(
                    va, noise_sigmas, noise_probs,
                    seed=int(args.noise_aug_seed) + 100003, split_name='val'
                )
            manifest = {
                'enabled': True,
                'position_only': True,
                'future_global_gt_clean': True,
                'velocity_clean': True,
                'behavior_labels_clean': True,
                'rebuild_local_frame_after_noise': True,
                'sigmas_m': noise_sigmas,
                'probabilities': (np.asarray(noise_probs, dtype=float) / np.sum(noise_probs)).tolist(),
                'noise_aug_seed': int(args.noise_aug_seed),
                'validation_augmented': bool(args.noise_aug_val),
                'test_augmented_during_training': False,
            }
            (out_dir / 'position_noise_augmentation_manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print('[POSITION-NOISE-AUG] manifest:', manifest)
        else:
            print('[POSITION-NOISE-AUG] disabled: clean training/validation.')
        if args.mode_source == 'auto':
            train_has_behavior = any(('behavior' in df.columns for df in train_chains))
            args.mode_source = 'action_label' if train_has_behavior else 'label_free'
            print('auto-selected mode_source =', args.mode_source)
        global_pack = base.fit_no_action_global(tr, va, args)
        gtr = base.predict_no_action_global(global_pack, tr, args).astype(np.float32)
        gva = base.predict_no_action_global(global_pack, va, args).astype(np.float32)
        gte = base.predict_no_action_global(global_pack, te, args).astype(np.float32)
        Xtr0 = prior.make_sequence_features(tr, args.dt)
        Xva0 = prior.make_sequence_features(va, args.dt)
        Xte0 = prior.make_sequence_features(te, args.dt)
        seq_std = prior.SeqStandardizer.fit(Xtr0)
        Xtr, Xva, Xte = (seq_std.transform(Xtr0), seq_std.transform(Xva0), seq_std.transform(Xte0))
        ytr = tr.future_global[:, :, :3].astype(np.float32)
        yva = va.future_global[:, :, :3].astype(np.float32)
        yte = te.future_global[:, :, :3].astype(np.float32)
        if args.mode_source == 'label_free':
            frozen = e2.fit_frozen_base_dkmoe(tr, va, te, gtr, gva, gte, Xtr, Xva, Xte, ytr, yva, args, device, out_dir)
            base_raw_method = 'differentiable_koopman_moe_raw'
            base_safe_method = 'differentiable_koopman_moe_safety'
        elif args.mode_source == 'action_label':
            frozen = fit_action_supervised_frozen_candidates(train_chains, val_chains, test_chains, tr, va, te, gtr, gva, gte, args, out_dir)
            base_raw_method = 'action_supervised_koopman_raw'
            base_safe_method = 'action_supervised_koopman_safety'
        else:
            raise ValueError(f'Unsupported mode_source={args.mode_source}')
        qtr0, dtr0, btr = e4.build_candidate_queries(tr, frozen['train']['raw'], frozen['train']['calibrated'], frozen['train']['pi'])
        qva0, dva0, bva = e4.build_candidate_queries(va, frozen['val']['raw'], frozen['val']['calibrated'], frozen['val']['pi'])
        qte0, dte0, bte = e4.build_candidate_queries(te, frozen['test']['raw'], frozen['test']['calibrated'], frozen['test']['pi'])
        cq_std, cd_std = (MaskedStandardizer.fit(qtr0), MaskedStandardizer.fit(dtr0))
        c_arrays = {'train': {'X': Xtr, 'raw_pi': frozen['train']['pi'], 'candidates': frozen['train']['calibrated'], 'query': cq_std.transform(qtr0), 'dyn': cd_std.transform(dtr0), 'basis': btr, 'y': ytr}, 'val': {'X': Xva, 'raw_pi': frozen['val']['pi'], 'candidates': frozen['val']['calibrated'], 'query': cq_std.transform(qva0), 'dyn': cd_std.transform(dva0), 'basis': bva, 'y': yva}, 'test': {'X': Xte, 'raw_pi': frozen['test']['pi'], 'candidates': frozen['test']['calibrated'], 'query': cq_std.transform(qte0), 'dyn': cd_std.transform(dte0), 'basis': bte, 'y': yte}}
        print('\n=== Building global sparse topological graph tokens ===')
        offsets, graph_shape = make_coarse_offsets(args)
        print('coarse grid nodes before sparsification =', len(offsets), '| shape =', graph_shape)
        gr_tr_raw, gm_tr, graph_geom_dim, graph_sem_dim = build_topological_graph_tokens(tr, voxel_map, offsets, graph_shape, args)
        gr_va_raw, gm_va, _, _ = build_topological_graph_tokens(va, voxel_map, offsets, graph_shape, args)
        gr_te_raw, gm_te, _, _ = build_topological_graph_tokens(te, voxel_map, offsets, graph_shape, args)
        graph_std = MaskedStandardizer.fit(gr_tr_raw, gm_tr)
        graph_tokens = {'train': graph_std.transform(gr_tr_raw), 'val': graph_std.transform(gr_va_raw), 'test': graph_std.transform(gr_te_raw)}
        graph_masks = {'train': gm_tr, 'val': gm_va, 'test': gm_te}
        null_te, null_mask_te = make_null_uniform_graph(len(te.hist_global), offsets, args.graph_max_nodes, graph_geom_dim, graph_sem_dim, graph_std, args)
        zero_attr_te = make_zero_attr_same_graph(gr_te_raw, graph_geom_dim, graph_std)
        print('\n=== Training Stage C: Global Topology Conditioned Candidate Generation ===')
        c_history, c_selection, c_sweeps = ([], [], [])
        c_best = None
        for rank_lambda in parse_float_list(args.c_rank_lambda_grid):
            for gate_lambda in parse_float_list(args.c_gate_lambda_grid):
                for residual_lambda in parse_float_list(args.c_residual_lambda_grid):
                    for bok_lambda in parse_float_list(args.c_bestofk_lambda_grid):
                        set_seed(args.seed + int(rank_lambda * 1000) + int(gate_lambda * 10000))
                        model, history, info, safety, sweep = train_c(c_arrays, graph_tokens, graph_masks, va, gva, args, device, rank_lambda, gate_lambda, residual_lambda, bok_lambda)
                        c_history.append(history)
                        c_selection.append(info)
                        sweep['rank_lambda'] = rank_lambda
                        sweep['gate_lambda'] = gate_lambda
                        sweep['residual_lambda'] = residual_lambda
                        sweep['bestofk_lambda'] = bok_lambda
                        c_sweeps.append(sweep)
                        if c_best is None or info['val_safety_score'] < c_best['info']['val_safety_score']:
                            c_best = {'model': copy.deepcopy(model).cpu(), 'info': info, 'safety': safety}
        pd.concat(c_history, ignore_index=True).to_csv(out_dir / 'e6_stage_c_training_history.csv', index=False)
        pd.DataFrame(c_selection).to_csv(out_dir / 'e6_stage_c_selection_val.csv', index=False)
        pd.concat(c_sweeps, ignore_index=True).to_csv(out_dir / 'e6_stage_c_safety_sweep_val.csv', index=False)
        c_model = c_best['model'].to(device)
        c_outputs = {}
        for split in ['train', 'val', 'test']:
            c_outputs[split] = infer_c(c_model, c_arrays[split], graph_tokens[split], graph_masks[split], args.c_batch_size, device)
        c_test_pred, c_test_pi, c_test_cand, c_test_res = c_outputs['test']
        c_test_safe, c_eff, _ = dk.apply_safety(gte, c_test_pred, c_best['safety']['alpha'], c_best['safety']['safety_tau'])
        ab_baseline = None
        ab_safe = ab_pi = ab_cand = ab_res = ab_eff = None
        if args.run_ab_baseline:
            print('\n=== [Ablation] Training same-run Local AB baseline on original Koopman anchors ===')
            ab_base_arrays, ab_base_tokens, _, ab_base_stds = ({}, {}, {}, None)
            arr, tok, raw, standards = build_local_pack(tr, frozen['train']['raw'], frozen['train']['pi'], frozen['train']['calibrated'], voxel_map, args, fit=True)
            ab_base_stds = standards
            ab_base_arrays['train'], ab_base_tokens['train'] = (set_X(arr, Xtr), tok)
            for split, data, X, raw_pred, raw_pi, candidates in [('val', va, Xva, frozen['val']['raw'], frozen['val']['pi'], frozen['val']['calibrated']), ('test', te, Xte, frozen['test']['raw'], frozen['test']['pi'], frozen['test']['calibrated'])]:
                arr, tok, _, _ = build_local_pack(data, raw_pred, raw_pi, candidates, voxel_map, args, query_std=ab_base_stds['query'], dyn_std=ab_base_stds['dyn'], corridor_std=ab_base_stds['corridor'], intention_std=ab_base_stds['intention'])
                ab_base_arrays[split], ab_base_tokens[split] = (set_X(arr, X), tok)
            ab_baseline, ab_hist, ab_sel, ab_sweep = train_ab(ab_base_arrays, ab_base_tokens, va, gva, args, device, 'local_ab_baseline')
            ab_hist.to_csv(out_dir / 'e6_ab_baseline_training_history.csv', index=False)
            ab_sel.to_csv(out_dir / 'e6_ab_baseline_selection_val.csv', index=False)
            ab_sweep.to_csv(out_dir / 'e6_ab_baseline_safety_sweep_val.csv', index=False)
            ab_pred, ab_pi, ab_cand, ab_res = e4.infer_model(ab_baseline['model'].to(device), ab_base_arrays['test'], ab_base_tokens['test']['corridor'], ab_base_tokens['test']['intention'], args.e4_batch_size, device)
            ab_safe, ab_eff, _ = dk.apply_safety(gte, ab_pred, ab_baseline['safety']['alpha'], ab_baseline['safety']['safety_tau'])
        else:
            print('\n=== Skipping same-run Local AB baseline (formal repeated-run mode) ===')
        print('\n=== Re-querying Local Fine map tokens around Stage-C candidates ===')
        cab_arrays, cab_tokens, cab_raw = ({}, {}, {})
        c_pred_tr, c_pi_tr, c_cand_tr, _ = c_outputs['train']
        arr, tok, raw, cab_stds = build_local_pack(tr, c_pred_tr, c_pi_tr, c_cand_tr, voxel_map, args, fit=True)
        cab_arrays['train'], cab_tokens['train'], cab_raw['train'] = (set_X(arr, Xtr), tok, raw)
        for split, data, X in [('val', va, Xva), ('test', te, Xte)]:
            cpred, cpi, ccand, _ = c_outputs[split]
            arr, tok, raw, _ = build_local_pack(data, cpred, cpi, ccand, voxel_map, args, query_std=cab_stds['query'], dyn_std=cab_stds['dyn'], corridor_std=cab_stds['corridor'], intention_std=cab_stds['intention'])
            cab_arrays[split], cab_tokens[split], cab_raw[split] = (set_X(arr, X), tok, raw)
        print('\n=== Training Stage AB around Global-Corrected candidate anchors ===')
        cab_best, cab_hist, cab_sel, cab_sweep = train_ab(cab_arrays, cab_tokens, va, gva, args, device, 'e6_c_plus_ab')
        cab_hist.to_csv(out_dir / 'e6_cab_training_history.csv', index=False)
        cab_sel.to_csv(out_dir / 'e6_cab_selection_val.csv', index=False)
        cab_sweep.to_csv(out_dir / 'e6_cab_safety_sweep_val.csv', index=False)
        cab_model = cab_best['model'].to(device)
        cab_pred, cab_pi, cab_cand, cab_res = e4.infer_model(cab_model, cab_arrays['test'], cab_tokens['test']['corridor'], cab_tokens['test']['intention'], args.e4_batch_size, device)
        cab_safe, cab_eff, _ = dk.apply_safety(gte, cab_pred, cab_best['safety']['alpha'], cab_best['safety']['safety_tau'])
        rows = [
            method_row(te, gte, 'v5_no_action_global', voxel_map, args),
            method_row(te, frozen['test']['raw'], base_raw_method, voxel_map, args),
            method_row(te, frozen['test']['safe'], base_safe_method, voxel_map, args),
        ]
        if args.run_ab_baseline:
            rows.append(method_row(te, ab_safe, 'e6_same_run_local_ab_baseline', voxel_map, args, {**candidate_diagnostics(ab_pi, ab_cand, yte, ab_res), 'test_mean_effective_safety_alpha': float(ab_eff.mean())}))
        rows.extend([
            method_row(te, c_test_safe, 'e6_stage_c_global_topology_only', voxel_map, args, {**candidate_diagnostics(c_test_pi, c_test_cand, yte, c_test_res), 'test_mean_effective_safety_alpha': float(c_eff.mean())}),
            method_row(te, cab_safe, 'e6_c_plus_ab_correct_graph_correct_local', voxel_map, args, {**candidate_diagnostics(cab_pi, cab_cand, yte, cab_res), 'test_mean_effective_safety_alpha': float(cab_eff.mean())}),
        ])
        compare_df = pd.DataFrame(rows)
        compare_df.to_csv(out_dir / 'compare_e6_cab_test.csv', index=False)
        np.save(out_dir / 'topokoop_test_pred_h10.npy', cab_safe.astype(np.float32))
        np.save(out_dir / 'topokoop_test_pred_raw_h10.npy', cab_pred.astype(np.float32))
        np.save(out_dir / 'topokoop_test_y_h10.npy', te.future_global[:, :, :3].astype(np.float32))
        np.save(out_dir / 'topokoop_stage_c_pred_h10.npy', c_test_safe.astype(np.float32))
        np.save(out_dir / 'topokoop_base_dkmoe_safety_pred_h10.npy', frozen['test']['safe'].astype(np.float32))
        report_horizons = parse_report_horizons(args.report_horizons, te.future_global.shape[1])
        report_predictions = {
            'v5_no_action_global': gte,
            base_raw_method: frozen['test']['raw'],
            base_safe_method: frozen['test']['safe'],
            'e6_stage_c_global_topology_only': c_test_safe,
            'TopoKoop': cab_safe,
            'e6_c_plus_ab_correct_graph_correct_local': cab_safe,
        }
        if args.run_ab_baseline:
            report_predictions['e6_same_run_local_ab_baseline'] = ab_safe
        multi_horizon_df = make_multi_horizon_table(te, report_predictions, voxel_map, args, report_horizons)
        multi_horizon_df.to_csv(out_dir / 'compare_topokoop_all_horizons_test.csv', index=False)
        multi_horizon_df.to_csv(out_dir / 'compare_e6_cab_all_horizons_test.csv', index=False)
        counter_df = pd.DataFrame()
        counter_predictions = {}
        if args.run_counterfactuals:
            print('\n=== Running sequential scale-decomposed counterfactual tests ===')
            rng = np.random.default_rng(args.seed + 20260528)
            permutation = rng.permutation(len(te.hist_global))
            global_conditions = {'correct_graph': (graph_tokens['test'], graph_masks['test']), 'shuffled_graph': (graph_tokens['test'][permutation], graph_masks['test'][permutation]), 'zero_attr_same_graph': (zero_attr_te, graph_masks['test']), 'null_uniform_graph': (null_te, null_mask_te)}
            requested_conditions = [('correct_graph', 'correct_local'), ('shuffled_graph', 'correct_local'), ('zero_attr_same_graph', 'correct_local'), ('null_uniform_graph', 'correct_local'), ('correct_graph', 'shuffled_local'), ('correct_graph', 'zero_local'), ('null_uniform_graph', 'zero_local')]
            counter_rows = []
            global_cache = {}
            for gname, (gtokens, gmask) in global_conditions.items():
                global_cache[gname] = infer_c(c_model, c_arrays['test'], gtokens, gmask, args.c_batch_size, device)
            for gname, lname in requested_conditions:
                cpred, cpi, ccand, cres = global_cache[gname]
                arr, tok, raw, _ = build_local_pack(te, cpred, cpi, ccand, voxel_map, args, query_std=cab_stds['query'], dyn_std=cab_stds['dyn'], corridor_std=cab_stds['corridor'], intention_std=cab_stds['intention'])
                arr = set_X(arr, Xte)
                if lname == 'shuffled_local':
                    tok['corridor'] = replace_semantics_with_permutation(raw['corridor'], raw['corridor_geom_dim'], permutation, cab_stds['corridor'])
                    tok['intention'] = replace_semantics_with_permutation(raw['intention'], raw['intention_geom_dim'], permutation, cab_stds['intention'])
                elif lname == 'zero_local':
                    tok['corridor'] = e4.zero_semantic(raw['corridor'], raw['corridor_geom_dim'], cab_stds['corridor'])
                    tok['intention'] = e4.zero_semantic(raw['intention'], raw['intention_geom_dim'], cab_stds['intention'])
                pred, pi, refined, residual = e4.infer_model(cab_model, arr, tok['corridor'], tok['intention'], args.e4_batch_size, device)
                pred_safe, _, _ = dk.apply_safety(gte, pred, cab_best['safety']['alpha'], cab_best['safety']['safety_tau'])
                method = f'e6_cab_{gname}_{lname}_plus_global_safety'
                counter_predictions[method] = pred_safe
                counter_rows.append(method_row(te, pred_safe, method, voxel_map, args, {'global_condition': gname, 'local_condition': lname, **candidate_diagnostics(pi, refined, yte, residual)}))
            counter_df = pd.DataFrame(counter_rows)
            counter_df.to_csv(out_dir / 'e6_cab_counterfactual_test.csv', index=False)
            counter_subset_df = subset_metrics(te, counter_predictions, voxel_map, frozen['test']['calibrated'], args)
            counter_subset_df.to_csv(out_dir / 'e6_cab_counterfactual_subset_test.csv', index=False)
        else:
            print('\n=== Skipping counterfactual diagnostics (formal repeated-run mode) ===')

        predictions = {
            base_safe_method: frozen['test']['safe'],
            'e6_stage_c_global_topology_only': c_test_safe,
            'e6_c_plus_ab_correct_graph_correct_local': cab_safe,
        }
        if args.run_ab_baseline:
            predictions['e6_same_run_local_ab_baseline'] = ab_safe
        subset_df = subset_metrics(te, predictions, voxel_map, frozen['test']['calibrated'], args)
        subset_df.to_csv(out_dir / 'e6_subset_metrics_test.csv', index=False)
        # Self-contained Stage-C checkpoint: include every standardizer needed to
        # rebuild its input tensors during real-time inference.
        stage_c_checkpoint = {
            'format_version': 2,
            'model_state_dict': e2._cpu_state_dict(c_best['model']),
            'model_config': {
                'sequence_dim': int(c_arrays['train']['X'].shape[-1]),
                'query_dim': int(c_arrays['train']['query'].shape[-1]),
                'dynamic_dim': int(c_arrays['train']['dyn'].shape[-1]),
                'graph_dim': int(graph_tokens['train'].shape[-1]),
                'K': int(args.K),
                'horizon': int(args.horizon),
            },
            'selection_info': c_best['info'],
            'safety': c_best['safety'],
            'standardizers': {
                'sequence': e2._pack_standardizer(seq_std),
                'candidate_query': e2._pack_standardizer(cq_std),
                'candidate_dynamic': e2._pack_standardizer(cd_std),
                'graph': e2._pack_standardizer(graph_std),
            },
            'graph_geometry_dim': int(graph_geom_dim),
            'graph_semantic_dim': int(graph_sem_dim),
            'graph_offsets': np.asarray(offsets).copy(),
            'graph_shape': tuple(int(v) for v in graph_shape),
            'map_metadata': voxel_map.metadata(),
            'args': vars(args),
        }
        torch.save(stage_c_checkpoint, out_dir / 'e6_stage_c_global_topology_generator.pt')

        # Self-contained Stage-AB checkpoint for the final sequential TopoKoop stage.
        stage_ab_checkpoint = {
            'format_version': 2,
            'model_state_dict': e2._cpu_state_dict(cab_best['model']),
            'model_config': {
                'variant': 'AB',
                'sequence_dim': int(cab_arrays['train']['X'].shape[-1]),
                'query_dim': int(cab_arrays['train']['query'].shape[-1]),
                'dynamic_dim': int(cab_arrays['train']['dyn'].shape[-1]),
                'corridor_dim': int(cab_tokens['train']['corridor'].shape[-1]),
                'intention_dim': int(cab_tokens['train']['intention'].shape[-1]),
                'K': int(args.K),
                'horizon': int(args.horizon),
            },
            'selection_info': cab_best['info'],
            'safety': cab_best['safety'],
            'standardizers': {
                'sequence': e2._pack_standardizer(seq_std),
                'query': e2._pack_standardizer(cab_stds['query']),
                'dynamic': e2._pack_standardizer(cab_stds['dyn']),
                'corridor': e2._pack_standardizer(cab_stds['corridor']),
                'intention': e2._pack_standardizer(cab_stds['intention']),
            },
            'map_metadata': voxel_map.metadata(),
            'args': vars(args),
        }
        torch.save(stage_ab_checkpoint, out_dir / 'e6_stage_ab_local_refiner.pt')

        # Export one complete deployment bundle for label-free real-time inference.
        # It contains the global Koopman predictor, all K Switching-Koopman operators,
        # the frozen D-KMoE, Stage C, Stage AB, and every fitted standardizer.
        deployment_bundle_path = None
        if args.mode_source == 'label_free':
            deployment_bundle_path = out_dir / 'topokoop_e6_label_free_realtime_bundle.pt'
            deployment_bundle = {
                'format_version': 2,
                'model_name': 'TopoKoop_E6_CAB_label_free',
                'input_state_order': ['x', 'y', 'z', 'vx', 'vy', 'vz'],
                'input_len': int(args.input_len),
                'horizon': int(args.horizon),
                'dt': float(args.dt),
                'K': int(args.K),
                'config': vars(args),
                'global_koopman': e2._pack_global_koopman(global_pack),
                'switching_koopman': e2._pack_switching_koopman(frozen['latent'], args),
                'base_dkmoe': {
                    'model_state_dict': e2._cpu_state_dict(frozen['model']),
                    'model_config': {
                        'input_dim': int(Xtr.shape[-1]),
                        'K': int(args.K),
                        'horizon': int(args.horizon),
                        'hidden_dim': int(args.hidden_dim),
                        'gru_layers': int(args.gru_layers),
                        'dropout': float(args.dropout),
                        'max_scale_deviation': float(args.max_scale_deviation),
                    },
                    'best_pack': frozen['best_pack'],
                    'sequence_standardizer': e2._pack_standardizer(seq_std),
                },
                'stage_c': stage_c_checkpoint,
                'stage_ab': stage_ab_checkpoint,
                'map': {
                    'metadata': voxel_map.metadata(),
                    'binvox_file_at_training': str(Path(args.binvox_file).resolve()),
                },
            }
            torch.save(deployment_bundle, deployment_bundle_path)
            print('saved complete label-free real-time bundle:', deployment_bundle_path)
        elif args.mode_source == 'action_label':
            deployment_bundle_path = out_dir / 'topokoop_e6_action_label_realtime_bundle.pt'
            action_model = frozen['action_model']
            gate_model = frozen.get('action_gate_model')
            gate_pack = None
            if gate_model is not None:
                gate_pack = {
                    'model_state_dict': e2._cpu_state_dict(gate_model),
                    'input_dim': int(gate_model.net[0].in_features),
                    'hidden_dim': int(args.action_gate_hidden_dim),
                    'dropout': float(args.action_gate_dropout),
                    'use_current_behavior': bool(args.action_gate_use_current_behavior),
                }
            deployment_bundle = {
                'format_version': 1,
                'model_name': 'TopoKoop_E6_CAB_action_label',
                'input_state_order': ['x', 'y', 'z', 'vx', 'vy', 'vz'],
                'input_len': int(args.input_len),
                'horizon': int(args.horizon),
                'dt': float(args.dt),
                'K': int(args.K),
                'config': vars(args),
                'global_koopman': e2._pack_global_koopman(global_pack),
                'action_frontend': {
                    'koopman': _pack_action_koopman(action_model),
                    'probability_mode': str(args.action_probability_mode),
                    'transition_matrix': np.asarray(frozen['transition_matrix']).copy(),
                    'gate': gate_pack,
                    'base_safety': frozen['best_pack']['safe'],
                },
                'stage_c': stage_c_checkpoint,
                'stage_ab': stage_ab_checkpoint,
                'map': {
                    'metadata': voxel_map.metadata(),
                    'binvox_file_at_training': str(Path(args.binvox_file).resolve()),
                },
            }
            torch.save(deployment_bundle, deployment_bundle_path)
            print('saved complete action-label real-time bundle:', deployment_bundle_path)
        report_models = {
            'stage_c': 'Global sparse free-space graph conditions motion modes and creates corrected anchors.',
            'c_plus_ab': 'Local AB refinement re-queried around Stage-C corrected anchors.',
        }
        if args.run_ab_baseline:
            report_models['same_run_local_ab_baseline'] = 'Local AB refinement around original frozen Koopman anchors.'
        success_criteria = ['Stage-C candidate oracle improvement indicates global topology improves candidate generation itself.']
        if args.run_ab_baseline:
            success_criteria.append('C+AB should outperform same-run Local-AB and the frozen base safety predictor.')
        if args.run_counterfactuals:
            success_criteria.extend([
                'Correct graph/correct local should outperform null or shuffled global graph conditions.',
                'Correct graph/correct local should outperform shuffled or zero local conditions.',
            ])
        report = {
            'experiment': 'E6 sequential Global Topological C + Local Fine AB Koopman',
            'protocol': {
                'action_labels_used': bool(args.mode_source == 'action_label'),
                'mode_source': args.mode_source,
                'action_probability_mode': getattr(args, 'action_probability_mode', None),
                'future_behavior_target': getattr(args, 'future_behavior_target', None),
                'base_koopman_candidates_frozen': True,
                'sequential_local_requery': True,
                'env_hparam_mode': args.env_hparam_mode,
                'formal_fixed_environment_hparams': bool(args.env_hparam_mode == 'fixed'),
                'run_ab_baseline': bool(args.run_ab_baseline),
                'run_counterfactuals': bool(args.run_counterfactuals),
                'note': 'Formal repeated runs freeze Stage-C/AB loss weights; Local-AB baseline and counterfactuals are diagnostic-only.',
                'map_coordinate_convention': 'Validated AirSim binvox decoding and NED-Z reversal.',
            },
            'models': report_models,
            'stage_c_selection': c_best['info'],
            'stage_ab_selection': cab_best['info'],
            'same_run_ab_selection': None if ab_baseline is None else ab_baseline['info'],
            'test_results': compare_df.to_dict(orient='records'),
            'counterfactual_results': counter_df.to_dict(orient='records'),
            'success_criteria': success_criteria,
            'args': vars(args),
        }
        (out_dir / 'report_e6_cab.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('\n=== E6 test performance ===')
        print(compare_df.to_string(index=False))
        if args.run_counterfactuals:
            print('\n=== E6 sequential counterfactual tests ===')
            print(counter_df.to_string(index=False))
        print('\nKey files:')
        key_files = ['formal_training_config.json', 'compare_e6_cab_test.csv', 'compare_topokoop_all_horizons_test.csv', 'topokoop_test_pred_h10.npy', 'topokoop_test_y_h10.npy', 'e6_subset_metrics_test.csv', 'e6_stage_c_selection_val.csv', 'e6_cab_selection_val.csv', 'e6_stage_c_global_topology_generator.pt', 'e6_stage_ab_local_refiner.pt', 'report_e6_cab.json']
        if args.run_ab_baseline:
            key_files.extend(['e6_ab_baseline_training_history.csv', 'e6_ab_baseline_selection_val.csv'])
        if args.run_counterfactuals:
            key_files.extend(['e6_cab_counterfactual_test.csv', 'e6_cab_counterfactual_subset_test.csv'])
        if args.mode_source == 'label_free':
            key_files.extend(['label_free_switching_koopman_frontend.pt', 'label_free_base_dkmoe.pt', 'topokoop_e6_label_free_realtime_bundle.pt'])
        elif args.mode_source == 'action_label':
            key_files.extend(['action_moe_gate.pt', 'action_supervised_candidate_report.json', 'topokoop_e6_action_label_realtime_bundle.pt'])
        for filename in key_files:
            print(' ', out_dir / filename)

    return _SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

topokoop_formal = _init_topokoop_formal()


# ========================================================================================
# MAIN ENTRY
# ========================================================================================

if __name__ == "__main__":
    topokoop_formal.main()
