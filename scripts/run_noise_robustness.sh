#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="DATA/topokoop_v4_320_label_free/processed"
TOPO_SOURCE="src/topokoop.py"
BASE="DATA/topokoop_v4_320_label_free/baselines"
OUT_BASE="${BASE}/noise_robustness_position_threshold_full"
SIGMAS="0,0.005,0.01,0.02,0.03,0.05,0.1"
NOISE_SEEDS="42,43,44,45,46"
DEVICE="cuda:0"

mkdir -p "${OUT_BASE}" logs

CUDA_VISIBLE_DEVICES=0 python -u evaluation/evaluate_baseline_noise.py \
  --model flightbertpp \
  --model_source_py baselines/flightbertpp.py \
  --topokoop_source_py "${TOPO_SOURCE}" \
  --checkpoints_root "${BASE}/flightbertpp_v3_paper_faithful_multiseed" \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_BASE}/flightbertpp" \
  --noise_mode position_only \
  --position_sigmas "${SIGMAS}" \
  --velocity_sigmas 0 \
  --noise_seeds "${NOISE_SEEDS}" \
  --max_samples 0 \
  --sample_mode evenly \
  --report_horizons 1,3,5,10 \
  --batch_size 512 \
  --device "${DEVICE}" \
  2>&1 | tee logs/flightbertpp_noise_position_threshold_full.log

CUDA_VISIBLE_DEVICES=0 python -u evaluation/evaluate_baseline_noise.py \
  --model pecnet \
  --model_source_py baselines/pecnet.py \
  --topokoop_source_py "${TOPO_SOURCE}" \
  --checkpoints_root "${BASE}/pecnet_v2_paper_faithful_multiseed" \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_BASE}/pecnet" \
  --noise_mode position_only \
  --position_sigmas "${SIGMAS}" \
  --velocity_sigmas 0 \
  --noise_seeds "${NOISE_SEEDS}" \
  --max_samples 0 \
  --sample_mode evenly \
  --report_horizons 1,3,5,10 \
  --batch_size 512 \
  --device "${DEVICE}" \
  2>&1 | tee logs/pecnet_noise_position_threshold_full.log

CUDA_VISIBLE_DEVICES=0 python -u evaluation/evaluate_baseline_noise.py \
  --model mtr \
  --model_source_py baselines/mtr.py \
  --topokoop_source_py "${TOPO_SOURCE}" \
  --checkpoints_root "${BASE}/mtr_v3_paper_faithful_multiseed" \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_BASE}/mtr" \
  --noise_mode position_only \
  --position_sigmas "${SIGMAS}" \
  --velocity_sigmas 0 \
  --noise_seeds "${NOISE_SEEDS}" \
  --max_samples 0 \
  --sample_mode evenly \
  --report_horizons 1,3,5,10 \
  --batch_size 128 \
  --device "${DEVICE}" \
  2>&1 | tee logs/mtr_noise_position_threshold_full.log

echo "[DONE] all paper-faithful baseline noise evaluations"
echo "Output: ${OUT_BASE}"
