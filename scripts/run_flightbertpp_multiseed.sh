#!/usr/bin/env bash
set -euo pipefail

# FlightBERT++ paper/source-faithful continuous-regression adaptation, 5 seeds.
# Run from the project root where flightbertpp_uav_regression_v3_paper_faithful.py exists.

DATA_ROOT="${1:-DATA/topokoop_v4_320_label_free/processed}"
BASE_OUT="${2:-DATA/topokoop_v4_320_label_free/baselines/flightbertpp_v3_paper_faithful_multiseed}"

mkdir -p "$BASE_OUT"

for seed in 1 2 3 4 5
do
  echo "========================================"
  echo "Running FlightBERT++ V3 paper-faithful seed=${seed}"
  echo "Output: ${BASE_OUT}/seed${seed}"
  echo "========================================"

  python -u baselines/flightbertpp.py \
    --data_root "$DATA_ROOT" \
    --input_len 20 \
    --horizon 10 \
    --report_horizons 1,3,5,10 \
    --stride 5 \
    --dt 0.1 \
    --epochs 80 \
    --d_model 192 \
    --n_heads 4 \
    --n_layers_enc 4 \
    --n_layers_dec 2 \
    --batch_size 1024 \
    --lr 1e-4 \
    --embedding_type conv \
    --val_metric pos_rmse \
    --patience 20 \
    --seed "$seed" \
    --out_dir "${BASE_OUT}/seed${seed}"
done

# Reuse the existing summarizer if present.
