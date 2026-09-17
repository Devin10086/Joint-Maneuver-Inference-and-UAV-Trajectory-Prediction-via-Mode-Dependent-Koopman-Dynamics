#!/usr/bin/env bash
set -euo pipefail

# PECNet-UAV V2 paper-faithful multi-seed runner.
# Run from the UAV project root.
#
# Usage:
#   bash run_pecnet_v2_paper_faithful_multiseed.sh [DATA_ROOT] [BASE_OUT]

DATA_ROOT="${1:-DATA/topokoop_v4_320_label_free/processed}"
BASE_OUT="${2:-DATA/topokoop_v4_320_label_free/baselines/pecnet_v2_paper_faithful_multiseed}"

mkdir -p "$BASE_OUT"

for seed in 1 2 3 4 5
do
  echo "========================================"
  echo "Running PECNet-UAV V2 paper-faithful seed=${seed}"
  echo "Data:   ${DATA_ROOT}"
  echo "Output: ${BASE_OUT}/seed${seed}"
  echo "========================================"

  python -u baselines/pecnet.py \
    --data_root "$DATA_ROOT" \
    --out_dir "${BASE_OUT}/seed${seed}" \
    --input_len 20 \
    --horizon 10 \
    --report_horizons 1,3,5,10 \
    --stride 5 \
    --dt 0.1 \
    --past_feature_mode state6 \
    --fdim 16 \
    --zdim 16 \
    --enc_past_sizes 512,256 \
    --enc_dest_sizes 8,16 \
    --enc_latent_sizes 8,50 \
    --dec_sizes 1024,512,1024 \
    --predictor_sizes 1024,512,256 \
    --dropout -1 \
    --epochs 650 \
    --batch_size 512 \
    --lr 3e-4 \
    --weight_decay 0 \
    --patience 50 \
    --kld_reg 1.0 \
    --adl_reg 1.0 \
    --num_samples 20 \
    --sigma 1.3 \
    --truncation 0 \
    --val_metric pos_rmse \
    --seed "$seed"
done

