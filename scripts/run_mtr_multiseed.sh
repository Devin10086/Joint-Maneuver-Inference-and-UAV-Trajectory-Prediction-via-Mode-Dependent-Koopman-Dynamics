#!/usr/bin/env bash
set -euo pipefail

# MTR-UAV V3 paper-faithful multi-seed runner.
# Run from the project root containing:
#   mtr_uav_lite_v3_paper_faithful.py

DATA_ROOT="${1:-DATA/topokoop_v4_320_label_free/processed}"
BASE_OUT="${2:-DATA/topokoop_v4_320_label_free/baselines/mtr_v3_paper_faithful_multiseed}"

mkdir -p "$BASE_OUT"

for seed in 1 2 3 4 5
do
  echo "========================================"
  echo "Running MTR-UAV V3 paper-faithful seed=${seed}"
  echo "Data:   ${DATA_ROOT}"
  echo "Output: ${BASE_OUT}/seed${seed}"
  echo "========================================"

  python -u baselines/mtr.py \
    --data_root "$DATA_ROOT" \
    --out_dir "${BASE_OUT}/seed${seed}" \
    --input_len 20 \
    --horizon 10 \
    --report_horizons 1,3,5,10 \
    --stride 5 \
    --dt 0.1 \
    --num_queries 64 \
    --num_output_modes 6 \
    --kmeans_iters 100 \
    --nms_dist_thresh 0.5 \
    --d_model 192 \
    --n_heads 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 6 \
    --dropout 0.1 \
    --epochs 30 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --lr_milestones 20,22,24,26,28 \
    --lr_gamma 0.5 \
    --patience 12 \
    --val_metric pos_rmse \
    --cls_weight 1.0 \
    --reg_weight 1.0 \
    --vel_weight 0.5 \
    --grad_clip 1000 \
    --seed "$seed"
done

