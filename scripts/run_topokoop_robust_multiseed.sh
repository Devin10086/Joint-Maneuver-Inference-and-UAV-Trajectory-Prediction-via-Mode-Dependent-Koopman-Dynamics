#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/label_free}"
BINVOX="${2:-DATA/airsim_city_voxel/city.binvox}"
BASE_OUT="${3:-outputs/topokoop_robust_multiseed}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$BASE_OUT" logs

for seed in 1 2 3 4 5; do
  OUT="${BASE_OUT}/seed${seed}"
  echo "============================================================"
  echo "TopoKoop V5 position-noise augmentation | seed=${seed}"
  echo "OUT=${OUT}"
  echo "============================================================"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES="$GPU" python -u \
    src/topokoop_robust.py \
    --data_root "$DATA_ROOT" \
    --binvox_file "$BINVOX" \
    --out_dir "$OUT" \
    --input_len 20 \
    --horizon 10 \
    --report_horizons 1,3,5,10 \
    --stride 5 \
    --dt 0.1 \
    --K 12 \
    --mode_source label_free \
    --map_expected_xyz 417 359 62 \
    --map_center_xyz 156.5914516449 0.4333877563 -28.8091073036 \
    --map_resolution 1.0 \
    --map_axis_mode current_xyz \
    --train_position_noise_aug \
    --noise_aug_sigmas 0,0.005,0.01,0.02,0.03,0.05 \
    --noise_aug_probs 0.4,0.2,0.2,0.1,0.07,0.03 \
    --noise_aug_train_copies 1 \
    --noise_aug_val_copies 1 \
    --noise_aug_seed 20260806 \
    --seed "$seed" \
    --device cuda:0 \
    --fast \
    2>&1 | tee "logs/topokoop_v5_position_noise_aug_seed${seed}.log"
done

echo "[DONE] TopoKoop 5-seed position-noise augmented training finished: $BASE_OUT"
