#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_topokoop_multiseed.sh [DATA_ROOT] [OUT_ROOT] [MODE] [GPU]
#
# MODE:
#   action_label
#   label_free
#   auto

DATA_ROOT="${1:-data/action_label}"
OUT_ROOT="${2:-outputs/topokoop_multiseed}"
MODE="${3:-action_label}"
GPU="${4:-0}"

BINVOX_FILE="data/map/city_trajectory_roi_train_val.binvox"

MAP_EXPECTED_XYZ=(417 359 62)
MAP_CENTER_XYZ=(156.5914516449 0.4333877563 -28.8091073036)
MAP_RESOLUTION="1.0"
MAP_AXIS_MODE="current_xyz"

SEEDS=(42 43 44 45 46)

mkdir -p "${OUT_ROOT}"
mkdir -p logs

for SEED in "${SEEDS[@]}"; do

    OUT="${OUT_ROOT}/seed${SEED}"

    echo "============================================================"
    echo "TopoKoop"
    echo "mode  : ${MODE}"
    echo "seed  : ${SEED}"
    echo "output: ${OUT}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" python -u src/topokoop.py \
        --data_root "${DATA_ROOT}" \
        --binvox_file "${BINVOX_FILE}" \
        --out_dir "${OUT}" \
        --input_len 20 \
        --horizon 10 \
        --report_horizons 1,3,5,10 \
        --stride 5 \
        --mode_source "${MODE}" \
        --condensed_operator_mode independent \
        --map_expected_xyz "${MAP_EXPECTED_XYZ[@]}" \
        --map_center_xyz "${MAP_CENTER_XYZ[@]}" \
        --map_resolution "${MAP_RESOLUTION}" \
        --map_axis_mode "${MAP_AXIS_MODE}" \
        --seed "${SEED}" \
        2>&1 | tee "logs/topokoop_${MODE}_seed${SEED}.log"

done
