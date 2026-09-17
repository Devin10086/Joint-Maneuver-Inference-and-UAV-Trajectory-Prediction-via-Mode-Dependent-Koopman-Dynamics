# Joint Maneuver Inference and UAV Trajectory Prediction via Mode-Dependent Koopman Dynamics

Official research code and processed data for **Joint Maneuver Inference and UAV Trajectory Prediction via Mode-Dependent Koopman Dynamics**.

> This repository is maintained as the reproducibility package associated with the manuscript.

## Overview

This repository implements **TopoKoop**, a UAV trajectory prediction framework based on mode-dependent Koopman dynamics and environment-aware trajectory refinement.

The main implementation supports multiple maneuver-information settings through a unified interface:

- **Action-label mode (`action_label`)**: maneuver information is available during training.
- **Label-free mode (`label_free`)**: maneuver modes are inferred without explicit action labels.
- **Auto mode (`auto`)**: action labels are used automatically when available.

A separate robust-training implementation is provided for position-noise augmentation experiments.

## Repository Structure

```text
.
├── src/
│   ├── topokoop.py
│   └── topokoop_robust.py
├── baselines/
│   ├── flightbertpp.py
│   ├── mtr.py
│   └── pecnet.py
├── evaluation/
│   └── evaluate_baseline_noise.py
├── scripts/
│   ├── run_topokoop_multiseed.sh
│   ├── run_topokoop_robust_multiseed.sh
│   ├── run_flightbertpp_multiseed.sh
│   ├── run_mtr_multiseed.sh
│   ├── run_pecnet_multiseed.sh
│   └── run_noise_robustness.sh
├── data/
│   ├── action_label/
│   ├── label_free/
│   ├── map/
│   │   └── city_trajectory_roi_train_val.binvox
│   └── README.md
├── results/
├── requirements.txt
└── README.md
```

## Environment

The core Python dependencies are:

- Python
- PyTorch
- NumPy
- pandas

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

For GPU experiments, install a CUDA-compatible PyTorch build for your local CUDA environment.

## Included Data

The processed UAV trajectory datasets used in the paper are included in this repository.

### Action-label dataset

```text
data/action_label/
```

### Label-free dataset

```text
data/label_free/
```

The voxel environment map used in the paper is also included:

```text
data/map/city_trajectory_roi_train_val.binvox
```

The map configuration used in the formal experiments is:

```text
Decoded XYZ shape : 417 × 359 × 62
Map center (XYZ)  : (156.5914516449, 0.4333877563, -28.8091073036)
Resolution        : 1.0 m
Axis mode         : current_xyz
```

See `data/README.md` for details.

## Main Experimental Configuration

The main trajectory-prediction experiments use:

```text
Observation length : 20 steps
Prediction horizon : 10 steps
Sampling interval  : 0.1 s
Stride             : 5
```

Reported prediction horizons are:

```text
1, 3, 5, 10
```

The repeated-run experiments use five random seeds:

```text
42, 43, 44, 45, 46
```

## Main TopoKoop Model

The main implementation is:

```text
src/topokoop.py
```

### Action-label mode

```bash
python -u src/topokoop.py   --data_root data/action_label   --binvox_file data/map/city_trajectory_roi_train_val.binvox   --out_dir outputs/action_label   --input_len 20   --horizon 10   --report_horizons 1,3,5,10   --stride 5   --mode_source action_label   --condensed_operator_mode independent   --map_expected_xyz 417 359 62   --map_center_xyz 156.5914516449 0.4333877563 -28.8091073036   --map_resolution 1.0   --map_axis_mode current_xyz   --seed 42
```

### Label-free mode

```bash
python -u src/topokoop.py   --data_root data/label_free   --binvox_file data/map/city_trajectory_roi_train_val.binvox   --out_dir outputs/label_free   --input_len 20   --horizon 10   --report_horizons 1,3,5,10   --stride 5   --mode_source label_free   --condensed_operator_mode independent   --map_expected_xyz 417 359 62   --map_center_xyz 156.5914516449 0.4333877563 -28.8091073036   --map_resolution 1.0   --map_axis_mode current_xyz   --seed 42
```

### Auto mode

When action labels may or may not be available, use:

```text
--mode_source auto
```

## Multi-Seed Experiments

For the action-label setting:

```bash
bash scripts/run_topokoop_multiseed.sh
```

By default, the script uses:

```text
data/action_label
data/map/city_trajectory_roi_train_val.binvox
```

For the label-free setting:

```bash
bash scripts/run_topokoop_multiseed.sh   data/label_free   outputs/topokoop_label_free   label_free   0
```

The final argument specifies the GPU index.

## Robust Training

The position-noise robust implementation is:

```text
src/topokoop_robust.py
```

The corresponding multi-seed experiment is launched with:

```bash
bash scripts/run_topokoop_robust_multiseed.sh
```

The default robust-training script uses:

```text
data/label_free
data/map/city_trajectory_roi_train_val.binvox
```

and writes outputs under:

```text
outputs/topokoop_robust_multiseed/
```

## Baselines

Paper-oriented baseline implementations are provided for:

- FlightBERT++
- MTR
- PECNet

The corresponding source files are located under:

```text
baselines/
```

Multi-seed launch scripts are provided under:

```text
scripts/
```

## Noise-Robustness Evaluation

The baseline noise-robustness evaluation entry point is:

```text
evaluation/evaluate_baseline_noise.py
```

The corresponding launch script is:

```bash
bash scripts/run_noise_robustness.sh
```

## Results

The `results/` directory contains aggregated CSV results associated with the paper experiments, including:

- main trajectory-prediction comparisons;
- action-label results;
- label-free results;
- horizon-wise evaluation;
- ablation results.

These files are provided as reference outputs for reproducibility checks.

## Reproducibility Status

The reorganized repository has been smoke-tested from the repository root.

The action-label TopoKoop pipeline successfully completed:

- processed trajectory-data loading;
- voxel-map loading;
- action-supervised mode-dependent Koopman prediction;
- global topology conditioning;
- local trajectory refinement;
- test-set evaluation;
- real-time deployment bundle export.

The Python source files and evaluation scripts pass Python syntax compilation, and all shell launch scripts pass `bash -n` syntax checks.

## Notes

Large experiment outputs, checkpoints, logs, and temporary files are excluded from Git tracking.

The included processed datasets and voxel map are intended to support direct reproduction of the paper experiments.

## Citation

Citation information will be added after publication.
