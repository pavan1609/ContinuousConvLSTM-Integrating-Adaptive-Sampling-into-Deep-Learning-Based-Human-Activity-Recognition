# ContinuousConvLSTM: Integrating Adaptive Sampling into Deep Learning-Based Human Activity Recognition

Official code release for the paper:

> **Integrating Adaptive Sampling into Deep Learning-Based Human Activity Recognition**

The repository contains the model implementations, dataset-preparation utilities, experiment configurations, SLURM launchers, LOSO training pipeline, and evaluation scripts used for the experiments reported in the paper.

The main model, **ContinuousConvLSTM**, replaces fixed discrete temporal convolution kernels with continuously parameterized kernels. The effective discrete kernel is evaluated according to the active sampling frequency, allowing one trained model to operate on multiple sensor sampling rates.

The repository supports the following datasets:

| Dataset | Native rate | Evaluated rates | Subjects | Classes |
|---|---:|---:|---:|---:|
| WEAR | 50 Hz | 50, 25, 12, 6 Hz | 22 | 18 activities + null |
| RealWorld-HAR | 50 Hz | 50, 25, 12, 6 Hz | 15 | 8 activities |
| WISDM-watch | 20 Hz | 20, 10, 5 Hz | 51 | 18 activities |

The code supports:

- standard DeepConvLSTM;
- pure CNN without recurrent layers;
- ContinuousConvLSTM with rate-specific branches;
- single-checkpoint multirate training;
- single-branch ContinuousConv ablations;
- fixed-kernel sweeps;
- LOSO evaluation;
- per-activity and aggregate metric generation.

---

## 1. Repository structure

```text
ContinuousConvLSTM/
├── configs/
│   ├── DeepConvLSTM/
│   │   ├── WEAR/
│   │   ├── RWHAR/
│   │   └── WISDM_WATCH/
│   │
│   ├── PureCNN/
│   │   ├── WEAR/
│   │   ├── RWHAR/
│   │   └── WISDM_WATCH/
│   │
│   ├── ContinuousConvLSTM/
│   │   ├── WEAR/
│   │   ├── RWHAR/
│   │   └── WISDM_WATCH/
│   │
│   ├── Ablations/
│   ├── MANIFEST.csv
│   ├── README.md
│   └── TREE.txt
│
├── jobs/
│   ├── DeepConvLSTM/
│   │   ├── WEAR/
│   │   ├── RWHAR/
│   │   └── WISDM_WATCH/
│   │
│   ├── PureCNN/
│   │   ├── WEAR/
│   │   ├── RWHAR/
│   │   └── WISDM_WATCH/
│   │
│   ├── ContinuousConvLSTM/
│   │   ├── WEAR/
│   │   ├── RWHAR/
│   │   └── WISDM_WATCH/
│   │
│   ├── Ablations/
│   └── KernelSweep/
│
├── models/
│   ├── DeepConvLSTM.py
│   ├── continuous_conv.py
│   ├── train.py
│   ├── map_metric.py
│   └── __init__.py
│
├── scripts/
│   ├── run_loso.py
│   ├── run_kernel_sweep_from_configs.py
│   ├── preprocess_realworld_waist_accel.py
│   └── create_realworld_loso_annotations.py
│
├── utils/
│   ├── tools/
│   │   └── prepare_dataset.py
│   ├── data_utils.py
│   ├── logging_utils.py
│   ├── os_utils.py
│   ├── torch_utils.py
│   └── __init__.py
│
├── evaluation/
├── requirements.txt
├── LICENSE
└── README.md
```

### Important source files

| File | Purpose |
|---|---|
| `models/DeepConvLSTM.py` | Unified implementation of all model variants |
| `models/continuous_conv.py` | Continuous temporal convolution operator |
| `models/train.py` | Training, validation, checkpointing, and metric generation |
| `scripts/run_loso.py` | Main LOSO experiment entry point |
| `utils/tools/prepare_dataset.py` | Unified preparation utility for all three datasets |
| `scripts/run_kernel_sweep_from_configs.py` | Fixed-kernel sweep runner |
| `evaluation/` | Aggregation and paper-figure utilities |

---

## 2. Installation

Python 3.10 or newer is recommended.

### Conda installation

```bash
git clone https://github.com/pavan1609/ContinuousConvLSTM.git
cd ContinuousConvLSTM

conda create -n continuousconv python=3.10 -y
conda activate continuousconv

pip install -r requirements.txt
```

Verify that the main modules import correctly:

```bash
python - <<'PY'
modules = [
    "models.DeepConvLSTM",
    "models.continuous_conv",
    "models.train",
    "scripts.run_loso",
    "scripts.run_kernel_sweep_from_configs",
]

for module in modules:
    __import__(module)
    print("OK:", module)
PY
```

Expected output:

```text
OK: models.DeepConvLSTM
OK: models.continuous_conv
OK: models.train
OK: scripts.run_loso
OK: scripts.run_kernel_sweep_from_configs
```

A complete syntax check can also be performed with:

```bash
python -m compileall -q models scripts utils evaluation
```

---

## 3. Dataset preparation

The datasets are not redistributed with this repository. Download each dataset from its original source and prepare it locally.

A single utility handles all three datasets:

```text
utils/tools/prepare_dataset.py
```

Supported dataset identifiers are:

```text
wear
rwhar
wisdm_watch
all
```

The preparation utility can perform four stages:

| Stage | Description |
|---|---|
| `streams` | Creates native-rate and lower-rate sensor streams |
| `annotations` | Generates LOSO annotation JSON files |
| `configs` | Refreshes dataset-dependent values in existing YAML configs |
| `validate` | Verifies generated folders, annotations, and config paths |
| `all` | Runs all stages in sequence |

Display all options:

```bash
python utils/tools/prepare_dataset.py --help
```

### Important behavior

The utility updates dataset-dependent configuration fields only, including:

```yaml
anno_json:

dataset:
  sens_folder:
  sampling_rate:
  window_size:
  input_dim:
  num_classes:

train_cfg:
  log_subdir:
```

It does not replace architecture or optimization settings such as:

```yaml
model:
  conv_type:
  temporal_head:
  conv_kernel_size:
  conv_rank:
  conv_mlp_hidden_dim:
  lstm_units:

train_cfg:
  epochs:
  batch_size:
  learning_rate:
  weight_decay:
```

This ensures that the manually curated experiment configurations remain intact.

---

## 4. Preparing WEAR

WEAR contains data from four limb-mounted inertial sensors. The native stream is sampled at 50 Hz.

Download WEAR from:

- https://mariusbock.github.io/wear/

The source directory supplied to the utility must contain the native subject CSV files:

```text
/path/to/wear/native_50hz/
├── sbj_0.csv
├── sbj_1.csv
├── sbj_2.csv
├── ...
└── sbj_21.csv
```

Run:

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --source /path/to/wear/native_50hz \
  --repo-root . \
  --steps all \
  --native-mode symlink \
  --overwrite
```

The generated layout is:

```text
data/wear/
├── 50hz/
│   ├── sbj_0.csv
│   └── ...
├── 25hz/
│   ├── sbj_0.csv
│   └── ...
├── 12hz/
│   ├── sbj_0.csv
│   └── ...
├── 6hz/
│   ├── sbj_0.csv
│   └── ...
├── multirate/
│   ├── sbj_0.csv
│   └── ...
└── annotations/
    ├── 50Hz/
    │   ├── loso_sbj_0.json
    │   └── ...
    ├── 25Hz/
    ├── 12Hz/
    ├── 6Hz/
    └── Multirate/
```

The expected number of WEAR LOSO files is:

```text
22
```

Validate the generated WEAR data:

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --repo-root . \
  --steps validate
```

---

## 5. Preparing RealWorld-HAR

The RealWorld-HAR experiments use waist accelerometer data from:

- 15 subjects;
- 8 activities;
- native sampling frequency of 50 Hz.

The source argument may point to either:

1. the original RealWorld-HAR download structure; or
2. a prepared folder containing activity recordings such as:

```text
proband1_walking.csv
proband1_running.csv
proband1_sitting.csv
proband2_walking.csv
...
```

Run:

```bash
python utils/tools/prepare_dataset.py \
  --dataset rwhar \
  --source /path/to/realworld/native_50hz \
  --repo-root . \
  --steps all \
  --native-mode symlink \
  --overwrite
```

The utility creates:

```text
data/rwhar/
├── 50hz/
├── 25hz/
├── 12hz/
├── 6hz/
└── multirate/
```

The LOSO annotation files are generated for all 15 held-out subjects.

Depending on the existing config schema, the annotation hierarchy is placed under the RealWorld-HAR annotation root referenced by the YAML files.

Validate:

```bash
python utils/tools/prepare_dataset.py \
  --dataset rwhar \
  --repo-root . \
  --steps validate
```

Expected LOSO split count:

```text
15
```

---

## 6. Preparing WISDM-watch

The WISDM-watch experiments use:

- wrist accelerometer data;
- 51 subjects;
- 18 activities;
- three acceleration channels;
- native sampling frequency of 20 Hz.

The source may be a prepared folder containing:

```text
sbj_0.csv
sbj_1.csv
...
sbj_50.csv
```

Run:

```bash
python utils/tools/prepare_dataset.py \
  --dataset wisdm_watch \
  --source /path/to/wisdm_watch/native_20hz \
  --repo-root . \
  --steps all \
  --native-mode symlink \
  --overwrite
```

The generated stream layout is:

```text
data/wisdm_watch/
├── 20hz/
├── 10hz/
├── 5hz/
└── multirate/
```

The LOSO annotations are generated under the annotation path referenced by the WISDM-watch YAML configs, for example:

```text
data/wisdm/annotations/Multirate/watch/
├── loso_sbj_0.json
├── loso_sbj_1.json
├── ...
└── loso_sbj_50.json
```

Expected LOSO split count:

```text
51
```

Validate:

```bash
python utils/tools/prepare_dataset.py \
  --dataset wisdm_watch \
  --repo-root . \
  --steps validate
```

---

## 7. Preparing all datasets with one command

All datasets can be prepared in one invocation:

```bash
python utils/tools/prepare_dataset.py \
  --dataset all \
  --wear-source /path/to/wear/native_50hz \
  --rwhar-source /path/to/realworld/native_50hz \
  --wisdm-source /path/to/wisdm_watch/native_20hz \
  --repo-root . \
  --steps all \
  --native-mode symlink \
  --overwrite
```

Use `--native-mode symlink` when the original processed data should remain outside the repository.

This avoids duplicating large datasets.

Use the corresponding copy mode only when a self-contained local data folder is required.

---

## 8. Running individual preparation stages

### Generate sensor streams only

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --source /path/to/wear/native_50hz \
  --repo-root . \
  --steps streams \
  --native-mode symlink \
  --overwrite
```

### Generate annotations only

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --repo-root . \
  --steps annotations \
  --overwrite
```

### Refresh config paths only

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --repo-root . \
  --steps configs
```

### Validate only

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --repo-root . \
  --steps validate
```

### Run multiple selected stages

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --source /path/to/wear/native_50hz \
  --repo-root . \
  --steps streams,annotations,validate \
  --native-mode symlink \
  --overwrite
```

---

## 9. Downsampling protocol

Lower-rate streams are generated by recursive, unfiltered stride-based decimation.

### WEAR and RealWorld-HAR

```text
Native stream: 50 Hz

50 Hz -> 25 Hz: retain every second sample
50 Hz -> 12 Hz: retain every fourth sample
50 Hz ->  6 Hz: retain every eighth sample
```

The exact effective frequencies produced by integer stride decimation are approximately:

```text
25.0 Hz
12.5 Hz
6.25 Hz
```

The repository retains the nominal names:

```text
25hz
12hz
6hz
```

These names match the paper, experiment configurations, log directories, and tables.

### WISDM-watch

```text
Native stream: 20 Hz

20 Hz -> 10 Hz: retain every second sample
20 Hz ->  5 Hz: retain every fourth sample
```

No anti-alias filtering is applied because the experiments intentionally reproduce the recursive sample-retention protocol used in the paper.

---

## 10. Window construction

All experiments use:

```text
window duration: 1 second
window overlap:  50%
```

Therefore, the nominal number of samples per window is:

| Rate | Window samples |
|---:|---:|
| 50 Hz | 50 |
| 25 Hz | 25 |
| 12 Hz | 12 |
| 6 Hz | 6 |
| 20 Hz | 20 |
| 10 Hz | 10 |
| 5 Hz | 5 |

Normalization statistics are calculated using training-subject data only and are then applied to validation/test data.

---

## 11. Configuration hierarchy

Each experiment is controlled by a YAML file.

Example:

```text
configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml
```

The configuration hierarchy separates:

- architecture;
- dataset;
- sampling rate;
- multirate or per-rate training;
- ablation type.

Example model section:

```yaml
name: continuousconvlstm

model:
  conv_type: continuous
  temporal_head: lstm
  multirate_training: true

  sampling_rates:
    - 50
    - 25
    - 12
    - 6

  conv_kernel_size: 9
  conv_rank: 8
  conv_mlp_hidden_dim: 32

  nb_filters: 64
  lstm_units: 128
  dropout: 0.5
```

Example dataset section:

```yaml
dataset:
  sens_folder: data/wear/multirate
  sampling_rate: 50
  window_size: 50
  input_dim: 12
  num_classes: 19
```

Example training section:

```yaml
train_cfg:
  epochs: 30
  batch_size: 100
  learning_rate: 0.0001
  weight_decay: 0.000001
  lr_decay: 0.9
  lr_step: 10
```

The exact values depend on the dataset and experiment.

---

## 12. Model variants

All paper architectures are implemented through one model class:

```text
models/DeepConvLSTM.py
```

The variant is selected with YAML keys.

### Standard DeepConvLSTM

```yaml
model:
  conv_type: standard
  temporal_head: lstm
  multirate_training: false
```

Architecture:

```text
fixed Conv2d blocks
        ↓
LSTM
        ↓
classifier
```

### Pure CNN

```yaml
model:
  conv_type: standard
  temporal_head: cnn
  multirate_training: false
```

Architecture:

```text
fixed Conv2d blocks
        ↓
temporal pooling
        ↓
classifier
```

### ContinuousConvLSTM

```yaml
model:
  conv_type: continuous
  temporal_head: lstm
  multirate_training: true
```

Architecture:

```text
rate-aware ContinuousConv branches
        ↓
shared LSTM
        ↓
shared classifier
```

### Single-branch ContinuousConv ablation

```yaml
model:
  conv_type: continuous_single
  temporal_head: lstm
  multirate_training: true
```

This variant is used to investigate whether one continuous branch can replace explicitly rate-aware branches.

### Standard multibranch ablation

```yaml
model:
  conv_type: standard_multibranch
  temporal_head: lstm
  multirate_training: true
```

This variant retains rate-specific standard discrete convolution branches.

---

## 13. Continuous convolution

The continuous operator is implemented in:

```text
models/continuous_conv.py
```

Instead of storing a fixed discrete temporal kernel directly, the operator learns a continuous kernel function evaluated at temporal coordinates.

Conceptually, the effective kernel length is determined by:

```text
sampling frequency
×
configured temporal support
```

A corresponding odd discrete support length is then used for the active rate.

The model can therefore evaluate related kernels at different sampling frequencies while preserving an approximately comparable physical-time receptive field.

The multirate architecture contains rate-aware branches inside one model object. All branches, the shared LSTM, and the classifier are stored in one checkpoint.

Therefore:

```text
multiple branches != multiple checkpoints
```

The multirate model produces one checkpoint per LOSO split.

---

## 14. Running experiments locally

The main entry point is:

```text
scripts/run_loso.py
```

### Run all LOSO splits

```bash
python scripts/run_loso.py \
  --config configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml \
  --seed 1
```

### Run one LOSO split

Split indices are one-based:

```bash
python scripts/run_loso.py \
  --config configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml \
  --seed 1 \
  --start_split 1 \
  --end_split 1
```

### Run a subset of splits

```bash
python scripts/run_loso.py \
  --config configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml \
  --seed 1 \
  --start_split 1 \
  --end_split 5
```

### Standard DeepConvLSTM example

```bash
python scripts/run_loso.py \
  --config configs/DeepConvLSTM/WEAR/wear_50hz.yaml \
  --seed 1 \
  --start_split 1 \
  --end_split 1
```

### Pure-CNN example

```bash
python scripts/run_loso.py \
  --config configs/PureCNN/WEAR/wear_50hz.yaml \
  --seed 1 \
  --start_split 1 \
  --end_split 1
```

---

## 15. Running experiments with SLURM

The `jobs/` hierarchy mirrors the `configs/` hierarchy.

For example:

```text
Config:
configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml

SLURM job:
jobs/ContinuousConvLSTM/WEAR/wear_multirate.slurm
```

Before submitting, open the job file and update the marked variables:

```bash
PROJECT_ROOT=/path/to/ContinuousConvLSTM
CONDA_ENV=continuousconv
```

Submit:

```bash
sbatch jobs/ContinuousConvLSTM/WEAR/wear_multirate.slurm
```

Monitor jobs:

```bash
squeue -u "$USER"
```

List the latest logs:

```bash
ls -lt slurm_logs | head
```

Follow the newest output:

```bash
tail -f "$(ls -t slurm_logs/*.out | head -1)"
```

Cancel a job:

```bash
scancel JOB_ID
```

A valid multirate run should report branch usage similar to:

```text
TRAIN BRANCH USAGE:
50hz 24.2%
25hz 26.8%
12hz 23.2%
6hz  25.8%
```

The exact percentages vary because sampling-rate selection is performed during training.

If a per-rate config is used, seeing one branch at 100% is expected.

For example:

```text
wear_25hz.yaml
```

should use:

```text
25hz 100%
```

That is not a multirate experiment.

---

## 16. Default training protocol

Unless overridden in a YAML config, the paper experiments use:

```text
Optimizer:             Adam
Learning rate:         1e-4
Weight decay:          1e-6
Epochs:                30
LR scheduler:          step decay
LR decay factor:       0.9
LR step interval:      10 epochs
Loss:                  class-weighted cross-entropy
Normalization:         per channel
Normalization source:  training subjects only
Evaluation:            LOSO
Primary metric:        macro F1
```

The random seed is passed through:

```bash
--seed 1
```

---

## 17. Output structure

Training outputs are written under:

```text
logs/
```

SLURM output and error files are written under:

```text
slurm_logs/
```

A typical output structure is:

```text
logs/
└── deepconvlstm/
    └── continuousconvlstm/
        └── wear/
            └── multirate/
                └── 50hz/
                    └── loso_sbj_0/
                        ├── best_loso_sbj_0.pth.tar
                        ├── best_macro_metrics_loso_sbj_0.csv
                        └── ...
```

Typical files include:

| File | Purpose |
|---|---|
| `best_loso_sbj_X.pth.tar` | Best model checkpoint for held-out subject X |
| `best_macro_metrics_loso_sbj_X.csv` | Aggregate metrics for that split |
| per-class metric CSVs | Per-activity precision, recall, and F1 |
| SLURM `.out` | Standard output |
| SLURM `.err` | Errors and warnings |

Runtime outputs are excluded from Git through `.gitignore`.

---

## 18. Kernel-sweep experiments

The fixed-kernel sensitivity experiments are launched through:

```text
scripts/run_kernel_sweep_from_configs.py
```

and the corresponding jobs under:

```text
jobs/KernelSweep/
```

Example:

```bash
sbatch jobs/KernelSweep/wear_kernel_sweep.slurm
```

The sweep evaluates discrete temporal kernels while keeping the remaining model and training protocol fixed.

Results can be collected with:

```bash
python evaluation/collect_kernel_sweep.py
```

The combined paper plot can be generated with:

```bash
python evaluation/plot_kernel_sweep_combined_3dataset_paper.py
```

---

## 19. Reproducing the experiment groups

### Standard DeepConvLSTM baselines

```text
configs/DeepConvLSTM/
jobs/DeepConvLSTM/
```

### Pure-CNN baselines

```text
configs/PureCNN/
jobs/PureCNN/
```

### ContinuousConvLSTM per-rate experiments

```text
configs/ContinuousConvLSTM/<DATASET>/*hz.yaml
jobs/ContinuousConvLSTM/<DATASET>/*hz.slurm
```

### ContinuousConvLSTM single-checkpoint experiments

```text
configs/ContinuousConvLSTM/<DATASET>/*multirate.yaml
jobs/ContinuousConvLSTM/<DATASET>/*multirate.slurm
```

### Single-branch ablation

```text
configs/Ablations/SingleBranch/
jobs/Ablations/SingleBranch/
```

### Kernel-size sweep

```text
jobs/KernelSweep/
scripts/run_kernel_sweep_from_configs.py
```

---

## 20. Evaluation

The `evaluation/` folder contains scripts for:

- averaging LOSO metrics;
- computing per-activity F1;
- selecting activity-dependent sampling rates;
- collecting kernel-sweep results;
- generating figure-ready CSV files;
- generating paper tables.

Examples:

```bash
python evaluation/aggregate_macro_metrics.py
```

```bash
python evaluation/make_per_activity_f1_table.py
```

```bash
python evaluation/aggregate_optimal_frequency.py
```

```bash
python evaluation/choose_frequency_tradeoff.py
```

```bash
python evaluation/collect_kernel_sweep.py
```

Exact arguments can be inspected with:

```bash
python evaluation/aggregate_macro_metrics.py --help
```

---

## 21. Reproducibility checklist

Before launching a complete experiment, verify the following.

### Check Python files

```bash
python -m compileall -q models scripts utils evaluation
```

### Check dataset utility

```bash
python utils/tools/prepare_dataset.py --help
```

### Validate the selected dataset

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --repo-root . \
  --steps validate
```

### Check annotation count

WEAR:

```bash
find data/wear/annotations/Multirate \
  -type f \
  -name "loso_sbj_*.json" \
  | wc -l
```

Expected:

```text
22
```

RealWorld-HAR expected:

```text
15
```

WISDM-watch expected:

```text
51
```

### Run one split

```bash
python scripts/run_loso.py \
  --config configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml \
  --seed 1 \
  --start_split 1 \
  --end_split 1
```

### Confirm model construction

The output should contain:

```text
ContinuousConv2d
```

for ContinuousConv configurations.

### Confirm multirate branch usage

A multirate configuration should eventually report non-zero usage for all supported rates.

### Confirm output checkpoint

```bash
find logs -type f -name "best_loso_sbj_*.pth.tar" | head
```

---

## 22. Troubleshooting

### `FileNotFoundError: data/.../sbj_X.csv`

The `sens_folder` in the selected YAML does not point to the prepared dataset.

Inspect it with:

```bash
grep -n "sens_folder" path/to/config.yaml
```

Then validate the dataset:

```bash
python utils/tools/prepare_dataset.py \
  --dataset wear \
  --repo-root . \
  --steps validate
```

### YAML loads as `None`

The YAML file is empty or corrupted.

Check:

```bash
ls -lh path/to/config.yaml
head -40 path/to/config.yaml
```

Restore a tracked config with:

```bash
git restore path/to/config.yaml
```

### Training is very slow on a login node

Do not train large LOSO experiments directly on a cluster login node.

Use:

```bash
sbatch jobs/ContinuousConvLSTM/WEAR/wear_multirate.slurm
```

### Only one branch is used

Check which config was launched.

A per-rate config such as:

```text
wear_25hz.yaml
```

is expected to report:

```text
25hz 100%
```

A multirate config such as:

```text
wear_multirate.yaml
```

should distribute training batches across the supported rates.

### No SLURM log is found

Do not guess the log filename.

Use:

```bash
ls -lt slurm_logs | head
```

Then:

```bash
tail -f "$(ls -t slurm_logs/*.out | head -1)"
```

### Import error involving utility modules

Run:

```bash
python - <<'PY'
import models.train
import scripts.run_loso
import utils.torch_utils
print("Imports successful")
PY
```

### CUDA is unavailable

Check:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
```

On a cluster, CUDA may only be visible inside an allocated GPU job.

---

## 23. Notes on the single-checkpoint model

The term **single checkpoint** means that one saved model contains:

```text
rate-aware convolution branches
shared LSTM
shared classifier
```

For WEAR and RealWorld-HAR, the checkpoint contains branches for:

```text
50 Hz
25 Hz
12 Hz
6 Hz
```

For WISDM-watch:

```text
20 Hz
10 Hz
5 Hz
```

This is different from training one independent checkpoint for every rate.

The complete multirate model is serialized into one:

```text
best_loso_sbj_X.pth.tar
```

file for each LOSO fold.

---

Update the BibTeX entry with the final proceedings metadata when available.

---

## 25. License

The source code is released under the license provided in:

```text
LICENSE
```

The datasets are not included and remain subject to their original licenses and terms of use.
