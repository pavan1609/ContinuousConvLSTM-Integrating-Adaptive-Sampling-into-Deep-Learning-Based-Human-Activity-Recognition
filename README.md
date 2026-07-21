# ContinuousConvLSTM: Integrating Adaptive Sampling into Deep Learning-Based Human Activity Recognition

Code release for the paper *"Integrating Adaptive Sampling into Deep Learning-Based
Human Activity Recognition"* (IWOAR 2026).

A single-checkpoint **ContinuousConvLSTM** replaces the discrete temporal kernels of
DeepConvLSTM with parametric continuous kernels whose effective length
`k_t = odd(max(1, round(tau * f_s)))` is derived from the operating sampling rate,
so one model covers 50/25/12/6 Hz (WEAR, RealWorld-HAR) or 20/10/5 Hz (WISDM-watch)
without per-rate kernel tuning or per-rate checkpoints.

## Repository layout

```text
configs/            One YAML per (architecture, dataset, rate) — see configs/README.md
  DeepConvLSTM/       standard conv + LSTM head (per-rate baselines + multirate)
  PureCNN/            standard conv + temporal-pooling CNN head (no LSTM)
  ContinuousConvLSTM/ continuous kernels + LSTM head (per-rate + single-checkpoint multirate)
  Ablations/          single-branch ContinuousConv ablation (Appendix C)
jobs/               SLURM launchers, one per config, mirroring configs/ exactly
models/             Model + training code (single source of truth, no duplicates)
scripts/            Entry points: LOSO runner, kernel sweep, RealWorld-HAR preprocessing
utils/              Data pipeline, logging, torch helpers; utils/tools for annotations
evaluation/         Scripts that aggregate logs into the paper's tables and figures
results/            CSV/TeX provenance for the LSTM-vs-CNN tables in the paper
```

## Installation

```bash
conda create -n gamma python=3.10 -y
conda activate gamma
pip install -r requirements.txt
```

## Data preparation

Datasets are not redistributed; download them from their original sources.

- **WEAR** (4 limb IMUs, 22 subjects, 18 activities + null, 50 Hz):
  download from the [WEAR benchmark](https://mariusbock.github.io/wear/), then build
  the multirate streams and LOSO annotations with `utils/tools/prepare_multirate_wear.py`
  and `utils/tools/make_annotations.py`.
- **RealWorld-HAR** (waist accelerometer, 15 subjects, 8 activities, 50 Hz):
  `scripts/preprocess_realworld_waist_accel.py` followed by
  `scripts/create_realworld_loso_annotations.py`.
- **WISDM-watch** (wrist accelerometer, 51 subjects, 18 activities, 20 Hz, 3 channels):
  `utils/tools/make_annotations_wisdm_watch.py`.

Each config expects `data/<dataset>/<rate-or-multirate>` sensor folders and
`data/<dataset>/annotations/<rate-or-multirate>/loso_sbj_*.json` LOSO splits;
see `configs/README.md` for the exact schema. Lower-rate streams are produced by
recursively retaining every second sample (unfiltered decimation), with one-second
windows at 50% overlap throughout.

## Running experiments

Everything runs through one entry point:

```bash
python scripts/run_loso.py --config configs/ContinuousConvLSTM/WEAR/wear_multirate.yaml --seed 1
```

`--start_split N --end_split N` runs a single LOSO fold (1-based), which is what the
SLURM array jobs do:

```bash
sbatch jobs/ContinuousConvLSTM/WEAR/wear_multirate.slurm
```

Every job file has a two-line `EDIT ME` block (project root, conda env) and otherwise
needs no changes. Training uses Adam (lr 1e-4, weight decay 1e-6, step decay 0.9 every
10 epochs), 30 epochs, class-weighted cross-entropy, per-channel normalisation from
training statistics only.

## Reproducing the paper's claims

| Paper artifact | Train with | Aggregate with |
|---|---|---|
| Table 1 / Sec. 4 — per-rate DeepConvLSTM vs. pure CNN (WEAR) | `jobs/DeepConvLSTM/WEAR/*hz.slurm`, `jobs/PureCNN/WEAR/*hz.slurm` | `evaluation/standard_cnn_tables/`, `evaluation/standard_cnn_nolstm_tables/`, `evaluation/lstm_vs_cnn_tables/` |
| Fig. 4 — discrete kernel sweep vs. ContinuousConv (all datasets) | `jobs/KernelSweep/*.slurm` + `jobs/ContinuousConvLSTM/*/[rate].slurm` | `evaluation/collect_kernel_sweep.py`, `evaluation/plot_kernel_sweep_combined_3dataset_paper.py` |
| Table 4 / RQ2 — single-checkpoint ContinuousConvLSTM across rates | `jobs/ContinuousConvLSTM/{WEAR,RWHAR,WISDM_WATCH}/*_multirate.slurm` | `evaluation/continuous_single_no_gamma_tables/`, `evaluation/aggregate_macro_metrics.py` |
| Appendix A — LSTM-vs-CNN per activity (RealWorld-HAR, WISDM-watch) | `jobs/DeepConvLSTM/{RWHAR,WISDM_WATCH}/*hz.slurm`, `jobs/PureCNN/{RWHAR,WISDM_WATCH}/*hz.slurm` | `evaluation/lstm_vs_cnn_tables/collect_lstm_vs_cnn_*_from_logs.py` |
| Appendix C — single-branch vs. multi-branch | `jobs/Ablations/SingleBranch/*.slurm` | `evaluation/aggregate_macro_metrics.py` |
| Optimal-rate-per-activity analysis | (uses the runs above) | `evaluation/aggregate_optimal_frequency.py`, `evaluation/choose_frequency_tradeoff.py` |

Model variants are selected purely by config keys — `conv_type`
(`standard`, `standard_multibranch`, `continuous`, `continuous_single`),
`temporal_head` (`lstm`, `cnn`) and `multirate_training` — all handled by
`models/DeepConvLSTM.py`. The continuous-kernel operator lives in
`models/continuous_conv.py`; its temporal support defaults to
`conv_kernel_size / sampling_rate` (0.18 s for the 50 Hz datasets, 0.45 s for
WISDM-watch), matching the paper.

## License

Released under CC0 1.0 (see `LICENSE`). WEAR, RealWorld-HAR, and WISDM retain their
original dataset licenses.
