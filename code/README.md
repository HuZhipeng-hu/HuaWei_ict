# NeuroGrip Pro V2

AI-driven sEMG gesture recognition and prosthesis control stack.

## Repository Layout

```text
code/
├── shared/        # shared model/config/preprocess definitions
├── training/      # model training and offline evaluation
├── conversion/    # checkpoint -> mindir export/quantization
├── runtime/       # deployment/runtime control loop
├── scripts/       # utility entry scripts (realtime, preflight, evaluate)
├── configs/       # YAML configs
└── tests/         # unit/integration tests
```

## 6-Class Gestures

`RELAX`, `FIST`, `PINCH`, `OK`, `YE`, `SIDEGRIP`

## Trusted Train/Eval Protocol

- Split mode default: `grouped_file`
- Fixed split manifest: `artifacts/splits/default_split_manifest.json`
- Manifest v2 groups by `user::session::recording` when metadata is available
- Data augmentation only applies to train split
- Training/evaluation/conversion/realtime benchmark all emit run-scoped artifacts under `artifacts/runs/<run_id>/`
- Final KPI requires both offline test metrics and trusted realtime benchmark results

## Train

```bash
python -m training.train \
  --config configs/training.yaml \
  --data_dir ../data \
  --run_id 20260308_baseline
```

If `data.split_manifest_path` is configured but the manifest file does not
exist yet, training will auto-generate it and save it to the configured path.
Default path: `artifacts/splits/default_split_manifest.json`.

Current default training config enables dual-branch features (200Hz + 1000Hz)
with fused input shape `(12, 24, 6)`. This is a breaking change versus legacy
single-branch `(6, 24, 6)` models.

Useful experiment overrides:

```bash
python -m training.train \
  --config configs/training.yaml \
  --data_dir ../data \
  --run_id 20260308_lite_b24 \
  --model_type lite \
  --base_channels 24 \
  --use_se false \
  --loss_type focal \
  --hard_mining_ratio 0.5 \
  --augment_factor 3 \
  --use_mixup true
```

Optional metadata manifest:

- Place `recordings_manifest.csv` under the data root, or pass `--recordings_manifest <path>`.
- Preferred columns: `relative_path,gesture,user_id,session_id,device_id,timestamp,wearing_state`.

## Evaluate (checkpoint + manifest)

```bash
python scripts/evaluate_ckpt.py \
  --config configs/training.yaml \
  --data_dir ../data \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --split_manifest artifacts/splits/default_split_manifest.json \
  --run_id 20260308_eval_baseline
```

## Convert

```bash
python -m conversion.convert \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --run_id 20260308_convert_baseline \
  --config configs/conversion.yaml
```

`configs/conversion.yaml` now defaults to `input_shape: [1, 12, 24, 6]`.
Legacy 6-channel checkpoints/mindir are not compatible with this config and
must be retrained/re-converted.

## Runtime

```bash
python -m runtime.run --config configs/runtime.yaml
```

Runtime preprocess is aligned with training dual-branch config. Ensure deployed
`.mindir` was exported with matching shape `(1, 12, 24, 6)`.

Standalone smoke test:

```bash
python -m runtime.run --standalone --max_cycles 200
```

## Trusted Realtime Benchmark

Use this as the acceptance path for realtime checkpoint replay. It reuses the
locked test manifest and writes structured outputs into the run directory.

```bash
python scripts/benchmark_realtime_ckpt.py \
  --training_config configs/training.yaml \
  --runtime_config configs/runtime.yaml \
  --data_dir ../data \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --split_manifest artifacts/splits/default_split_manifest.json \
  --run_id 20260308_rt_baseline
```

Local non-MindSpore smoke test:

```bash
python scripts/benchmark_realtime_ckpt.py \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --split_manifest artifacts/splits/default_split_manifest.json \
  --data_dir ../data \
  --mock \
  --run_id local_benchmark_smoke
```

## Legacy Realtime Checkpoint Debug (Armband)

```bash
python scripts/realtime_ckpt.py \
  --port /dev/ttyUSB0 \
  --device CPU \
  --ckpt checkpoints/neurogrip_best.ckpt \
  --threshold 0.6 \
  --infer_rate_hz 20
```

`infer_rate_hz=0` keeps old behavior (no inference rate limit).
This script is for manual debugging only; it is not the trusted benchmark path.

## Experiment Matrix

```bash
python scripts/run_experiment_matrix.py \
  --config configs/training.yaml \
  --data_dir ../data \
  --manifest artifacts/splits/default_split_manifest.json
```

This generates a structured command matrix in `artifacts/experiment_matrix.md`.

## Tests

```bash
python -m pytest -q
```

## More Details

See [docs/evaluation_retrain_runbook.md](docs/evaluation_retrain_runbook.md) for
full retrain -> conversion -> deployment -> realtime retest procedure.
