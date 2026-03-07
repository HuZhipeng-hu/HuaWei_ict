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
- Same source CSV file never crosses train/val/test in grouped mode
- Data augmentation only applies to train split
- Model selection uses val split
- Final KPI uses test split only
- Reproducibility is enforced by split manifest

## Train

```bash
python -m training.train --config configs/training.yaml --data_dir ../data
```

If `training.split_manifest_path` is configured but the manifest file does not
exist yet, training will auto-generate it and save it to the configured path
(or to `--split_manifest_out` if explicitly provided).

Useful split options:

```bash
python -m training.train \
  --config configs/training.yaml \
  --data_dir ../data \
  --split_mode grouped_file \
  --test_ratio 0.2 \
  --split_manifest_out artifacts/splits/default_split_manifest.json
```

## Evaluate (checkpoint + manifest)

```bash
python scripts/evaluate_ckpt.py \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --split_manifest artifacts/splits/default_split_manifest.json \
  --output_dir logs/eval_recheck
```

## Convert

```bash
python -m conversion.convert \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --output models/neurogrip \
  --config configs/conversion.yaml
```

## Runtime

```bash
python -m runtime.run --config configs/runtime.yaml
```

Standalone smoke test:

```bash
python -m runtime.run --standalone --max_cycles 200
```

## Realtime Checkpoint Debug (Armband)

```bash
python scripts/realtime_ckpt.py \
  --port /dev/ttyUSB0 \
  --device CPU \
  --ckpt checkpoints/neurogrip_best.ckpt \
  --threshold 0.6 \
  --infer_rate_hz 20
```

`infer_rate_hz=0` keeps old behavior (no inference rate limit).

## Tests

```bash
python -m pytest -q
```

## More Details

See [docs/evaluation_retrain_runbook.md](docs/evaluation_retrain_runbook.md) for
full retrain -> conversion -> deployment -> realtime retest procedure.
