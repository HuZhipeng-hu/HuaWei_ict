# Evaluation + Retrain Closed Loop Runbook

This runbook defines the trusted workflow for 6-class retraining and deployment.

Note: current default configs use dual-branch fused features with model input
shape `(1, 16, 24, 6)`. Legacy single-branch `(1, 6, 24, 6)` and legacy
dual-branch `(1, 12, 24, 6)` artifacts are not
compatible and must be retrained/re-converted.

## 1) Build/Reuse Split Manifest (no file leakage)

Preferred: create once, then reuse for all model comparisons.

On first run, if `training/data.split_manifest_path` is configured but missing,
`training.train` will auto-generate and persist a manifest at that path
(or at `--split_manifest_out` if provided).

```bash
python -m training.train \
  --config configs/training.yaml \
  --data_dir ../data \
  --split_mode grouped_file \
  --test_ratio 0.2 \
  --split_manifest_out artifacts/splits/default_split_manifest.json
```

If a manifest already exists, lock it:

```bash
python -m training.train \
  --config configs/training.yaml \
  --data_dir ../data \
  --split_manifest_in artifacts/splits/default_split_manifest.json
```

## 2) Train (train-only augmentation, model selection on val)

```bash
python -m training.train \
  --config configs/training.yaml \
  --data_dir ../data \
  --split_manifest_in artifacts/splits/default_split_manifest.json
```

Artifacts:
- best checkpoint: `checkpoints/neurogrip_best.ckpt`
- training history: `logs/training_history.csv`
- test report: `logs/evaluation/test_metrics.json`
- per-class table: `logs/evaluation/test_per_class_metrics.csv`
- confusion matrix: `logs/evaluation/test_confusion_matrix.csv`

## 3) Independent Evaluation Entry (checkpoint + manifest)

```bash
python scripts/evaluate_ckpt.py \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --split_manifest artifacts/splits/default_split_manifest.json \
  --output_dir logs/eval_recheck
```

Use this for reproducibility audits and side-by-side model comparisons.

## 4) Convert + Deploy

```bash
python -m conversion.convert \
  --checkpoint checkpoints/neurogrip_best.ckpt \
  --output models/neurogrip \
  --config configs/conversion.yaml
```

Deploy `models/neurogrip.mindir` and runtime config to Orange Pi.

## 5) Realtime Retest on Orange Pi (CPU)

Baseline manual retest at 20Hz (acceptance still uses the trusted realtime benchmark):

```bash
python scripts/realtime_ckpt.py \
  --runtime_config configs/runtime.yaml \
  --training_config configs/training.yaml \
  --port /dev/ttyUSB0 \
  --device CPU \
  --ckpt checkpoints/neurogrip_best.ckpt \
  --threshold 0.6 \
  --infer_rate_hz 20
```

A/B retest for scheduling:
- profile A: `--infer_rate_hz 20`
- profile B: `--infer_rate_hz 15`

Optional fallback:
- if misses increase too much at 15Hz, test `--infer_rate_hz 25`

## 6) Acceptance Checklist (must all pass)

- [ ] split manifest fixed and versioned
- [ ] grouped_file split has no cross-set source overlap
- [ ] augmentation is applied to train only (never val/test)
- [ ] final KPI is reported on test only
- [ ] evaluation reproducible with same checkpoint + same manifest
- [ ] deployment retest includes hit/false-trigger/latency p50,p95 records

## 7) Recommended Realtime Parameter Table

| Scenario | threshold | vote_window | vote_min | infer_rate_hz |
|---|---:|---:|---:|---:|
| first deployment retest | 0.60 | 5 | 3 | 20 |
| reduce CPU load | 0.60 | 5 | 3 | 15 |
| recover missed gestures | 0.60 | 5 | 3 | 25 |

Notes:
- `infer_rate_hz=0` means no rate limit (backward-compatible behavior).
- Keep threshold/voting tunable in config/CLI; do not hard-code in logic.


