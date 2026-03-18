# Handoff Runbook (Demo3 Release)

This runbook is the deployment handoff baseline for the demo3 release path.

## 1) Cloud: Finetune

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
  --run_root artifacts/runs \
  --run_id event_finetune_v1
```

## 2) Cloud: Convert

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_root artifacts/runs \
  --run_id event_convert_v1
```

Before deployment, validate that `CKPT` and converted `MindIR/Lite` stay within the release parity band on the same split and runtime config.

## 3) PI: Runtime

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite
```

Standalone smoke:

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite --standalone --duration_sec 10
```

## 4) Optional Evaluation

```bash
python scripts/evaluate_ckpt.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt

python scripts/benchmark_event_runtime_ckpt.py \
  --training_config configs/training_event_onset.yaml \
  --runtime_config configs/runtime_event_onset.yaml \
  --data_dir ../data \
  --backend both \
  --output artifacts/event_runtime_benchmark_compare.json
```

## 5) Scope Boundary

- This repo: collection, training, conversion, runtime, evaluation
- Experimental DB5 pretrain and algorithm baselines: `experimental/`
- App side: data upload/download is out of scope
