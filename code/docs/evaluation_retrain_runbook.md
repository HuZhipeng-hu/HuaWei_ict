# Event-Onset Retrain + Deploy Runbook

This runbook defines the production workflow for the event-onset branch.

## 1) DB5 Pretrain

```bash
python scripts/pretrain_ninapro_db5.py \
  --config configs/pretrain_ninapro_db5.yaml \
  --data_dir ../data_ninaproDB5 \
  --run_id db5_pretrain_v1
```

## 2) Wearer Finetune

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --pretrained_emg_checkpoint artifacts/runs/db5_pretrain_v1/checkpoints/db5_pretrain_best.ckpt \
  --run_id event_finetune_v1
```

## 3) Evaluate Checkpoint

```bash
python scripts/evaluate_ckpt.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --split_manifest artifacts/splits/event_onset_split_manifest.json
```

## 4) Convert to MindIR

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_id event_convert_v1
```

Conversion outputs:

- `models/event_onset.mindir`
- `models/event_onset.model_metadata.json`

## 5) Runtime Benchmark (CKPT vs Lite)

```bash
python scripts/benchmark_event_runtime_ckpt.py \
  --training_config configs/training_event_onset.yaml \
  --runtime_config configs/runtime_event_onset.yaml \
  --data_dir ../data \
  --backend both \
  --output artifacts/event_runtime_benchmark_compare.json
```

## 6) Deploy and Run on Orange Pi

Copy `models/event_onset.mindir`, `models/event_onset.model_metadata.json`, and
`configs/runtime_event_onset.yaml` to target machine.

Production run:

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite
```

Debug run (non-production):

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend ckpt
```

## 7) Merge Gate

Before merging to `master`, verify:

- `transition_hit_rate_mindir >= transition_hit_rate_ckpt - 0.03`
- `false_trigger_rate_mindir <= false_trigger_rate_ckpt + 0.03`
- `state_hold_accuracy_mindir >= state_hold_accuracy_ckpt - 0.03`
- `release_accuracy_mindir >= release_accuracy_ckpt - 0.03`
- `latency_p95_mindir <= latency_p95_ckpt * 1.30`
