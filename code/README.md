# NeuroGrip Event-Onset Pipeline

This branch is event-onset only. Legacy 6-gesture standalone training/runtime flow has been removed.

## Pipeline

`DB5 pretrain -> wearer finetune -> MindIR conversion -> Orange Pi runtime`

## 1) DB5 Pretrain

```bash
python scripts/pretrain_ninapro_db5.py \
  --config configs/pretrain_ninapro_db5.yaml \
  --data_dir ../data_ninaproDB5 \
  --run_id db5_pretrain_v1
```

Output checkpoint (example):

`artifacts/runs/db5_pretrain_v1/checkpoints/db5_pretrain_best.ckpt`

## 2) Wearer Finetune (Event-Onset)

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --pretrained_emg_checkpoint artifacts/runs/db5_pretrain_v1/checkpoints/db5_pretrain_best.ckpt \
  --run_id event_finetune_v1
```

Output checkpoint (example):

`artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt`

## 3) Convert Event Model to MindIR

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_id event_convert_v1
```

Outputs:

- `models/event_onset.mindir`
- `models/event_onset.model_metadata.json`

## 4) Runtime (Orange Pi)

Production backend (`lite`):

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend lite
```

Debug backend (`ckpt`):

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend ckpt
```

## 5) Runtime Benchmark (CKPT vs MindIR)

```bash
python scripts/benchmark_event_runtime_ckpt.py \
  --training_config configs/training_event_onset.yaml \
  --runtime_config configs/runtime_event_onset.yaml \
  --data_dir ../data \
  --backend both \
  --output artifacts/event_runtime_benchmark_compare.json
```

## 6) Checkpoint Evaluation

```bash
python scripts/evaluate_ckpt.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --split_manifest artifacts/splits/event_onset_split_manifest.json
```

## 7) Preflight

```bash
python scripts/preflight.py --mode local
python scripts/preflight.py --mode ascend
```
