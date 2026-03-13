# Handoff Runbook (Cloud + PI)

This runbook is the deployment handoff baseline.

## 1) Cloud: Pretrain

```bash
python scripts/pretrain_db5_repr_method_matrix.py \
  --pretrain_config configs/pretrain_ninapro_db5.yaml \
  --db5_data_dir ../data_ninaproDB5 \
  --run_root artifacts/runs \
  --run_prefix db5_repr_stage2_v1 \
  --device_target Ascend \
  --device_id 0 \
  --fewshot_mode off
```

## 2) Cloud: Finetune

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
  --pretrained_emg_checkpoint <foundation_ckpt_path> \
  --run_root artifacts/runs \
  --run_id event_finetune_v1
```

## 3) Cloud: Convert

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_root artifacts/runs \
  --run_id event_convert_v1
```

## 4) PI: Runtime

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite
```

Standalone smoke:

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite --standalone --duration_sec 10
```

## 5) Optional Evaluation

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

## 6) Scope Boundary

- This repo: model training/conversion/runtime
- App side: data upload to cloud and model return to PI
