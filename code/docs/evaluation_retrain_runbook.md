# Evaluation and Retrain Runbook (Core 4 Chain)

## Workflow

1. Cloud pretrain (DB5 public)
2. Cloud finetune (wearer private)
3. Cloud convert (MindIR + metadata)
4. PI runtime deployment

## Commands

### Pretrain

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

### Finetune

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
  --pretrained_emg_checkpoint <foundation_ckpt_path> \
  --run_root artifacts/runs \
  --run_id event_finetune_v1
```

### Convert

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_root artifacts/runs \
  --run_id event_convert_v1
```

### Runtime

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite
```

## Quality Checks

```bash
python scripts/preflight.py --mode local --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
python scripts/preflight.py --mode ascend --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
```

Optional:

```bash
python scripts/evaluate_ckpt.py --config configs/training_event_onset.yaml --data_dir ../data --checkpoint <event_best_ckpt>
python scripts/benchmark_event_runtime_ckpt.py --training_config configs/training_event_onset.yaml --runtime_config configs/runtime_event_onset.yaml --data_dir ../data --backend both --output artifacts/event_runtime_benchmark_compare.json
```

## Notes

- Cloud side outputs are authoritative artifacts.
- PI side only consumes converted model package for inference.
- App transport (upload/download) is out of scope for this codebase.
