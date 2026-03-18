# Evaluation and Retrain Runbook (Demo3 Release)

## Workflow

1. collect wearer data
2. run `prepare`
3. finetune demo3 model
4. convert to MindIR + metadata
5. validate `ckpt` and `lite` control behavior
6. deploy runtime

The release default runtime thresholds are the tuned demo3 thresholds already written into `configs/runtime_event_onset.yaml`.

## Commands

### Prepare

```bash
python scripts/train_event_model_90_sprint.py \
  --stage prepare \
  --device_target CPU \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --run_prefix s2_model90
```

### Finetune

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
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
python scripts/preflight.py --mode local --wearer_data_dir ../data
python scripts/preflight.py --mode ascend --wearer_data_dir ../data
```

Optional:

```bash
python scripts/evaluate_ckpt.py --config configs/training_event_onset.yaml --data_dir ../data --checkpoint <event_best_ckpt>
python scripts/benchmark_event_runtime_ckpt.py --training_config configs/training_event_onset.yaml --runtime_config configs/runtime_event_onset.yaml --data_dir ../data --backend both --output artifacts/event_runtime_benchmark_compare.json
```

## Notes

- Release authority comes from the demo3 mainline only.
- `experimental/` contains retired or ablation-only branches.
- PI side only consumes the converted model package for inference.
