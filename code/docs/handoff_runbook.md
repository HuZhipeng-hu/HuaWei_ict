# Event-Onset Handoff Runbook

This runbook is for handing off the event-onset production pipeline to a target machine.

## 1) Environment

- Python >= 3.8
- Required: `numpy`, `scipy`, `pyyaml`
- Target deployment: `mindspore`, `mindspore_lite`
- Optional hardware libs: `pyserial`, `smbus2`

## 2) Preflight

Local:

```powershell
python scripts/preflight.py --mode local
```

Target:

```powershell
python scripts/preflight.py --mode ascend
```

## 3) Training + Conversion Chain

```powershell
python scripts/pretrain_ninapro_db5.py --config configs/pretrain_ninapro_db5.yaml --data_dir ../data_ninaproDB5 --run_id db5_pretrain_v1
python scripts/finetune_event_onset.py --config configs/training_event_onset.yaml --data_dir ../data --pretrained_emg_checkpoint artifacts/runs/db5_pretrain_v1/checkpoints/db5_pretrain_best.ckpt --run_id event_finetune_v1
python scripts/convert_event_onset.py --config configs/conversion_event_onset.yaml --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt --run_id event_convert_v1
```

## 4) Runtime Benchmark

```powershell
python scripts/benchmark_event_runtime_ckpt.py --training_config configs/training_event_onset.yaml --runtime_config configs/runtime_event_onset.yaml --data_dir ../data --backend both --output artifacts/event_runtime_benchmark_compare.json
```

## 5) Runtime Startup

Standalone smoke:

```powershell
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite --standalone --duration_sec 10
```

Real hardware:

```powershell
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite
```

## 6) Deployment Artifacts

- `models/event_onset.mindir`
- `models/event_onset.model_metadata.json`
- `configs/runtime_event_onset.yaml`
