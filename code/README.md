# NeuroGrip Cloud-to-PI Pipeline (Core 4 Chain)

This repository now uses a single production path:

- Cloud (Huawei Cloud): pretrain, finetune, convert
- PI device: runtime inference and actuation only
- App side (out of scope here): upload wearer data to cloud and return converted model to PI

## Responsibility Split

### Cloud side (this repo)
1. DB5 foundation pretraining
2. Wearer finetuning (using uploaded private clips)
3. CKPT -> MindIR conversion

### PI side (this repo)
1. Load MindIR + metadata
2. Read armband stream
3. Trigger actuator by predicted class/state

### App side (not in this repo)
1. Upload private finetune data to cloud
2. Download converted model package and deliver to PI

## Core 4 Entrypoints

- Pretrain: `scripts/pretrain_db5_repr_method_matrix.py`
- Finetune: `scripts/finetune_event_onset.py`
- Convert: `scripts/convert_event_onset.py`
- Runtime: `scripts/run_event_runtime.py`

## Cloud Commands

### 1) Pretrain (public DB5 only)

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

### 2) Finetune (wearer private data)

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
  --pretrained_emg_checkpoint <foundation_ckpt_path> \
  --run_root artifacts/runs \
  --run_id event_finetune_v1
```

### 3) Convert to MindIR

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_root artifacts/runs \
  --run_id event_convert_v1
```

## PI Runtime Command

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend lite
```

Standalone smoke:

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --backend lite --standalone --duration_sec 10
```

## Key Artifacts

Cloud outputs:
- Foundation checkpoint
- Finetuned event checkpoint
- `models/event_onset.mindir`
- `models/event_onset.model_metadata.json`
- run summaries under `artifacts/runs/*`

PI inputs:
- `event_onset.mindir`
- `event_onset.model_metadata.json`
- `configs/runtime_event_onset.yaml`
- `configs/event_actuation_mapping.yaml`

## Preflight

```bash
python scripts/preflight.py --mode local --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
python scripts/preflight.py --mode ascend --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
```

## Repository Hygiene

Generated run outputs and temp caches are not source-of-truth and should not be committed:
- `code/artifacts/runs/**`
- `code/artifacts/splits/*.json`
- `**/__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.tmp_pytest*`
