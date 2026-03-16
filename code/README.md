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

Optional dual-track tools (offline compare only):
- Algo train/export: `scripts/train_event_algo_baseline.py`
- Model vs Algo A/B: `scripts/evaluate_event_dualtrack.py`

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

## Wearer Data Collection (Live Armband)

Use this to collect one event clip directly from the armband and append `recordings_manifest.csv`:

```bash
python scripts/collect_event_data.py \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --target_state V_SIGN \
  --start_state RELAX \
  --user_id demo_user \
  --session_id s1 \
  --device_id armband01 \
  --wearing_state normal \
  --duration_sec 3 \
  --port COM4 \
  --baudrate 115200
```

Collector cue logic for action labels: keep `RELAX` for ~0.4s, then perform target action quickly and hold ~1.2s, then relax until clip end.

Continuous capture + auto-slice (higher throughput, recommended for manual collection):

```bash
python scripts/collect_event_data_continuous.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --target_state V_SIGN \
  --start_state RELAX \
  --user_id demo_user \
  --session_id s1 \
  --device_id armband01 \
  --wearing_state normal \
  --duration_sec 45 \
  --clip_duration_sec 3 \
  --pre_roll_ms 500 \
  --port COM5 \
  --baudrate 115200 \
  --save_stream_csv
```

## PI Runtime Command

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --recognizer_backend model \
  --backend lite
```

Standalone smoke:

```bash
python scripts/run_event_runtime.py --config configs/runtime_event_onset.yaml --recognizer_backend model --backend lite --standalone --duration_sec 10
```

Algorithm backend (startup single-select, no runtime switching):

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --recognizer_backend algo \
  --algo_model_path artifacts/runs/event_algo_baseline/models/algo_model.json
```

If selected backend artifact is missing, runtime exits immediately (no fallback).

## Dual-track Offline A/B (same split, same control settings)

Train/export algorithm recognizer:

```bash
python scripts/train_event_algo_baseline.py \
  --config configs/training_event_onset_demo_p0.yaml \
  --data_dir ../data \
  --recordings_manifest s2_train_manifest_relax12.csv \
  --split_manifest artifacts/splits/s2_stable4_relax12_seed77.json \
  --run_root artifacts/runs \
  --run_id s2_algo_baseline_seed77
```

Evaluate model vs algo:

```bash
python scripts/evaluate_event_dualtrack.py \
  --run_root artifacts/runs \
  --model_run_id s2_stable4_relax12_seed77 \
  --algo_model_path artifacts/runs/s2_algo_baseline_seed77/models/algo_model.json \
  --training_config configs/training_event_onset_demo_p0.yaml \
  --runtime_config configs/runtime_event_onset_demo_latch.yaml \
  --data_dir ../data \
  --recordings_manifest s2_train_manifest_relax12.csv \
  --split_manifest artifacts/splits/s2_stable4_relax12_seed77.json
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
