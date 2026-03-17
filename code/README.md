# NeuroGrip Cloud-to-PI Pipeline (Core 4 Chain)

This repository now uses a single recommended production path:

- Cloud side: foundation pretrain, wearer finetune, CKPT -> MindIR conversion
- PI side: runtime inference and actuation only
- App side: out of scope in this repo

## Recommended Control Contract

This project targets event-driven latched control, not continuous motion mirroring.

- `CONTINUE`: no new command, keep the current prosthesis state latched
- `TENSE_OPEN`: explicit open/release command
- Default demo actions are `THUMB_UP` and `WRIST_CW`
- Other labels such as `WRIST_CCW`, `V_SIGN`, and `OK_SIGN` are extension paths, not the default release-candidate demo

The intended user experience is:

1. wearer produces an onset intent
2. model recognizes the onset class
3. prosthesis switches to the target gesture
4. prosthesis keeps that state until a new command arrives
5. `TENSE_OPEN` explicitly returns the prosthesis to open/release

## Responsibility Split

### Cloud side
1. Optional DB5 foundation pretraining
2. Wearer finetuning on private event-onset data
3. CKPT -> MindIR conversion

### PI side
1. Load MindIR + metadata
2. Read armband stream
3. Recognize onset intent and update the latched prosthesis state

### App side
1. Upload private finetune data
2. Download converted model package and deliver it to PI

## Core Entrypoints

- Pretrain: `scripts/pretrain_db5_repr_method_matrix.py`
- Finetune: `scripts/finetune_event_onset.py`
- Convert: `scripts/convert_event_onset.py`
- Runtime: `scripts/run_event_runtime.py`

Optional comparison tools only:

- Algo baseline export: `scripts/train_event_algo_baseline.py`
- Model vs algo offline compare: `scripts/evaluate_event_dualtrack.py`

The optional algo line and DB5 pretrain line are kept for baseline or ablation work. They are not the primary release path.

## Wearer Data Collection

Single clip collection from the live armband:

```bash
python scripts/collect_event_data.py \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --target_state THUMB_UP \
  --start_state CONTINUE \
  --user_id demo_user \
  --session_id s1 \
  --device_id armband01 \
  --wearing_state normal \
  --duration_sec 3 \
  --port COM4 \
  --baudrate 115200
```

Cue logic for action labels:

- keep `CONTINUE` for about `0.4s`
- perform a fast, clear onset toward the target action
- let the motion settle naturally without trying to maintain a continuous EMG hold

Continuous capture + auto-slice:

```bash
python scripts/collect_event_data_continuous.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --target_state THUMB_UP \
  --start_state CONTINUE \
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

## Recommended 4-Action Demo Workflow

The recommended release-candidate path for the 4-action demo line is:

1. collect data
2. `prepare`
3. `screen`
4. `longrun`
5. `neighbor`
6. `audit`
7. run demo

The single entrypoint is:

- `scripts/train_event_model_90_sprint.py`

One-command flow after collecting new data:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage all \
  --device_target Ascend \
  --device_id 0 \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --run_prefix s2_model90
```

Prepare-only smoke:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage prepare \
  --device_target CPU \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --run_prefix s2_model90
```

Current default screen grid is intentionally bounded:

- `loss_type in {cross_entropy, cb_focal}`
- `base_channels in {16, 24}`
- `freeze_emg_epochs in {6, 8, 10}`
- `encoder_lr_ratio in {0.24, 0.30, 0.36}`
- `pretrained_mode = off`

`neighbor` is verification-only and checks five variants around the best longrun candidate:

- `ref`
- `lr_down`
- `lr_up`
- `freeze_down`
- `freeze_up`

See `docs/model_90_sprint_runbook.md` for the full release-candidate workflow.

## Finetune and Convert

Wearer finetune:

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
  --pretrained_emg_checkpoint <optional_foundation_ckpt> \
  --run_root artifacts/runs \
  --run_id event_finetune_v1
```

Convert to MindIR:

```bash
python scripts/convert_event_onset.py \
  --config configs/conversion_event_onset.yaml \
  --checkpoint artifacts/runs/event_finetune_v1/checkpoints/event_onset_best.ckpt \
  --run_root artifacts/runs \
  --run_id event_convert_v1
```

Converted metadata now exposes `CONTINUE` as the public background label while remaining backward-compatible with legacy `RELAX` inputs.

## PI Runtime

Recommended demo runtime:

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset_demo3_latch.yaml \
  --recognizer_backend model \
  --backend lite
```

Standalone smoke:

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset_demo3_latch.yaml \
  --recognizer_backend model \
  --backend lite \
  --standalone \
  --duration_sec 10
```

If the selected backend artifact is missing, runtime exits immediately. There is no silent fallback.

## Optional Offline Compare

The algo baseline remains available for offline comparison, but it is not the primary delivery path:

```bash
python scripts/train_event_algo_baseline.py \
  --config configs/training_event_onset_demo_p0.yaml \
  --data_dir ../data \
  --recordings_manifest s2_train_manifest_relax12.csv \
  --split_manifest artifacts/splits/s2_stable4_relax12_seed77.json \
  --run_root artifacts/runs \
  --run_id s2_algo_baseline_seed77
```

```bash
python scripts/evaluate_event_dualtrack.py \
  --run_root artifacts/runs \
  --model_run_id s2_stable4_relax12_seed77 \
  --algo_model_path artifacts/runs/s2_algo_baseline_seed77/models/algo_model.json \
  --training_config configs/training_event_onset_demo_p0.yaml \
  --runtime_config configs/runtime_event_onset_demo3_latch.yaml \
  --data_dir ../data \
  --recordings_manifest s2_train_manifest_relax12.csv \
  --split_manifest artifacts/splits/s2_stable4_relax12_seed77.json
```

## Known Boundary

This repository does not promise continuous real-time mirroring of every subtle hand pose detail. The supported target is discrete command switching with stable hold and explicit release.

## Preflight

```bash
python scripts/preflight.py --mode local --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
python scripts/preflight.py --mode ascend --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
```

## Repository Hygiene

Generated outputs are not source-of-truth and should not be committed:

- `code/artifacts/runs/**`
- `code/artifacts/splits/*.json`
- `**/__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `.tmp_pytest*`
