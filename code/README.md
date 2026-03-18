# NeuroGrip Demo3 Release Path

This repository now exposes a single recommended production path for the prosthesis demo:

- collect wearer data
- audit and prepare demo3 manifest/split
- finetune the event-onset model
- convert `CKPT -> MindIR + metadata`
- run latched prosthesis control with the converted model

The supported target is **event-driven latch control**, not continuous motion mirroring.

## Default Control Contract

- `CONTINUE`: no new command, keep the current prosthesis state
- `TENSE_OPEN`: explicit open/release command
- `THUMB_UP`: switch to `THUMB_UP` and latch
- `WRIST_CW`: switch to `WRIST_CW` and latch

Runtime default is `release_mode=command_only`.
The release default runtime thresholds are the tuned demo3 baseline thresholds baked into `configs/runtime_event_onset.yaml`.

## Main Entrypoints

- collect single clip: `scripts/collect_event_data.py`
- collect continuous stream: `scripts/collect_event_data_continuous.py`
- prepare + bounded search: `scripts/train_event_model_90_sprint.py`
- direct finetune: `scripts/finetune_event_onset.py`
- convert: `scripts/convert_event_onset.py`
- runtime: `scripts/run_event_runtime.py`
- control eval: `scripts/evaluate_event_demo_control.py`

Experimental and retired branches are kept under `experimental/` and are not part of the release path.

## Collection

Single clip:

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
  --port COM5 \
  --baudrate 115200
```

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

Cue logic for action labels:

- keep `CONTINUE` for about `0.4s`
- perform a fast, clear onset toward the target action
- return to `CONTINUE`
- do not try to maintain a continuous EMG hold

## Release-Candidate Workflow

One-command flow after collecting data:

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

The bounded default screen grid is:

- `loss_type in {cross_entropy, cb_focal}`
- `base_channels in {16, 24}`
- `freeze_emg_epochs in {6, 8, 10}`
- `encoder_lr_ratio in {0.24, 0.30, 0.36}`
- `pretrained_mode = off`

See [model_90_sprint_runbook.md](/F:/ICT/义肢核心代码/code/docs/model_90_sprint_runbook.md) for the full sprint flow.

## Finetune and Convert

Direct finetune:

```bash
python scripts/finetune_event_onset.py \
  --config configs/training_event_onset.yaml \
  --data_dir ../data \
  --recordings_manifest ../data/recordings_manifest.csv \
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

Converted metadata exposes `CONTINUE` as the public background label.
Release freeze requires `CKPT` and `MindIR/Lite` control behavior to stay within a narrow parity band before deployment.

## Runtime

Recommended runtime:

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend lite
```

Standalone smoke:

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend lite \
  --standalone \
  --duration_sec 10
```

If the selected backend artifact is missing, runtime exits immediately. There is no silent fallback.

## Known Boundary

This repository does not promise continuous real-time mirroring of every subtle hand pose detail. The supported target is discrete command switching with stable hold and explicit release.

## Preflight

```bash
python scripts/preflight.py --mode local --wearer_data_dir ../data
python scripts/preflight.py --mode ascend --wearer_data_dir ../data
```

## Repository Hygiene

Do not commit generated outputs:

- `code/artifacts/runs/**`
- `code/artifacts/splits/*.json`
- `**/__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `.tmp_pytest*`
