# Model 90 Sprint Runbook

## Goal

This runbook is for the 4-action demo line:

- `RELAX`
- `TENSE_OPEN`
- `THUMB_UP`
- `WRIST_CW`
- `WRIST_CCW`

The main entry is [`scripts/train_event_model_90_sprint.py`](/F:/ICT/义肢核心代码/code/scripts/train_event_model_90_sprint.py).
The default workflow now starts from the raw collection manifest and performs:

1. collection audit
2. filtered 4-action manifest build
3. deterministic grouped-file split build
4. training / evaluation stages
5. unified summary and audit output

## Default One-Command Flow

After collecting new data, run:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage all \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

Default assumptions:

- raw source manifest: `recordings_manifest.csv`
- target actions: `TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW`
- prepare stage is enabled by default

Main outputs:

- `artifacts/runs/s2_model90_prepare_summary.json`
- `artifacts/runs/s2_model90_baseline_summary.json`
- `artifacts/runs/s2_model90_screen_summary.json`
- `artifacts/runs/s2_model90_longrun_summary.json`
- `artifacts/runs/s2_model90_neighbor_summary.json`
- `artifacts/runs/s2_model90_tune_summary.json`
- `artifacts/runs/s2_model90_audit_report.json`
- `artifacts/runs/s2_model90_pipeline_report.json`

## Prepare Only

Use this to verify the data gate before long GPU runs:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage prepare \
  --device_target CPU \
  --run_prefix s2_model90
```

This stage will:

- audit `recordings_manifest.csv`
- build the audited 4-action training manifest
- build deterministic split manifests for screen / longrun / neighbor seeds

Default prepare rules:

- only `event_onset` capture mode is kept
- dead-channel clips are dropped
- action clips require at least `2` selected windows
- `RELAX` clips require at least `1` selected window
- `RELAX` may keep `retake_recommended` quality if no dead channel is present

Useful optional flags:

```bash
--prepare_session_id s2
--prepare_target_per_class 12
--prepare_relax_target_count 24
--prepare_action_min_selected_windows 2
--prepare_relax_min_selected_windows 1
--prepare_relax_allow_retake_quality true
--prepare_output_manifest ../data/s2_model90_4class_train_manifest.csv
```

## Stage-by-Stage Commands

Baseline:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage baseline \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

Screen:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage screen \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90 \
  --screen_split_seed 42 \
  --screen_loss_types cross_entropy,cb_focal \
  --screen_base_channels 16,24 \
  --screen_freeze_emg_epochs 5,8 \
  --screen_encoder_lr_ratios 0.3,0.2 \
  --screen_pretrained_modes off,on \
  --pretrained_emg_checkpoint <db5_or_repr_ckpt_path>
```

Longrun:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage longrun \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90 \
  --longrun_seeds 42,52,62
```

Neighbor tuning:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage neighbor \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

Runtime threshold tuning:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage tune \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

Pipeline audit:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage audit \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

## Audit Reading Guide

The audit report now separates root cause classes:

- `artifact_contract_bug`
- `implementation_bug`
- `hyperparameter_underfit`
- `data_bottleneck`

Primary file:

- `artifacts/runs/<run_prefix>_audit_report.json`

Key fields:

- `root_cause_category`
- `root_cause_summary`
- `blocking_issues`
- `goal_assessment`

If `root_cause_category=data_bottleneck`, the code path and artifact contract are considered clear enough that more data is the main next lever.

## Remote 4090 Bootstrap

Bootstrap SSH key auth from the local machine:

```bash
python -m pip install paramiko
python scripts/setup_ssh_key_ccnu.py --host 10.102.65.27 --user wxb --print_test_command
```

After that, test login and inspect the remote environment:

```bash
ssh -i ~/.ssh/id_ed25519_ccnu4090 wxb@10.102.65.27 "hostname && python -V && nvidia-smi"
```

Use the dual 4090 host for independent runs on separate GPUs rather than distributed training.

## Acceptance Targets

Primary offline gate:

- `event_action_accuracy >= 0.90`

Secondary offline gate:

- `event_action_macro_f1 >= 0.88`

Runtime gate:

- `command_success_rate >= 0.90`
- `false_trigger_rate <= 0.05`
- `false_release_rate <= 0.05`
