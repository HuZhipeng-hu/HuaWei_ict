# Model 90 Sprint Runbook

## Goal

This runbook is for the demo3 two-stage release candidate:

- `CONTINUE`
- `TENSE_OPEN`
- `THUMB_UP`
- `WRIST_CW`

Recommended semantics:

- `CONTINUE`: no new command, keep the current prosthesis state
- `TENSE_OPEN`: explicit open/release command
- `THUMB_UP` and `WRIST_CW`: switch to target state and latch
- `WRIST_CCW`, `V_SIGN`, and `OK_SIGN` remain extension classes outside the default demo3 path

This is an event-driven latch protocol. It is not continuous motion mirroring.

Main entry:

- `scripts/train_event_model_90_sprint.py`

Default workflow:

1. collection audit
2. filtered demo3 manifest build
3. deterministic grouped-file split build
4. bounded screen
5. longrun stability check
6. neighbor verification
7. audit

## One-Command Flow

After collecting new data:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage all \
  --device_target Ascend \
  --device_id 0 \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --run_prefix s2_model90
```

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

Use this before long device runs:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage prepare \
  --device_target CPU \
  --data_dir ../data \
  --recordings_manifest recordings_manifest.csv \
  --run_prefix s2_model90
```

Default prepare rules:

- only `event_onset` capture mode is kept
- dead-channel clips are dropped
- action clips require at least `2` selected windows
- `CONTINUE` clips require at least `1` selected window
- `CONTINUE` clips may keep `retake_recommended` quality if no dead channel is present

Useful optional flags:

- `--prepare_session_id s2`
- `--prepare_target_per_class 12`
- `--prepare_relax_target_count 24`
- `--prepare_action_min_selected_windows 2`
- `--prepare_relax_min_selected_windows 1`
- `--prepare_relax_allow_retake_quality true`
- `--prepare_output_manifest ../data/s2_model90_demo3_train_manifest.csv`

The `prepare_relax_*` flag names are kept for backward compatibility. They control the public `CONTINUE` background class.

## Stage Commands

Baseline:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage baseline \
  --device_target Ascend \
  --device_id 0 \
  --run_prefix s2_model90
```

Screen:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage screen \
  --device_target Ascend \
  --device_id 0 \
  --run_prefix s2_model90 \
  --screen_split_seed 42 \
  --screen_loss_types cross_entropy,cb_focal \
  --screen_base_channels 16,24 \
  --screen_freeze_emg_epochs 6,8,10 \
  --screen_encoder_lr_ratios 0.24,0.3,0.36
```

Longrun:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage longrun \
  --device_target Ascend \
  --device_id 0 \
  --run_prefix s2_model90 \
  --longrun_seeds 42,52,62
```

Neighbor verification:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage neighbor \
  --device_target Ascend \
  --device_id 0 \
  --run_prefix s2_model90
```

Neighbor only checks five variants around the best longrun candidate:

- `ref`
- `lr_down`
- `lr_up`
- `freeze_down`
- `freeze_up`

Runtime threshold tuning:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage tune \
  --device_target Ascend \
  --device_id 0 \
  --run_prefix s2_model90
```

The current release default runtime thresholds are already baked into `configs/runtime_event_onset.yaml`.
Only keep a new tuned configuration if it beats the current release baseline on online control metrics.

Pipeline audit:

```bash
python scripts/train_event_model_90_sprint.py \
  --stage audit \
  --device_target Ascend \
  --device_id 0 \
  --run_prefix s2_model90
```

## Audit Reading Guide

Primary file:

- `artifacts/runs/<run_prefix>_audit_report.json`

Root cause categories:

- `artifact_contract_bug`
- `implementation_bug`
- `hyperparameter_underfit`
- `data_bottleneck`

Release-candidate interpretation:

- `design`, `implementation`, and `stability` must pass
- `param_coverage` should pass, or remain as the only clearly documented blocker
- if `neighbor` shows no significant improvement and the first three gates pass, the next likely bottlenecks are data quality, sampling consistency, and task difficulty rather than hidden implementation bugs

Key fields:

- `root_cause_category`
- `root_cause_summary`
- `blocking_issues`
- `goal_assessment`

## Remote Execution

ModelArts Ascend is the preferred training target for this release-candidate line. Use separate runs for independent experiments; do not treat this sprint script as distributed training.

## Acceptance Targets

Primary offline gate:

- `event_action_accuracy >= 0.90`

Secondary offline gate:

- `event_action_macro_f1 >= 0.88`

Runtime gate:

- `command_success_rate >= 0.90`
- `false_trigger_rate <= 0.05`
- `false_release_rate <= 0.05`

Deployment parity gate:

- `CKPT` vs `MindIR/Lite` `command_success_rate` delta `<= 0.05`
- `false_trigger_rate` delta `<= 0.05`
- `false_release_rate` delta `<= 0.02`
- `event_action_accuracy` delta `<= 0.02`
