# 模型线 0.9 冲刺 Runbook（4动作+RELAX）

## 1) 一次性配置 4090 免密（本机执行）

```bash
python scripts/setup_ssh_key_ccnu.py --host 10.102.65.27 --user wxb --print_test_command
```

说明：
- 脚本会读取环境变量 `CCNU_SSH_PASSWORD`。
- 若本机没有 `paramiko`，先安装：`python -m pip install paramiko`。

## 2) 4060 基线复现

```bash
python scripts/train_event_model_90_sprint.py \
  --stage baseline \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

产物：
- `artifacts/runs/s2_model90_baseline_summary.json`

## 3) 4060 快速筛参（固定 split，v2 grouped_file）

```bash
python scripts/train_event_model_90_sprint.py \
  --stage screen \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90 \
  --screen_split_seed 42 \
  --screen_split_manifest artifacts/splits/s2_relax12_4class_seed42_v2.json \
  --screen_loss_types cross_entropy,cb_focal \
  --screen_base_channels 16,24 \
  --screen_freeze_emg_epochs 5,8 \
  --screen_encoder_lr_ratios 0.3,0.2 \
  --screen_pretrained_modes off,on \
  --pretrained_emg_checkpoint <db5_or_repr_ckpt_path>
```

产物：
- `artifacts/runs/s2_model90_screen_summary.json`
- `artifacts/runs/s2_model90_screen_summary.csv`

## 4) 4090 长跑收敛（Top2 × seed=42/52/62）

```bash
python scripts/train_event_model_90_sprint.py \
  --stage longrun \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90 \
  --longrun_seeds 42,52,62
```

产物：
- `artifacts/runs/s2_model90_longrun_summary.json`
- `artifacts/runs/s2_model90_longrun_summary.csv`

## 5) 阈值与状态机调参（基于最佳 run）

```bash
python scripts/train_event_model_90_sprint.py \
  --stage tune \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

产物：
- `artifacts/runs/s2_model90_runtime_threshold_tuning_summary.json`
- `artifacts/runs/s2_model90_runtime_threshold_tuning_summary.csv`
- `artifacts/runs/s2_model90_runtime_event_onset_demo_latch_tuned.yaml`

## 6) 单命令串行全跑（可选）

```bash
python scripts/train_event_model_90_sprint.py \
  --stage all \
  --device_target GPU \
  --device_id 0 \
  --run_prefix s2_model90
```

## 7) 验收门槛（严格演示口径）

- `command_success_rate >= 0.90`
- `false_trigger_rate <= 0.05`
- `false_release_rate <= 0.05`
- 开发阶段同时关注：`event_action_accuracy >= 0.90`
