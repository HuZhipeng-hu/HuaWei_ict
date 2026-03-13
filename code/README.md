# NeuroGrip DB5 Foundation (Public Repro First)

本仓库当前默认复现口径是：

- 公共轨（评委开箱）：只依赖 NinaPro DB5，完成 foundation 表征预训练。
- 内部轨（可选附加）：在有 `recordings_manifest.csv` 的前提下运行 few-shot 迁移评估。

few-shot 是小样本微调评估，不是预训练本体的必需步骤。

## 1) 公共轨：一条命令开箱复现（推荐）

```bash
python scripts/pretrain_db5_repr_method_matrix.py \
  --pretrain_config configs/pretrain_ninapro_db5.yaml \
  --db5_data_dir ../data_ninaproDB5 \
  --run_root artifacts/runs \
  --run_prefix db5_public_repro_v1 \
  --device_target Ascend \
  --device_id 0 \
  --foundation_dir artifacts/foundation/db5_full53 \
  --fewshot_mode off
```

说明：

- `fewshot_mode=off` 为默认值，缺少 `recordings_manifest.csv` 也不会失败。
- 矩阵会按公共排序规则自动选出最佳预训练轮次：
  - `best_val_macro_f1` 优先
  - 平手看 `best_val_acc`
  - 再看 `test_macro_f1`
- 关键产物：
  - `artifacts/runs/<run_prefix>_summary/db5_repr_method_matrix_summary.json`
  - `artifacts/runs/<run_prefix>_summary/db5_repr_method_matrix_summary.csv`
  - `artifacts/runs/<run_prefix>_summary/referee_repro_card.md`

## 2) 内部轨：few-shot 迁移评估（可选）

仅当你有事件数据与 `recordings_manifest.csv` 时启用：

```bash
python scripts/pretrain_db5_repr_method_matrix.py \
  --pretrain_config configs/pretrain_ninapro_db5.yaml \
  --fewshot_config configs/training_event_onset.yaml \
  --db5_data_dir ../data_ninaproDB5 \
  --wearer_data_dir ../data \
  --recordings_manifest /path/to/recordings_manifest.csv \
  --run_root artifacts/runs \
  --run_prefix db5_internal_repro_v1 \
  --device_target Ascend \
  --device_id 0 \
  --foundation_dir artifacts/foundation/db5_full53 \
  --fewshot_mode on \
  --target_db5_keys E1_G01,E1_G02,E1_G03,E1_G04 \
  --budgets 10,20,35,60 \
  --seeds 11,22,33
```

说明：

- `fewshot_mode=on` 会强制检查 `--recordings_manifest`。
- 如果要“有则跑、无则跳过”，可用 `--fewshot_mode auto`。

## 3) 运行前预检

```bash
python scripts/preflight.py --mode local --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
python scripts/preflight.py --mode ascend --db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data
```

## 4) 常用脚本

- `scripts/pretrain_ninapro_db5_repr.py`：单轮 DB5 表征预训练
- `scripts/pretrain_db5_repr_method_matrix.py`：方法矩阵（公共轨/内部轨）
- `scripts/evaluate_event_fewshot_curve.py`：单独 few-shot 预算曲线评估
- `scripts/preflight.py`：环境与数据前置检查

## 5) 仓库卫生约定

- `code/artifacts/runs/` 与 `code/artifacts/splits/*.json` 为运行产物，不入库。
- `.ipynb_checkpoints/` 与 `__pycache__/` 不入库。
- 若需保留结果用于汇报，请导出到仓库外部或发布工件系统。
