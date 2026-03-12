# NeuroGrip Event-Onset v1 (DB5 Full53 Foundation)

主线已经固定为：

`DB5 全量53类固定底座 -> 选择 DB5 原始键微调 -> N类 event-onset 运行 -> MindIR 部署`

旧 `aligned3/legacy53` 流程已移除。

## 1) 一键主流程（推荐）

```bash
python scripts/train_event_pipeline.py \
  --db5_data_dir ../data_ninaproDB5 \
  --wearer_data_dir ../data_event_onset \
  --target_db5_keys E1_G01,E1_G02,E2_G05 \
  --budget_per_class 60 \
  --device_target Ascend \
  --device_id 0 \
  --run_id event_pipeline_v1
```

说明：

- 首次运行若固定底座缺失，会自动构建并缓存到 `artifacts/foundation/db5_full53/`。
- 后续运行默认直接复用该底座，不再重复预训练。
- `target_db5_keys` 决定当前微调与运行类别集合，`RELAX` 自动包含。

主产物：

- `artifacts/runs/<run_id>/final_selection.json`
- `artifacts/runs/<run_id>/final_artifacts.json`
- `artifacts/runs/<run_id>/models/event_onset_selected.mindir`
- `artifacts/runs/<run_id>/models/event_onset_selected.model_metadata.json`

## 2) 运行前预检

```bash
python scripts/preflight.py --mode local
python scripts/preflight.py --mode ascend
```

预检会检查：

- DB5 zip 覆盖与 full53 key 覆盖
- 动态类别与 `model.num_classes` 一致性
- 执行映射文件存在且键集合匹配
- MindIR metadata 输入形状与 `class_names` 一致性

## 3) 执行映射（运行必填）

运行侧必须提供完整映射：`DB5键 -> 执行动作`，否则拒绝启动。

示例：`configs/event_actuation_mapping.yaml`

```yaml
actuation_map:
  RELAX: RELAX
  E1_G01: FIST
  E1_G02: PINCH
  E2_G05: OK
```

可用执行动作：`RELAX/FIST/PINCH/OK/YE/SIDEGRIP`

## 4) 运行（MindIR 默认）

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend lite
```

调试可用 CKPT：

```bash
python scripts/run_event_runtime.py \
  --config configs/runtime_event_onset.yaml \
  --backend ckpt
```

## 5) 硬件联调（不走模型）

```bash
python scripts/test_actuator_gesture.py \
  --config configs/runtime_event_onset.yaml
```

键位：

- `r/f/p/o/y/s`：执行对应动作
- `i`：执行器信息
- `h`：帮助
- `q`：退出（自动 `RELAX`）

## 6) 分步入口（高级调试）

- `scripts/pretrain_ninapro_db5.py`（固定 full53 底座构建）
- `scripts/finetune_event_onset.py`（支持动态 `--target_db5_keys`）
- `scripts/convert_event_onset.py`
- `scripts/benchmark_event_runtime_ckpt.py`
- `scripts/evaluate_ckpt.py`
