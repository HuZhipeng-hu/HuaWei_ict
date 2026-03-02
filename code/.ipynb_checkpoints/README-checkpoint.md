# NeuroGrip Pro V2

AI 驱动的智能上肢义肢控制系统 — 基于 sEMG 肌电信号的手势识别与义肢控制。

## 架构概览

```
code_new/
├── shared/          # 共享核心（模型定义、预处理、手势定义、配置）
├── training/        # 模块1: 训练（在 NPU/GPU 上运行）
├── conversion/      # 模块2: 模型转换（.ckpt → .mindir）
├── runtime/         # 模块3: 义肢运行（在 Orange Pi 上运行）
├── configs/         # YAML 配置文件
└── tests/           # 单元测试
```

三个功能模块彼此**零依赖**，仅通过 `shared/` 共享核心和模型文件（`.ckpt` / `.mindir`）连接。

## 支持的手势

| ID | 手势 | 说明 |
|----|------|------|
| 0 | RELAX | 放松 / 静息 |
| 1 | FIST | 握拳 |
| 2 | PINCH | 捏取（拇指+食指）|
| 3 | OK | OK 手势 |
| 4 | YE | 剪刀手 / 耶 |
| 5 | SIDEGRIP | 侧握 |

## 快速开始

### 1. 训练模型

在有 MindSpore 环境的 NPU/GPU 机器上：

```bash
cd code_new/

# 使用默认配置训练
python -m training.train --data_dir ../data/ --config configs/training.yaml

# 自定义训练轮数和设备
python -m training.train --data_dir ../data/ --epochs 100 --device GPU

# 禁用数据增强
python -m training.train --data_dir ../data/ --no_augment
```

训练完成后，最优模型保存在 `checkpoints/neurogrip_best.ckpt`。

### 2. 转换模型

```bash
# 导出为 MindSpore Lite 格式
python -m conversion.convert \
    --checkpoint checkpoints/neurogrip_best.ckpt \
    --output models/neurogrip

# 导出 + INT8 量化
python -m conversion.convert \
    --checkpoint checkpoints/neurogrip_best.ckpt \
    --output models/neurogrip \
    --quantize
```

### 3. 运行义肢

在 Orange Pi 上：

```bash
# 正常运行
python -m runtime.run --config configs/runtime.yaml

# 指定模型和串口
python -m runtime.run --model models/neurogrip.mindir --port /dev/ttyUSB0

# Standalone 模式（模拟硬件，调试用）
python -m runtime.run --standalone
```

## 添加新手势

只需两步：

1. **修改 `shared/gestures.py`**：添加枚举值和手指映射
2. **采集数据**：在 `data/` 下新建文件夹，放入 CSV 文件

然后重新训练即可，不需要修改任何其他代码。

## 数据格式

数据目录结构：
```
data/
├── Relax/          # 放松
│   ├── xxx.csv
│   └── ...
├── fist/           # 握拳
├── Pinch/          # 捏取
├── ok/             # OK
├── ye/             # 剪刀手
└── Sidegrip/       # 侧握
```

CSV 格式（思知瑞臂环，1000Hz 采样率）：
```
emg1,emg2,...,emg8,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,...
128,130,125,...
```

## 依赖

### 训练环境（NPU/GPU）
- Python >= 3.8
- MindSpore >= 2.7.1
- NumPy, SciPy, PyYAML

### 运行环境（Orange Pi）
- Python >= 3.8
- MindSpore Lite
- NumPy, SciPy, PyYAML, pyserial, smbus2

## 配置说明

所有配置以 YAML 文件管理，每个模块一个：

| 文件 | 用途 |
|------|------|
| `configs/training.yaml` | 模型架构 + 训练超参数 + 预处理 + 数据增强 |
| `configs/conversion.yaml` | 导出格式 + 量化参数 |
| `configs/runtime.yaml` | 推理 + 硬件 + 后处理 + 控制频率 |
