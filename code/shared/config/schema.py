"""
配置管理 — 类型安全的 dataclass + YAML 加载

所有配置项以 dataclass 定义，提供默认值、类型检查和序列化。
通过 load_config() 从 YAML 文件加载，支持环境变量覆盖敏感信息。

设计原则:
- 每个配置类对应一个关注点（模型/预处理/训练/推理/硬件）
- 字段都有合理默认值，最小化必须配置的项
- YAML 文件结构与 dataclass 层次一一对应
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

yaml: Any = None

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

from ..gestures import NUM_CLASSES


# =============================================================================
# 配置 Dataclass 定义
# =============================================================================


@dataclass
class ModelConfig:
    """模型架构配置"""

    model_type: str = "standard"  # "standard" 或 "lite"
    in_channels: int = 6  # EMG 输入通道数
    num_classes: int = NUM_CLASSES  # 手势类别数
    base_channels: int = 16  # 基础通道宽度
    use_se: bool = True  # 是否使用 SE 注意力
    dropout_rate: float = 0.3  # Dropout 概率


@dataclass
class PreprocessConfig:
    """信号预处理配置"""

    sampling_rate: float = 200.0  # 目标采样率 (Hz)
    num_channels: int = 6  # 使用的 EMG 通道数
    total_channels: int = 8  # 设备总通道数

    device_sampling_rate: int = 1000  # 设备原始采样率 (Hz)

    # 带通滤波
    lowcut: float = 20.0  # 下截止频率 (Hz)
    highcut: float = 90.0  # 上截止频率 (Hz)
    filter_order: int = 4  # 滤波器阶数

    # STFT
    stft_window_size: int = 24  # STFT 窗口大小（采样点）
    stft_hop_size: int = 12  # STFT 步长
    stft_n_fft: int = 46  # FFT 点数

    # 滑动窗口（用于从连续信号中提取训练样本）
    segment_length: int = 84  # 一个训练样本的采样点数（200Hz * 0.42s）
    segment_stride: int = 42  # 样本间滑动步长（50% 重叠）


@dataclass
class AugmentationConfig:
    """数据增强配置"""

    enabled: bool = True  # 是否启用增强
    time_warp_rate: float = 0.1  # 时间扭曲比例 (±10%)
    amplitude_scale: float = 0.15  # 幅度缩放范围 (±15%)
    noise_std: float = 0.05  # 高斯噪声标准差
    augment_factor: int = 5  # 每个样本增强为 N 个（小数据集建议 ≥5）
    use_mixup: bool = True  # 是否启用 Mixup 增强
    mixup_alpha: float = 0.2  # Mixup Beta 分布参数 (越小越保守)


@dataclass
class TrainingConfig:
    """训练超参数配置"""

    epochs: int = 80  # 训练轮数（小数据集建议多轮）
    batch_size: int = 16  # 批大小（小数据集用小 batch）
    learning_rate: float = 0.001  # 初始学习率
    weight_decay: float = 1e-4  # 权重衰减

    # 学习率调度
    lr_scheduler: str = "cosine"  # "step" / "cosine"
    warmup_epochs: int = 3  # 预热轮数
    lr_step_size: int = 10  # StepLR 步长
    lr_gamma: float = 0.5  # StepLR 衰减因子

    # 正则化
    gradient_clip: float = 1.0  # 梯度裁剪阈值
    label_smoothing: float = 0.1  # 标签平滑系数 (0=关闭)
    early_stop_patience: int = 12  # 早停耐心值（配合更多 epochs）

    # 数据划分
    val_ratio: float = 0.2  # 验证集占比
    kfold: int = 0  # K-Fold 折数 (0=不使用, 5=5折交叉验证)

    # 设备
    device: str = "CPU"  # "CPU" / "GPU" / "Ascend"

    # 路径
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class InferenceConfig:
    """推理配置"""

    model_path: str = "models/neurogrip.mindir"
    device: str = "CPU"  # 推理设备: "CPU" / "GPU" / "Ascend"（兼容别名 NPU）
    num_threads: int = 4  # 推理线程数


@dataclass
class HardwareConfig:
    """硬件配置"""

    # 传感器
    sensor_mode: str = "armband"  # "armband" / "standalone"
    sensor_port: Optional[str] = None  # 串口端口（None=自动检测）
    sensor_baudrate: int = 115200
    armband_sampling_rate: int = 1000  # 臂环原始采样率
    sensor_buffer_size: int = 2000

    # 执行器
    actuator_mode: str = "standalone"  # "real" / "standalone"
    actuator_i2c_bus: int = 1
    actuator_i2c_address: int = 0x40
    actuator_frequency: int = 50  # PWM 频率 (Hz)

    # 舵机角度范围
    servo_angle_open: float = 0.0
    servo_angle_half: float = 90.0
    servo_angle_closed: float = 180.0


@dataclass
class RuntimeConfig:
    """运行时总配置（聚合各子配置）"""

    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # 后处理
    vote_window_size: int = 5  # 多数投票窗口
    vote_min_count: int = 3  # 最小投票数
    confidence_threshold: float = 0.5  # 置信度阈值

    # 控制循环
    control_rate_hz: float = 30.0  # 控制循环频率


# =============================================================================
# 配置加载与保存
# =============================================================================


def _merge_dict(base: dict, override: dict) -> dict:
    """递归合并字典（override 覆盖 base）"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_vars(config_dict: dict) -> dict:
    """
    将环境变量注入到配置中

    约定: 以 $ 开头的值从环境变量读取
    例: sensor_port: "$EMG_SERIAL_PORT" → 读取 os.environ["EMG_SERIAL_PORT"]
    """
    result = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            result[key] = _apply_env_vars(value)
        elif isinstance(value, str) and value.startswith("$"):
            env_key = value[1:]
            result[key] = os.environ.get(env_key, None)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(cls, data: dict):
    """将字典递归转换为 dataclass 实例"""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            # 如果字段类型也是 dataclass，递归转换
            if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
                kwargs[field_name] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)


def load_config(yaml_path: str, config_class=None) -> Any:
    """
    从 YAML 文件加载配置

    Args:
        yaml_path: YAML 文件路径
        config_class: 目标 dataclass 类。
            - 传入 dataclass: 返回对应 dataclass 实例
            - 传入 None: 返回原始 dict（不做类型推断）

    Returns:
        配置 dataclass 实例
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML 未安装，请运行: pip install pyyaml")
    assert yaml is not None

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # 环境变量注入
    config_dict = _apply_env_vars(raw_config)

    # 未指定目标类型时，返回原始字典。
    # 这样可避免“字段猜测类型”导致的误判和静默回退。
    if config_class is None:
        return config_dict

    return _dict_to_dataclass(config_class, config_dict)


def save_config(config, yaml_path: str) -> None:
    """
    将配置保存为 YAML 文件

    自动过滤敏感信息（包含 key/secret/password/token 的字段）。

    Args:
        config: dataclass 实例或字典
        yaml_path: 输出 YAML 文件路径
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML 未安装，请运行: pip install pyyaml")
    assert yaml is not None

    if hasattr(config, "__dataclass_fields__"):
        config_dict = asdict(config)
    else:
        config_dict = dict(config)

    # 过滤敏感字段
    config_dict = _filter_sensitive(config_dict)

    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def _filter_sensitive(d: dict) -> dict:
    """递归过滤包含敏感关键词的字段"""
    sensitive_keywords = {"key", "secret", "password", "token", "credential"}
    result = {}
    for k, v in d.items():
        if any(kw in k.lower() for kw in sensitive_keywords):
            result[k] = "***FILTERED***"
        elif isinstance(v, dict):
            result[k] = _filter_sensitive(v)
        else:
            result[k] = v
    return result


# =============================================================================
# 便捷加载函数
# =============================================================================


def load_training_config(yaml_path: str):
    """
    加载训练配置，返回 (ModelConfig, PreprocessConfig, TrainingConfig, AugmentationConfig)

    文件不存在时返回全默认配置。
    """
    try:
        raw = load_config(yaml_path)
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {yaml_path}，使用默认配置")
        return ModelConfig(), PreprocessConfig(), TrainingConfig(), AugmentationConfig()
    if not isinstance(raw, dict):
        return ModelConfig(), PreprocessConfig(), TrainingConfig(), AugmentationConfig()
    return (
        ModelConfig(**raw.get("model", {})),
        PreprocessConfig(**raw.get("preprocess", {})),
        TrainingConfig(**raw.get("training", {})),
        AugmentationConfig(**raw.get("augmentation", {})),
    )


def load_runtime_config(yaml_path: str) -> RuntimeConfig:
    """
    加载运行时配置，返回 RuntimeConfig

    文件不存在时返回全默认配置。
    """
    try:
        raw = load_config(yaml_path)
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {yaml_path}，使用默认配置")
        return RuntimeConfig()
    if not isinstance(raw, dict):
        return RuntimeConfig()
    return RuntimeConfig(
        model=ModelConfig(**raw.get("model", {})),
        preprocess=PreprocessConfig(**raw.get("preprocess", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        hardware=HardwareConfig(**raw.get("hardware", {})),
        vote_window_size=raw.get("vote_window_size", 5),
        vote_min_count=raw.get("vote_min_count", 3),
        confidence_threshold=raw.get("confidence_threshold", 0.5),
        control_rate_hz=raw.get("control_rate_hz", 30.0),
    )
