"""
Typed configuration schema and YAML loading helpers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

yaml: Any = None

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..gestures import NUM_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_type: str = "standard"
    in_channels: int = 6
    num_classes: int = NUM_CLASSES
    base_channels: int = 16
    use_se: bool = True
    dropout_rate: float = 0.3


@dataclass
class DualBranchConfig:
    enabled: bool = True
    fuse_mode: str = "concat_channels"
    low_rate: int = 200
    high_rate: int = 1000
    high_segment_length: int = 420
    high_segment_stride: int = 210
    high_stft_window_size: int = 120
    high_stft_hop_size: int = 60
    high_stft_n_fft: int = 230
    high_freq_bins_out: int = 24
    multi_phase_offsets: list[float] = field(default_factory=lambda: [0.0, 0.33, 0.66])


@dataclass
class PreprocessConfig:
    sampling_rate: float = 200.0
    num_channels: int = 6
    total_channels: int = 8

    device_sampling_rate: int = 1000

    lowcut: float = 20.0
    highcut: float = 90.0
    filter_order: int = 4

    stft_window_size: int = 24
    stft_hop_size: int = 12
    stft_n_fft: int = 46

    segment_length: int = 84
    segment_stride: int = 42
    dual_branch: DualBranchConfig = field(default_factory=DualBranchConfig)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    time_warp_rate: float = 0.1
    amplitude_scale: float = 0.15
    noise_std: float = 0.05
    augment_factor: int = 5
    use_mixup: bool = True
    mixup_alpha: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    lr_scheduler: str = "cosine"
    warmup_epochs: int = 3
    lr_step_size: int = 10
    lr_gamma: float = 0.5

    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    early_stop_patience: int = 12

    val_ratio: float = 0.2
    test_ratio: float = 0.2
    split_mode: str = "grouped_file"  # legacy | grouped_file
    split_manifest_path: Optional[str] = None
    split_seed: int = 42
    kfold: int = 0

    device: str = "CPU"

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class InferenceConfig:
    model_path: str = "models/neurogrip.mindir"
    device: str = "CPU"
    num_threads: int = 4


@dataclass
class HardwareConfig:
    sensor_mode: str = "armband"
    sensor_port: Optional[str] = None
    sensor_baudrate: int = 115200
    armband_sampling_rate: int = 1000
    sensor_buffer_size: int = 2000

    actuator_mode: str = "standalone"
    actuator_i2c_bus: int = 1
    actuator_i2c_address: int = 0x40
    actuator_frequency: int = 50

    servo_angle_open: float = 0.0
    servo_angle_half: float = 90.0
    servo_angle_closed: float = 180.0


@dataclass
class RuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    vote_window_size: int = 5
    vote_min_count: int = 3
    confidence_threshold: float = 0.5

    control_rate_hz: float = 30.0
    infer_rate_hz: float = 0.0


def _merge_dict(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_vars(config_dict: dict) -> dict:
    result = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            result[key] = _apply_env_vars(value)
        elif isinstance(value, str) and value.startswith("$"):
            result[key] = os.environ.get(value[1:], None)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(cls, data: dict):
    if not isinstance(data, dict):
        return data

    kwargs = {}
    for field_name, field_info in cls.__dataclass_fields__.items():
        if field_name not in data:
            continue
        value = data[field_name]
        nested_cls = None
        field_type = field_info.type
        if hasattr(field_type, "__dataclass_fields__"):
            nested_cls = field_type
        elif field_info.default_factory is not MISSING:
            try:
                default_instance = field_info.default_factory()
            except TypeError:
                default_instance = None
            if default_instance is not None and hasattr(default_instance, "__dataclass_fields__"):
                nested_cls = type(default_instance)

        if isinstance(value, dict) and nested_cls is not None:
            kwargs[field_name] = _dict_to_dataclass(nested_cls, value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def load_config(yaml_path: str, config_class=None) -> Any:
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is not installed. Run: pip install pyyaml")
    assert yaml is not None

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    config_dict = _apply_env_vars(raw_config)
    if config_class is None:
        return config_dict
    return _dict_to_dataclass(config_class, config_dict)


def save_config(config, yaml_path: str) -> None:
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is not installed. Run: pip install pyyaml")
    assert yaml is not None

    if hasattr(config, "__dataclass_fields__"):
        config_dict = asdict(config)
    else:
        config_dict = dict(config)

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
    sensitive_keywords = {"key", "secret", "password", "token", "credential"}
    result = {}
    for key, value in d.items():
        if any(kw in key.lower() for kw in sensitive_keywords):
            result[key] = "***FILTERED***"
        elif isinstance(value, dict):
            result[key] = _filter_sensitive(value)
        else:
            result[key] = value
    return result


def _resolve_training_config_sections(raw: Dict[str, Any]) -> Dict[str, Any]:
    training_section = dict(raw.get("training", {}))
    data_section = dict(raw.get("data", {}))

    # New canonical location: data.*
    # Keep backward compatibility: if both provided, data.* wins.
    for key in ("split_mode", "test_ratio", "split_manifest_path"):
        if key in data_section:
            training_section[key] = data_section[key]

    if "seed" in data_section:
        training_section["split_seed"] = data_section["seed"]

    return training_section


def load_training_config(yaml_path: str):
    """
    Load training config and return:
    (ModelConfig, PreprocessConfig, TrainingConfig, AugmentationConfig)
    """
    try:
        raw = load_config(yaml_path)
    except FileNotFoundError:
        logger.warning("Config file not found: %s, use defaults", yaml_path)
        return ModelConfig(), PreprocessConfig(), TrainingConfig(), AugmentationConfig()

    if not isinstance(raw, dict):
        return ModelConfig(), PreprocessConfig(), TrainingConfig(), AugmentationConfig()

    training_section = _resolve_training_config_sections(raw)

    return (
        _dict_to_dataclass(ModelConfig, raw.get("model", {})),
        _dict_to_dataclass(PreprocessConfig, raw.get("preprocess", {})),
        _dict_to_dataclass(TrainingConfig, training_section),
        _dict_to_dataclass(AugmentationConfig, raw.get("augmentation", {})),
    )


def load_runtime_config(yaml_path: str) -> RuntimeConfig:
    try:
        raw = load_config(yaml_path)
    except FileNotFoundError:
        logger.warning("Config file not found: %s, use defaults", yaml_path)
        return RuntimeConfig()

    if not isinstance(raw, dict):
        return RuntimeConfig()

    runtime_section = raw.get("runtime", {}) if isinstance(raw.get("runtime", {}), dict) else {}
    infer_rate_hz = runtime_section.get("infer_rate_hz", raw.get("infer_rate_hz", 0.0))

    return RuntimeConfig(
        model=_dict_to_dataclass(ModelConfig, raw.get("model", {})),
        preprocess=_dict_to_dataclass(PreprocessConfig, raw.get("preprocess", {})),
        inference=_dict_to_dataclass(InferenceConfig, raw.get("inference", {})),
        hardware=_dict_to_dataclass(HardwareConfig, raw.get("hardware", {})),
        vote_window_size=raw.get("vote_window_size", 5),
        vote_min_count=raw.get("vote_min_count", 3),
        confidence_threshold=raw.get("confidence_threshold", 0.5),
        control_rate_hz=raw.get("control_rate_hz", 30.0),
        infer_rate_hz=float(infer_rate_hz),
    )
