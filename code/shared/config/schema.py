"""Configuration schema and loaders used by training/runtime/conversion."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    model_type: str = "standard"
    in_channels: int = 12
    num_classes: int = 6
    base_channels: int = 16
    use_se: bool = True
    dropout_rate: float = 0.3
    hidden_dim: int = 64
    num_layers: int = 2


@dataclass
class DualBranchConfig:
    enabled: bool = True
    fuse_mode: str = "concat_channels"
    low_rate: int = 200
    high_rate: int = 1000
    high_segment_length: int = 420
    high_segment_stride: int = 210
    high_stft_window: int = 120
    high_stft_hop: int = 60
    high_stft_n_fft: int = 230
    high_freq_bins_out: int = 24
    multi_phase_offsets: List[float] = field(default_factory=lambda: [0.0, 0.33, 0.66])

    @property
    def high_stft_window_size(self) -> int:
        return int(self.high_stft_window)

    @property
    def high_stft_hop_size(self) -> int:
        return int(self.high_stft_hop)


@dataclass
class PreprocessConfig:
    sampling_rate: int = 200
    num_channels: int = 6
    lowcut: float = 20.0
    highcut: float = 90.0
    filter_order: int = 4
    device_sampling_rate: int = 1000
    target_length: int = 420
    overlap: float = 0.5
    stft_window: int = 120
    stft_hop: int = 60
    n_fft: int = 230
    freq_bins_out: int = 24
    normalize: str = "log"
    clip_min: float = 0.0
    clip_max: float = 10.0
    dual_branch: DualBranchConfig = field(default_factory=DualBranchConfig)

    @property
    def segment_length(self) -> int:
        return int(self.target_length)

    @property
    def segment_stride(self) -> int:
        stride = int(round(self.target_length * max(0.0, 1.0 - float(self.overlap))))
        return max(1, stride)

    @property
    def stft_window_size(self) -> int:
        return int(self.stft_window)

    @property
    def stft_hop_size(self) -> int:
        return int(self.stft_hop)

    @property
    def stft_n_fft(self) -> int:
        return int(self.n_fft)

    @property
    def total_channels(self) -> int:
        return int(self.num_channels)


@dataclass
class LossConfig:
    type: str = "cb_focal"
    focal_gamma: float = 1.5
    class_balance_beta: float = 0.999


@dataclass
class SamplerConfig:
    type: str = "balanced"
    hard_mining_ratio: float = 0.3
    confusion_pairs: List[List[Union[str, int]]] = field(
        default_factory=lambda: [["FIST", "PINCH"], ["YE", "SIDEGRIP"], ["OK", "RELAX"]]
    )


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.999


@dataclass
class QualityFilterConfig:
    enabled: bool = True
    energy_min: float = 2.5
    clip_ratio_max: float = 0.08
    saturation_abs: float = 126.0
    static_std_max: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    early_stopping_patience: int = 12
    split_seed: int = 42
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    kfold: Optional[int] = None
    num_workers: int = 2
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)


@dataclass
class DataConfig:
    split_mode: str = "grouped_file"
    split_manifest_path: str = "artifacts/splits/default_split_manifest.json"
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    augment_factor: int = 5
    noise_std: float = 0.02
    temporal_shift_max: int = 2
    scale_min: float = 0.9
    scale_max: float = 1.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.8
    smoothing_window_ms: int = 400
    hysteresis_count: int = 3
    tta_offsets: List[float] = field(default_factory=lambda: [0.0, 0.33, 0.66])
    use_lite: bool = True


@dataclass
class DeviceConfig:
    target: str = "CPU"
    id: int = 0
    sampling_rate: int = 1000


@dataclass
class HardwareConfig:
    sensor_mode: str = "armband"
    actuator_mode: str = "pca9685"
    sensor_port: Optional[str] = None
    sensor_baudrate: int = 115200
    sensor_buffer_size: int = 2000
    armband_sampling_rate: int = 1000
    target_sampling_rate: int = 200
    actuator_i2c_bus: int = 1
    actuator_i2c_address: int = 0x40
    actuator_frequency: int = 50
    actuator_channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    servo_angle_open: float = 0.0
    servo_angle_half: float = 90.0
    servo_angle_closed: float = 180.0


@dataclass
class RuntimeConfig:
    model_path: str = "models/neurogrip_6g.mindir"
    gestures_path: str = "gestures.yaml"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    control_rate_hz: float = 50.0
    infer_rate_hz: float = 20.0

    def __post_init__(self) -> None:
        self.hardware.armband_sampling_rate = int(self.device.sampling_rate)
        self.hardware.target_sampling_rate = int(self.preprocess.sampling_rate)


@dataclass
class ConversionConfig:
    checkpoint_path: str = "checkpoints/neurogrip_best.ckpt"
    output_path: str = "models/neurogrip_6g.mindir"
    input_shape: List[int] = field(default_factory=lambda: [1, 12, 24, 6])
    mindir_version: str = "latest"


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _coerce_scalar(value: Any, expected_type: Type[Any]) -> Any:
    if expected_type is bool and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    try:
        return expected_type(value)
    except (TypeError, ValueError):
        return value


def _convert_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is Any:
        return value

    if origin is Union:
        non_none = [arg for arg in args if arg is not type(None)]
        if value is None and len(non_none) < len(args):
            return None
        if non_none:
            return _convert_value(value, non_none[0])
        return value

    if is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return _dict_to_dataclass(value, annotation)
        return annotation()  # type: ignore[misc]

    if origin in (list, List, Sequence, tuple, Tuple):
        inner = args[0] if args else Any
        if not isinstance(value, (list, tuple)):
            return []
        converted = [_convert_value(v, inner) for v in value]
        if origin in (tuple, Tuple):
            return tuple(converted)
        return converted

    if annotation in (int, float, str, bool):
        return _coerce_scalar(value, annotation)

    return value


def _dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        annotation = type_hints.get(f.name, f.type)
        kwargs[f.name] = _convert_value(data[f.name], annotation)
    return cls(**kwargs)


def _resolve_training_config_sections(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep backward compatibility:
    - Prefer nested sections: model/preprocess/training/augmentation/data
    - Fallback to flattened keys at root if nested section is absent
    """
    root = data or {}
    training_section = dict(root.get("training", {}))
    data_section = dict(root.get("data", {}))

    if "quality_filter" in data_section and "quality_filter" not in training_section:
        training_section["quality_filter"] = data_section["quality_filter"]

    return {
        "model": root.get("model", root),
        "preprocess": root.get("preprocess", root),
        "training": training_section if training_section else root,
        "augmentation": root.get("augmentation", root),
        "data": data_section if data_section else root,
    }


def _resolve_runtime_config_sections(data: Dict[str, Any]) -> Dict[str, Any]:
    root = dict(data or {})
    inference_section = dict(root.get("inference", {}))
    device_section = dict(root.get("device", {}))
    hardware_section = dict(root.get("hardware", {}))

    legacy_device = inference_section.pop("device", None)
    if legacy_device is not None and "target" not in device_section:
        device_section["target"] = legacy_device

    if "sampling_rate" in device_section and "armband_sampling_rate" not in hardware_section:
        hardware_section["armband_sampling_rate"] = device_section["sampling_rate"]

    return {
        "model_path": root.get("model_path", "models/neurogrip_6g.mindir"),
        "gestures_path": root.get("gestures_path", "gestures.yaml"),
        "preprocess": root.get("preprocess", {}),
        "inference": inference_section,
        "device": device_section,
        "hardware": hardware_section,
        "control_rate_hz": root.get("control_rate_hz", 50.0),
        "infer_rate_hz": root.get("infer_rate_hz", 20.0),
    }


def load_training_config(path: Union[str, Path]) -> tuple[ModelConfig, PreprocessConfig, TrainingConfig, AugmentationConfig]:
    data = load_config(path)
    sections = _resolve_training_config_sections(data)
    model = _dict_to_dataclass(sections["model"], ModelConfig)
    preprocess = _dict_to_dataclass(sections["preprocess"], PreprocessConfig)
    training = _dict_to_dataclass(sections["training"], TrainingConfig)
    augmentation = _dict_to_dataclass(sections["augmentation"], AugmentationConfig)
    return model, preprocess, training, augmentation


def load_training_data_config(path: Union[str, Path]) -> DataConfig:
    data = load_config(path)
    sections = _resolve_training_config_sections(data)
    return _dict_to_dataclass(sections["data"], DataConfig)


def load_runtime_config(path: Union[str, Path]) -> RuntimeConfig:
    data = load_config(path)
    return _dict_to_dataclass(_resolve_runtime_config_sections(data), RuntimeConfig)


def load_conversion_config(path: Union[str, Path]) -> ConversionConfig:
    data = load_config(path)
    return _dict_to_dataclass(data, ConversionConfig)
