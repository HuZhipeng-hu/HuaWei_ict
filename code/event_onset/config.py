"""Configuration helpers for the event-onset experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar, get_args, get_origin, get_type_hints

from shared.config import AugmentationConfig, DeviceConfig, HardwareConfig, QualityFilterConfig, TrainingConfig, load_config
from shared.label_modes import get_label_mode_spec

T = TypeVar("T")


@dataclass
class EventFeatureConfig:
    context_window_ms: int = 240
    window_step_ms: int = 20
    emg_stft_window: int = 64
    emg_stft_hop: int = 24
    emg_n_fft: int = 96
    emg_freq_bins: int = 24
    imu_resample_steps: int = 16
    imu_input_dim: int = 6
    imu_motion_std_max: float = 2.5

    def context_samples(self, sampling_rate_hz: int) -> int:
        return max(1, int(round(float(self.context_window_ms) * float(sampling_rate_hz) / 1000.0)))

    def step_samples(self, sampling_rate_hz: int) -> int:
        return max(1, int(round(float(self.window_step_ms) * float(sampling_rate_hz) / 1000.0)))


@dataclass
class EventDataConfig:
    label_mode: str = "event_onset"
    capture_mode_filter: str = "event_onset"
    split_mode: str = "grouped_file"
    split_manifest_path: str = "artifacts/splits/event_onset_split_manifest.json"
    recordings_manifest_path: str = "recordings_manifest.csv"
    target_db5_keys: list[str] = field(
        default_factory=lambda: [
            "TENSE_OPEN",
            "V_SIGN",
            "OK_SIGN",
            "THUMB_UP",
            "WRIST_CW",
            "WRIST_CCW",
        ]
    )
    device_sampling_rate_hz: int = 500
    imu_sampling_rate_hz: int = 50
    clip_duration_ms: int = 1200
    pre_roll_ms: int = 400
    top_k_windows_per_clip: int = 2
    idle_top_k_windows_per_clip: int = 4
    action_window_policy: str = "onset_peak"
    action_onset_pre_ms: int = 120
    action_onset_post_ms: int = 220
    action_onset_min_gap_ms: int = 180
    action_onset_threshold_alpha: float = 0.25
    action_onset_energy_ratio_min: float = 0.65
    use_imu: bool = True
    feature: EventFeatureConfig = field(default_factory=EventFeatureConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)

    @property
    def context_samples(self) -> int:
        return self.feature.context_samples(self.device_sampling_rate_hz)

    @property
    def window_step_samples(self) -> int:
        return self.feature.step_samples(self.device_sampling_rate_hz)


@dataclass
class EventModelConfig:
    model_type: str = "event_onset"
    num_classes: int = 7
    emg_in_channels: int = 8
    emg_freq_bins: int = 24
    emg_time_frames: int = 5
    imu_input_dim: int = 6
    imu_num_steps: int = 16
    base_channels: int = 16
    imu_base_channels: int = 16
    fusion_hidden_dim: int = 64
    dropout_rate: float = 0.3
    use_se: bool = True
    pretrained_emg_checkpoint: str | None = None


@dataclass
class EventInferenceConfig:
    confidence_threshold: float = 0.75
    per_class_confidence_thresholds: dict[str, float] = field(default_factory=dict)
    vote_window: int = 3
    vote_min_count: int = 2
    activation_margin_threshold: float = 0.08
    switch_confidence_boost: float = 0.06
    switch_margin_threshold: float = 0.12


@dataclass
class EventRuntimeBehaviorConfig:
    idle_release_hold_ms: int = 700
    min_transition_gap_ms: int = 120
    post_transition_lock_ms: int = 220
    poll_interval_ms: int = 10
    low_energy_release_threshold: float | None = None
    release_mode: str = "idle_or_command"


@dataclass
class EventRuntimeConfig:
    training_config: str = "configs/training_event_onset.yaml"
    checkpoint_path: str = "checkpoints/event_onset_best.ckpt"
    model_path: str = "models/event_onset.mindir"
    model_metadata_path: str = "models/event_onset.model_metadata.json"
    actuation_mapping_path: str = "configs/event_actuation_mapping.yaml"
    data: EventDataConfig = field(default_factory=EventDataConfig)
    inference: EventInferenceConfig = field(default_factory=EventInferenceConfig)
    runtime: EventRuntimeBehaviorConfig = field(default_factory=EventRuntimeBehaviorConfig)
    device: DeviceConfig = field(default_factory=lambda: DeviceConfig(target="CPU", id=0, sampling_rate=500))
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def __post_init__(self) -> None:
        self.device.sampling_rate = int(self.data.device_sampling_rate_hz)
        self.hardware.armband_sampling_rate = int(self.data.device_sampling_rate_hz)
        self.hardware.target_sampling_rate = int(self.data.device_sampling_rate_hz)
        if self.runtime.low_energy_release_threshold is None:
            self.runtime.low_energy_release_threshold = float(self.data.quality_filter.energy_min)


def _coerce_scalar(value: Any, expected_type: type[Any]) -> Any:
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


def _dict_to_dataclass(data: dict[str, Any], cls: Type[T]) -> T:
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        annotation = type_hints.get(item.name, item.type)
        origin = get_origin(annotation)
        args = get_args(annotation)

        if hasattr(annotation, "__dataclass_fields__"):
            kwargs[item.name] = _dict_to_dataclass(value or {}, annotation)
            continue
        if origin in (list, list[str]):
            inner = args[0] if args else str
            kwargs[item.name] = [_coerce_scalar(entry, inner) for entry in value or []]
            continue
        if origin is None and annotation in (int, float, str, bool):
            kwargs[item.name] = _coerce_scalar(value, annotation)
            continue
        kwargs[item.name] = value
    return cls(**kwargs)


def _resolve_training_sections(data: dict[str, Any]) -> dict[str, Any]:
    root = dict(data or {})
    training_section = dict(root.get("training", {}))
    data_section = dict(root.get("data", {}))
    if "quality_filter" in data_section and "quality_filter" not in training_section:
        training_section["quality_filter"] = data_section["quality_filter"]
    return {
        "model": root.get("model", {}),
        "training": training_section,
        "augmentation": root.get("augmentation", {}),
        "data": data_section,
    }


def load_event_training_config(path: str | Path) -> tuple[EventModelConfig, EventDataConfig, TrainingConfig, AugmentationConfig]:
    sections = _resolve_training_sections(load_config(path))
    model = _dict_to_dataclass(sections["model"], EventModelConfig)
    data = _dict_to_dataclass(sections["data"], EventDataConfig)
    data.target_db5_keys = [
        str(item).strip().upper()
        for item in data.target_db5_keys
        if str(item).strip()
    ]
    label_spec = get_label_mode_spec(data.label_mode, data.target_db5_keys)
    model.emg_in_channels = 8
    model.emg_freq_bins = int(data.feature.emg_freq_bins)
    model.imu_input_dim = int(data.feature.imu_input_dim)
    model.imu_num_steps = int(data.feature.imu_resample_steps)
    model.num_classes = int(len(label_spec.class_names))
    model.emg_time_frames = max(
        1,
        (data.context_samples - int(data.feature.emg_stft_window)) // int(data.feature.emg_stft_hop) + 1,
    )
    training = _dict_to_dataclass(sections["training"], TrainingConfig)
    augmentation = _dict_to_dataclass(sections["augmentation"], AugmentationConfig)
    return model, data, training, augmentation


def load_event_runtime_config(path: str | Path) -> EventRuntimeConfig:
    root = load_config(path)
    payload = {
        "training_config": root.get("training_config", "configs/training_event_onset.yaml"),
        "checkpoint_path": root.get("checkpoint_path", "checkpoints/event_onset_best.ckpt"),
        "model_path": root.get("model_path", "models/event_onset.mindir"),
        "model_metadata_path": root.get("model_metadata_path", "models/event_onset.model_metadata.json"),
        "actuation_mapping_path": root.get("actuation_mapping_path", "configs/event_actuation_mapping.yaml"),
        "data": root.get("data", {}),
        "inference": root.get("inference", {}),
        "runtime": root.get("runtime", {}),
        "device": root.get("device", {}),
        "hardware": root.get("hardware", {}),
    }
    return _dict_to_dataclass(payload, EventRuntimeConfig)
