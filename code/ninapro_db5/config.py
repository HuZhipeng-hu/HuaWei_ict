"""Configuration for NinaPro DB5 full-coverage foundation pretraining."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar, Union, get_args, get_origin, get_type_hints

from shared.config import TrainingConfig, load_config

T = TypeVar("T")


@dataclass
class DB5FeatureConfig:
    source_sampling_rate_hz: int = 200
    target_sampling_rate_hz: int = 500
    use_first_myo_only: bool = False
    first_myo_channel_count: int = 16
    lowcut_hz: float = 20.0
    highcut_hz: float = 180.0
    energy_min: float = 0.25
    static_std_min: float = 0.08
    clip_ratio_max: float = 0.08
    saturation_abs: float = 126.0
    context_window_ms: int = 240
    window_step_ms: int = 80
    max_windows_per_segment: int = 6
    max_rest_windows_per_segment: int = 2
    emg_stft_window: int = 64
    emg_stft_hop: int = 24
    emg_n_fft: int = 96
    emg_freq_bins: int = 24

    @property
    def source_window_samples(self) -> int:
        return max(1, int(round(self.context_window_ms * self.source_sampling_rate_hz / 1000.0)))

    @property
    def source_step_samples(self) -> int:
        return max(1, int(round(self.window_step_ms * self.source_sampling_rate_hz / 1000.0)))

    @property
    def target_window_samples(self) -> int:
        return max(1, int(round(self.context_window_ms * self.target_sampling_rate_hz / 1000.0)))


@dataclass
class DB5PretrainConfig:
    data_dir: str = "../data_ninaproDB5"
    zip_glob: str = "s*.zip"
    include_rest_class: bool = False
    use_restimulus: bool = True
    feature: DB5FeatureConfig = field(default_factory=DB5FeatureConfig)
    model_type: str = "db5_pretrain_full53"
    foundation_version: str = "db5_full53_v1"
    base_channels: int = 16
    use_se: bool = True
    dropout_rate: float = 0.3
    classifier_hidden_dim: int = 64
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split_seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15


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


def _is_dataclass_type(annotation: Any) -> bool:
    return hasattr(annotation, "__dataclass_fields__")


def _resolve_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _dict_to_dataclass(data: dict[str, Any], cls: Type[T]) -> T:
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        if value is None:
            kwargs[item.name] = None
            continue

        annotation = _resolve_optional(type_hints.get(item.name, item.type))
        origin = get_origin(annotation)
        args = get_args(annotation)

        if _is_dataclass_type(annotation):
            kwargs[item.name] = _dict_to_dataclass(value or {}, annotation)
            continue
        if origin in (list, list[str]):
            inner = _resolve_optional(args[0]) if args else str
            kwargs[item.name] = [_coerce_scalar(entry, inner) for entry in value or []]
            continue
        if origin is None and annotation in (int, float, str, bool):
            kwargs[item.name] = _coerce_scalar(value, annotation)
            continue
        kwargs[item.name] = value
    return cls(**kwargs)


def load_db5_pretrain_config(path: str | Path) -> DB5PretrainConfig:
    root = load_config(path)
    payload = {
        "data_dir": root.get("data_dir", "../data_ninaproDB5"),
        "zip_glob": root.get("zip_glob", "s*.zip"),
        "include_rest_class": root.get("include_rest_class", False),
        "use_restimulus": root.get("use_restimulus", True),
        "feature": root.get("feature", {}),
        "model_type": root.get("model_type", "db5_pretrain_full53"),
        "foundation_version": root.get("foundation_version", "db5_full53_v1"),
        "base_channels": root.get("base_channels", 16),
        "use_se": root.get("use_se", True),
        "dropout_rate": root.get("dropout_rate", 0.3),
        "classifier_hidden_dim": root.get("classifier_hidden_dim", 64),
        "training": root.get("training", {}),
        "split_seed": root.get("split_seed", 42),
        "val_ratio": root.get("val_ratio", 0.15),
        "test_ratio": root.get("test_ratio", 0.15),
    }
    return _dict_to_dataclass(payload, DB5PretrainConfig)
