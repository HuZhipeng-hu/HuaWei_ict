"""Configuration for NinaPro DB5 pretraining."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints

from shared.config import TrainingConfig, load_config

T = TypeVar("T")


@dataclass
class DB5FeatureConfig:
    source_sampling_rate_hz: int = 200
    target_sampling_rate_hz: int = 500
    use_first_myo_only: bool = True
    first_myo_channel_count: int = 8
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
class DB5OnsetWindowPolicy:
    action_selection: str = "peak_energy"
    rest_selection: str = "low_energy"
    action_top_k_per_segment: int = 3
    rest_top_k_per_segment: int = 1


@dataclass
class DB5MappingProfileConfig:
    name: str
    fist: list[str] = field(default_factory=list)
    pinch: list[str] = field(default_factory=list)


@dataclass
class DB5Aligned3Config:
    enabled: bool = True
    candidate_mapping_profiles: list[DB5MappingProfileConfig] = field(
        default_factory=lambda: [
            DB5MappingProfileConfig(
                name="p1",
                fist=["E1_G01", "E2_G01", "E3_G01"],
                pinch=["E1_G02", "E2_G02", "E3_G02"],
            ),
            DB5MappingProfileConfig(
                name="p2",
                fist=["E1_G13", "E2_G13", "E3_G13"],
                pinch=["E1_G14", "E2_G14", "E3_G14"],
            ),
            DB5MappingProfileConfig(
                name="p3",
                fist=["E1_G17", "E2_G17", "E3_G17"],
                pinch=["E1_G18", "E2_G18", "E3_G18"],
            ),
        ]
    )
    mapping_override: Optional[DB5MappingProfileConfig] = None
    min_samples_per_class: int = 60
    onset_window_policy: DB5OnsetWindowPolicy = field(default_factory=DB5OnsetWindowPolicy)


@dataclass
class DB5PretrainConfig:
    data_dir: str = "../data_ninaproDB5"
    zip_glob: str = "s*.zip"
    include_rest_class: bool = True
    use_restimulus: bool = True
    aligned3: DB5Aligned3Config = field(default_factory=DB5Aligned3Config)
    feature: DB5FeatureConfig = field(default_factory=DB5FeatureConfig)
    model_type: str = "db5_pretrain"
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
        annotation = type_hints.get(item.name, item.type)
        if value is None:
            kwargs[item.name] = None
            continue
        annotation = _resolve_optional(annotation)
        origin = get_origin(annotation)
        args = get_args(annotation)
        if _is_dataclass_type(annotation):
            kwargs[item.name] = _dict_to_dataclass(value or {}, annotation)
            continue
        if origin in (list, list[str]):
            inner = _resolve_optional(args[0]) if args else str
            if _is_dataclass_type(inner):
                kwargs[item.name] = [_dict_to_dataclass(entry or {}, inner) for entry in value or []]
            else:
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
        "include_rest_class": root.get("include_rest_class", True),
        "use_restimulus": root.get("use_restimulus", True),
        "aligned3": root.get("aligned3", {}),
        "feature": root.get("feature", {}),
        "model_type": root.get("model_type", "db5_pretrain"),
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
