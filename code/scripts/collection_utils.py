"""Collection-only helpers for recording import, QC, and manifest upkeep."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shared.config import (
    DualBranchConfig,
    PreprocessConfig,
    QualityFilterConfig,
    load_config,
    load_training_config,
)
from shared.gestures import GESTURE_DEFINITIONS, NUM_EMG_CHANNELS
from shared.preprocessing import PreprocessPipeline

STANDARD_CSV_HEADERS = [
    "emg1",
    "emg2",
    "emg3",
    "emg4",
    "emg5",
    "emg6",
    "emg7",
    "emg8",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "angle_pitch",
    "angle_roll",
    "angle_yaw",
]

TOTAL_EMG_CHANNEL_COUNT = NUM_EMG_CHANNELS
DEFAULT_RECORDINGS_MANIFEST = "recordings_manifest.csv"
DEFAULT_REPORT_DIRNAME = "collection_reports"
QUALITY_PASS = "pass"
QUALITY_WARN = "warn"
QUALITY_RETAKE = "retake_recommended"

MANIFEST_FIELDS = [
    "relative_path",
    "gesture",
    "user_id",
    "session_id",
    "device_id",
    "timestamp",
    "wearing_state",
    "recording_id",
    "sample_count",
    "quality_status",
    "quality_reasons",
    "source_origin",
]

_GESTURE_LOOKUP = {
    re.sub(r"[^a-z0-9]+", "", gesture.name.lower()): gesture.name
    for gesture in GESTURE_DEFINITIONS
}

_SOURCE_COLUMN_ALIASES = {
    "emg1": ("emg1", "ch1"),
    "emg2": ("emg2", "ch2"),
    "emg3": ("emg3", "ch3"),
    "emg4": ("emg4", "ch4"),
    "emg5": ("emg5", "ch5"),
    "emg6": ("emg6", "ch6"),
    "emg7": ("emg7", "ch7"),
    "emg8": ("emg8", "ch8"),
    "acc_x": ("acc_x", "accx", "ax"),
    "acc_y": ("acc_y", "accy", "ay"),
    "acc_z": ("acc_z", "accz", "az"),
    "gyro_x": ("gyro_x", "gyrox", "gx"),
    "gyro_y": ("gyro_y", "gyroy", "gy"),
    "gyro_z": ("gyro_z", "gyroz", "gz"),
    "angle_pitch": ("angle_pitch", "pitch"),
    "angle_roll": ("angle_roll", "roll"),
    "angle_yaw": ("angle_yaw", "yaw"),
}

_REQUIRED_SOURCE_COLUMNS = [f"emg{i}" for i in range(1, TOTAL_EMG_CHANNEL_COUNT + 1)]


@dataclass(frozen=True)
class RecordingMetadata:
    """Required metadata for a standardized recording."""

    gesture: str
    user_id: str
    session_id: str
    device_id: str
    wearing_state: str


def normalize_gesture_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    if not cleaned:
        raise ValueError("Gesture cannot be empty")
    gesture = _GESTURE_LOOKUP.get(cleaned)
    if gesture is None:
        valid = ", ".join(item.name for item in GESTURE_DEFINITIONS)
        raise ValueError(f"Unsupported gesture {value!r}. Expected one of: {valid}")
    return gesture


def sanitize_token(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "unknown"


def validate_metadata(
    *,
    gesture: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
) -> RecordingMetadata:
    missing = [
        name
        for name, value in (
            ("gesture", gesture),
            ("user_id", user_id),
            ("session_id", session_id),
            ("device_id", device_id),
            ("wearing_state", wearing_state),
        )
        if not str(value).strip()
    ]
    if missing:
        raise ValueError(f"Missing required metadata: {', '.join(missing)}")

    return RecordingMetadata(
        gesture=normalize_gesture_name(gesture),
        user_id=sanitize_token(user_id),
        session_id=sanitize_token(session_id),
        device_id=sanitize_token(device_id),
        wearing_state=sanitize_token(wearing_state),
    )


def build_timestamp(value: datetime | None = None) -> str:
    timestamp = value or datetime.now()
    return timestamp.strftime("%Y%m%dT%H%M%S")


def timestamp_from_path(path: str | Path) -> str:
    source = Path(path)
    return build_timestamp(datetime.fromtimestamp(source.stat().st_mtime))


def load_collection_protocol(
    training_config_path: str | Path,
) -> tuple[PreprocessConfig, QualityFilterConfig]:
    raw = load_config(training_config_path)
    label_mode = str((raw.get("data", {}) or {}).get("label_mode", "")).strip().lower()
    if label_mode == "event_onset":
        from event_onset.config import load_event_training_config

        _, data_cfg, training_cfg, _ = load_event_training_config(training_config_path)
        preprocess_cfg = PreprocessConfig(
            sampling_rate=int(data_cfg.device_sampling_rate_hz),
            num_channels=8,
            target_length=int(data_cfg.context_samples),
            overlap=max(0.0, min(0.99, 1.0 - float(data_cfg.window_step_samples) / float(data_cfg.context_samples))),
            stft_window=int(data_cfg.feature.emg_stft_window),
            stft_hop=int(data_cfg.feature.emg_stft_hop),
            n_fft=int(data_cfg.feature.emg_n_fft),
            freq_bins_out=int(data_cfg.feature.emg_freq_bins),
            dual_branch=DualBranchConfig(enabled=False),
        )
        return preprocess_cfg, training_cfg.quality_filter

    _, preprocess_cfg, training_cfg, _ = load_training_config(training_config_path)
    return preprocess_cfg, training_cfg.quality_filter


def resolve_manifest_path(data_dir: str | Path, manifest_path: str | Path | None) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)
    return Path(data_dir) / DEFAULT_RECORDINGS_MANIFEST


def resolve_report_dir(data_dir: str | Path, report_dir: str | Path | None) -> Path:
    if report_dir is not None:
        return Path(report_dir)
    return Path(data_dir) / DEFAULT_REPORT_DIRNAME


def normalize_relative_path(path_value: str | Path) -> str:
    return Path(str(path_value).replace("\\", "/")).as_posix()


def build_relative_recording_path(
    metadata: RecordingMetadata,
    *,
    timestamp: str,
    recording_index: int,
) -> Path:
    stem = (
        f"{timestamp}"
        f"__u-{metadata.user_id}"
        f"__s-{metadata.session_id}"
        f"__d-{metadata.device_id}"
        f"__w-{metadata.wearing_state}"
        f"__n-{int(recording_index):03d}"
    )
    return Path(metadata.gesture) / f"{stem}.csv"


def ensure_unique_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        next_candidate = candidate.with_name(f"{candidate.stem}_{suffix:02d}{candidate.suffix}")
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def _resolve_source_field_map(fieldnames: Sequence[str]) -> dict[str, str | None]:
    lowered = {str(name).strip().lower(): name for name in fieldnames}
    mapping: dict[str, str | None] = {}
    for target in STANDARD_CSV_HEADERS:
        source_name = None
        for alias in _SOURCE_COLUMN_ALIASES[target]:
            if alias in lowered:
                source_name = lowered[alias]
                break
        mapping[target] = source_name

    missing = [name for name in _REQUIRED_SOURCE_COLUMNS if mapping[name] is None]
    if missing:
        readable = ", ".join(sorted(fieldnames))
        raise ValueError(
            "CSV is missing required EMG columns for training compatibility: "
            f"{', '.join(missing)}. Got headers: {readable}"
        )
    return mapping


def read_source_csv(source_path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(source_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {source_path}")
        field_map = _resolve_source_field_map(reader.fieldnames)
        for row in reader:
            rows.append(
                [
                    _coerce_float(row.get(field_map[column]), default=0.0) if field_map[column] else 0.0
                    for column in STANDARD_CSV_HEADERS
                ]
            )

    if not rows:
        raise ValueError(f"CSV has no data rows: {source_path}")

    return np.asarray(rows, dtype=np.float32)


def write_standard_csv(path: str | Path, matrix: np.ndarray) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(STANDARD_CSV_HEADERS):
        raise ValueError(
            f"Expected matrix shape (N, {len(STANDARD_CSV_HEADERS)}), got {tuple(array.shape)}"
        )

    with open(resolved, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STANDARD_CSV_HEADERS)
        for row in array:
            writer.writerow([float(value) for value in row.tolist()])
    return resolved


def frame_to_standard_rows(frame: Mapping[str, Any]) -> list[list[float]]:
    acc = frame.get("acc") or {}
    gyro = frame.get("gyro") or {}
    angle = frame.get("angle") or {}
    emg_packs = frame.get("emg") or []
    rows: list[list[float]] = []
    for pack in emg_packs:
        if len(pack) < TOTAL_EMG_CHANNEL_COUNT:
            raise ValueError(f"EMG pack has {len(pack)} channels, expected {TOTAL_EMG_CHANNEL_COUNT}")
        rows.append(
            [
                float(pack[0]),
                float(pack[1]),
                float(pack[2]),
                float(pack[3]),
                float(pack[4]),
                float(pack[5]),
                float(pack[6]),
                float(pack[7]),
                float(acc.get("x", 0.0)),
                float(acc.get("y", 0.0)),
                float(acc.get("z", 0.0)),
                float(gyro.get("x", 0.0)),
                float(gyro.get("y", 0.0)),
                float(gyro.get("z", 0.0)),
                float(angle.get("pitch", 0.0)),
                float(angle.get("roll", 0.0)),
                float(angle.get("yaw", 0.0)),
            ]
        )
    return rows


def _normalize_emg_domain(emg: np.ndarray) -> np.ndarray:
    normalized = np.asarray(emg, dtype=np.float32).copy()
    if normalized.size and float(np.min(normalized)) >= 0.0 and float(np.max(normalized)) > 64.0:
        normalized -= 128.0
    return normalized


def _segment_quality_reasons(segment: np.ndarray, quality_filter: QualityFilterConfig) -> list[str]:
    if not quality_filter.enabled:
        return []

    reasons: list[str] = []
    mean_abs = float(np.mean(np.abs(segment)))
    if mean_abs < float(quality_filter.energy_min):
        reasons.append("low_energy")

    clip_ratio = float(np.mean(np.abs(segment) >= float(quality_filter.saturation_abs)))
    if clip_ratio > float(quality_filter.clip_ratio_max):
        reasons.append("clipped")

    mean_std = float(np.mean(np.std(segment, axis=0)))
    if mean_std < float(quality_filter.static_std_max):
        reasons.append("static")

    return reasons


def _detect_channel_anomaly(signal: np.ndarray, quality_filter: QualityFilterConfig) -> dict[str, list[int]]:
    channel_std = np.std(signal, axis=0)
    channel_energy = np.mean(np.abs(signal), axis=0)

    median_std = float(np.median(channel_std)) if channel_std.size else 0.0
    median_energy = float(np.median(channel_energy)) if channel_energy.size else 0.0

    dead_threshold = max(float(quality_filter.static_std_max) * 0.35, median_std * 0.2)
    weak_threshold = median_energy * 0.15 if median_energy > 0.0 else 0.0
    dominant_threshold = max(5.0, median_energy * 4.0)

    dead_channels = [int(index) + 1 for index, value in enumerate(channel_std) if float(value) <= dead_threshold]
    weak_channels = [
        int(index) + 1
        for index, value in enumerate(channel_energy)
        if median_energy > 1.0 and float(value) <= weak_threshold
    ]
    dominant_channels = [
        int(index) + 1 for index, value in enumerate(channel_energy) if float(value) >= dominant_threshold
    ]

    return {
        "dead_channels": sorted(set(dead_channels)),
        "weak_channels": sorted(set(weak_channels)),
        "dominant_channels": sorted(set(dominant_channels)),
    }


def _rounded_list(values: Iterable[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def evaluate_recording_quality(
    matrix: np.ndarray,
    *,
    preprocess_config: PreprocessConfig,
    quality_filter: QualityFilterConfig,
) -> dict[str, Any]:
    array = np.asarray(matrix, dtype=np.float32)
    expected_channels = int(preprocess_config.num_channels)
    if array.ndim != 2 or array.shape[1] < expected_channels:
        raise ValueError("Recording matrix must have EMG columns in positions 1-8")

    emg_all = _normalize_emg_domain(array[:, :TOTAL_EMG_CHANNEL_COUNT])
    emg_train = emg_all[:, :expected_channels]

    pipeline = PreprocessPipeline(preprocess_config)
    required_window = int(pipeline.get_required_window_size())
    stride = int(pipeline.get_required_window_stride())
    segments = pipeline.extract_segments(emg_train)

    reason_counter: Counter[str] = Counter()
    kept_segments = 0
    for segment in segments:
        reasons = _segment_quality_reasons(segment, quality_filter)
        if reasons:
            reason_counter.update(reasons)
            continue
        kept_segments += 1

    anomaly = _detect_channel_anomaly(emg_train, quality_filter)
    if any(anomaly.values()):
        reason_counter["channel_anomaly"] += 1

    if emg_train.shape[0] < required_window:
        reason_counter["length_insufficient"] += 1

    unique_reasons = sorted(reason_counter.keys())
    total_segments = len(segments)
    kept_ratio = float(kept_segments / total_segments) if total_segments else 0.0

    status = QUALITY_PASS
    if "length_insufficient" in reason_counter or total_segments == 0 or kept_segments == 0:
        status = QUALITY_RETAKE
    elif unique_reasons or kept_ratio < 1.0:
        status = QUALITY_WARN

    return {
        "row_count": int(array.shape[0]),
        "required_window_size": required_window,
        "window_stride": stride,
        "total_segments": total_segments,
        "kept_segments": kept_segments,
        "kept_ratio": round(kept_ratio, 4),
        "training_channel_count": expected_channels,
        "emg_channel_mean_abs": _rounded_list(np.mean(np.abs(emg_train), axis=0)),
        "emg_channel_std": _rounded_list(np.std(emg_train, axis=0)),
        "quality_status": status,
        "quality_reasons": unique_reasons,
        "reason_counts": {key: int(value) for key, value in sorted(reason_counter.items())},
        "channel_anomaly": anomaly,
    }


def build_manifest_row(
    *,
    relative_path: str | Path,
    metadata: RecordingMetadata,
    timestamp: str,
    sample_count: int,
    quality_report: Mapping[str, Any],
    source_origin: str,
) -> dict[str, str]:
    normalized_relative = normalize_relative_path(relative_path)
    return {
        "relative_path": normalized_relative,
        "gesture": metadata.gesture,
        "user_id": metadata.user_id,
        "session_id": metadata.session_id,
        "device_id": metadata.device_id,
        "timestamp": timestamp,
        "wearing_state": metadata.wearing_state,
        "recording_id": Path(normalized_relative).stem,
        "sample_count": str(int(sample_count)),
        "quality_status": str(quality_report.get("quality_status", QUALITY_WARN)),
        "quality_reasons": "|".join(quality_report.get("quality_reasons", [])),
        "source_origin": source_origin,
    }


def load_manifest_rows(manifest_path: str | Path) -> dict[str, dict[str, str]]:
    resolved = Path(manifest_path)
    if not resolved.exists():
        return {}

    entries: dict[str, dict[str, str]] = {}
    with open(resolved, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row.get("relative_path")
            if not relative_path:
                continue
            normalized = normalize_relative_path(relative_path)
            entry = {field: str(row.get(field, "")) for field in MANIFEST_FIELDS}
            entry["relative_path"] = normalized
            entries[normalized] = entry
    return entries


def upsert_recordings_manifest(manifest_path: str | Path, row: Mapping[str, Any]) -> Path:
    resolved = Path(manifest_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    entries = load_manifest_rows(resolved)

    normalized_relative = normalize_relative_path(row["relative_path"])
    merged = {field: "" for field in MANIFEST_FIELDS}
    for field in MANIFEST_FIELDS:
        value = row.get(field, "")
        if field == "relative_path":
            value = normalized_relative
        merged[field] = str(value)
    entries[normalized_relative] = merged

    with open(resolved, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for key in sorted(entries):
            writer.writerow(entries[key])
    return resolved


def gather_source_csvs(
    *,
    source_dir: str | Path | None = None,
    source_csvs: Sequence[str | Path] | None = None,
) -> list[Path]:
    items = [Path(item) for item in (source_csvs or [])]
    if source_dir is not None:
        items.extend(sorted(Path(source_dir).glob("*.csv")))

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.resolve())
        if key in seen:
            continue
        if not item.exists():
            raise FileNotFoundError(f"Source CSV not found: {item}")
        deduped.append(item)
        seen.add(key)

    if not deduped:
        raise ValueError("No source CSVs provided")
    return deduped


def write_json_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
    return resolved


def build_quality_console_line(report: Mapping[str, Any]) -> str:
    reasons = ",".join(report.get("quality_reasons", [])) or "none"
    return (
        f"status={report.get('quality_status')} "
        f"rows={report.get('row_count')} "
        f"segments={report.get('kept_segments')}/{report.get('total_segments')} "
        f"reasons={reasons}"
    )
