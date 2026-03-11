"""Manifest helpers for event-onset recording metadata."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


EVENT_MANIFEST_FIELDS = [
    "relative_path",
    "gesture",
    "capture_mode",
    "start_state",
    "target_state",
    "user_id",
    "session_id",
    "device_id",
    "timestamp",
    "wearing_state",
    "recording_id",
    "sample_count",
    "clip_duration_ms",
    "pre_roll_ms",
    "device_sampling_rate_hz",
    "imu_sampling_rate_hz",
    "quality_status",
    "quality_reasons",
    "source_origin",
]


def normalize_relative_path(path_value: str | Path) -> str:
    return Path(str(path_value).replace("\\", "/")).as_posix()


def load_event_manifest_rows(manifest_path: str | Path) -> dict[str, dict[str, str]]:
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
            entry = {field: str(row.get(field, "")) for field in EVENT_MANIFEST_FIELDS}
            for key, value in row.items():
                if key not in entry:
                    entry[key] = str(value or "")
            entry["relative_path"] = normalized
            entries[normalized] = entry
    return entries


def upsert_event_manifest(manifest_path: str | Path, row: Mapping[str, Any]) -> Path:
    resolved = Path(manifest_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    entries = load_event_manifest_rows(resolved)

    normalized_relative = normalize_relative_path(row["relative_path"])
    merged = {field: "" for field in EVENT_MANIFEST_FIELDS}
    for field in EVENT_MANIFEST_FIELDS:
        value = row.get(field, "")
        if field == "relative_path":
            value = normalized_relative
        merged[field] = str(value)
    entries[normalized_relative] = merged

    with open(resolved, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_MANIFEST_FIELDS)
        writer.writeheader()
        for item in sorted(entries.values(), key=lambda current: current["relative_path"]):
            writer.writerow(item)
    return resolved
