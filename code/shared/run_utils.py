"""Helpers for consistent experiment run IDs and artifact directories."""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(tag).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "run"


def build_run_id(tag: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{sanitize_tag(tag)}"


def ensure_run_dir(run_root: str | Path, run_id: str | None, *, default_tag: str) -> tuple[str, Path]:
    resolved_run_id = run_id or build_run_id(default_tag)
    run_dir = Path(run_root) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return resolved_run_id, run_dir


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def copy_config_snapshot(src: str | Path, dest: str | Path) -> Path:
    src_path = Path(src)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dest_path)
    return dest_path


def dump_json(path: str | Path, data: Any) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return resolved


def dump_yaml(path: str | Path, data: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)
    return resolved


def append_csv_row(path: str | Path, fieldnames: Sequence[str], row: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    file_exists = resolved.exists()
    with open(resolved, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})
    return resolved
