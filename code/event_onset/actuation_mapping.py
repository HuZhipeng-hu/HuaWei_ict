"""Load and validate runtime class-to-actuator gesture mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from shared.config import load_config
from shared.gestures import GestureType


def _normalize_key(value: str) -> str:
    return str(value or "").strip().upper()


def load_and_validate_actuation_map(
    mapping_path: str | Path,
    *,
    class_names: Sequence[str],
) -> tuple[dict[int, GestureType], dict[str, str]]:
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Actuation mapping file not found: {path}")

    root = load_config(path)
    raw_map = root.get("actuation_map", root)
    if not isinstance(raw_map, dict):
        raise ValueError("actuation mapping must be a mapping object.")

    normalized_class_names = [_normalize_key(name) for name in class_names]
    required = set(normalized_class_names)
    provided = {_normalize_key(key) for key in raw_map.keys()}
    missing = sorted(required - provided)
    extra = sorted(provided - required)
    if missing or extra:
        raise ValueError(
            f"actuation mapping keys mismatch: missing={missing}, extra={extra}, required={sorted(required)}"
        )

    name_to_gesture = {gesture.name: gesture for gesture in GestureType}
    mapping_by_name: dict[str, str] = {}
    label_to_state: dict[int, GestureType] = {}
    for idx, class_name in enumerate(normalized_class_names):
        raw_target = _normalize_key(raw_map.get(class_name))
        if raw_target not in name_to_gesture:
            raise ValueError(
                f"Invalid actuator gesture={raw_target!r} for class={class_name!r}. "
                f"Allowed={sorted(name_to_gesture.keys())}"
            )
        mapping_by_name[class_name] = raw_target
        label_to_state[idx] = name_to_gesture[raw_target]

    if mapping_by_name.get("RELAX") != "RELAX":
        raise ValueError("Class RELAX must map to actuator gesture RELAX.")

    return label_to_state, mapping_by_name

