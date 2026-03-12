"""Shared label-mode definitions for training and runtime experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shared.gestures import GESTURE_DEFINITIONS


STATIC_GESTURE_LABEL_MODE = "static_gesture"
EVENT_ONSET_LABEL_MODE = "event_onset"


@dataclass(frozen=True)
class LabelModeSpec:
    label_mode: str
    gesture_to_idx: dict[str, int]
    class_names: list[str]


def _normalize_action_keys(action_keys: Sequence[str] | None) -> list[str]:
    if action_keys is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in action_keys:
        key = str(raw or "").strip().upper()
        if not key:
            continue
        if key == "RELAX":
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def get_label_mode_spec(label_mode: str | None, action_keys: Sequence[str] | None = None) -> LabelModeSpec:
    normalized = str(label_mode or STATIC_GESTURE_LABEL_MODE).strip().lower()
    if normalized == EVENT_ONSET_LABEL_MODE:
        actions = _normalize_action_keys(action_keys)
        if not actions:
            actions = ["E1_G01", "E1_G02"]
        class_names = ["RELAX", *actions]
        gesture_to_idx = {name: idx for idx, name in enumerate(class_names)}
        return LabelModeSpec(
            label_mode=EVENT_ONSET_LABEL_MODE,
            gesture_to_idx=gesture_to_idx,
            class_names=class_names,
        )

    if normalized != STATIC_GESTURE_LABEL_MODE:
        raise ValueError(
            f"Unsupported label_mode={label_mode!r}. "
            f"Expected one of: {STATIC_GESTURE_LABEL_MODE}, {EVENT_ONSET_LABEL_MODE}"
        )

    class_names = [gesture.name for gesture in GESTURE_DEFINITIONS]
    return LabelModeSpec(
        label_mode=STATIC_GESTURE_LABEL_MODE,
        gesture_to_idx={name: index for index, name in enumerate(class_names)},
        class_names=class_names,
    )
