"""Shared label-mode definitions for training and runtime experiments."""

from __future__ import annotations

from dataclasses import dataclass

from shared.gestures import GESTURE_DEFINITIONS


STATIC_GESTURE_LABEL_MODE = "static_gesture"
EVENT_ONSET_LABEL_MODE = "event_onset"


@dataclass(frozen=True)
class LabelModeSpec:
    label_mode: str
    gesture_to_idx: dict[str, int]
    class_names: list[str]


def get_label_mode_spec(label_mode: str | None) -> LabelModeSpec:
    normalized = str(label_mode or STATIC_GESTURE_LABEL_MODE).strip().lower()
    if normalized == EVENT_ONSET_LABEL_MODE:
        return LabelModeSpec(
            label_mode=EVENT_ONSET_LABEL_MODE,
            gesture_to_idx={"RELAX": 0, "FIST": 1, "PINCH": 2},
            class_names=["IDLE", "FIST", "PINCH"],
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
