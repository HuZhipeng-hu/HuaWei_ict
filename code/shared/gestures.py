"""
Gesture definitions and prosthesis finger mappings.
"""

from enum import IntEnum
from typing import Dict, List, Tuple


class FingerState(IntEnum):
    OPEN = 0
    HALF = 1
    CLOSED = 2


class GestureType(IntEnum):
    RELAX = 0
    FIST = 1
    PINCH = 2
    OK = 3
    YE = 4
    SIDEGRIP = 5
    TENSE_OPEN = 6
    V_SIGN = 7
    OK_SIGN = 8
    THUMB_UP = 9
    WRIST_CW = 10
    WRIST_CCW = 11


NUM_CLASSES = len(GestureType)
GESTURE_DEFINITIONS: Tuple[GestureType, ...] = tuple(GestureType)

GESTURE_LABEL_MAP: Dict[str, int] = {
    gesture.name.lower(): gesture.value
    for gesture in GestureType
}

LABEL_NAME_MAP: Dict[int, str] = {
    value: name for name, value in GESTURE_LABEL_MAP.items()
}

FOLDER_TO_GESTURE: Dict[str, GestureType] = {
    "relax": GestureType.RELAX,
    "fist": GestureType.FIST,
    "pinch": GestureType.PINCH,
    "ok": GestureType.OK,
    "ye": GestureType.YE,
    "sidegrip": GestureType.SIDEGRIP,
    "tense_open": GestureType.TENSE_OPEN,
    "v_sign": GestureType.V_SIGN,
    "vsign": GestureType.V_SIGN,
    "ok_sign": GestureType.OK_SIGN,
    "oksign": GestureType.OK_SIGN,
    "thumb_up": GestureType.THUMB_UP,
    "thumbup": GestureType.THUMB_UP,
    "wrist_cw": GestureType.WRIST_CW,
    "wristcw": GestureType.WRIST_CW,
    "wrist_ccw": GestureType.WRIST_CCW,
    "wristccw": GestureType.WRIST_CCW,
}

FINGER_THUMB = 0
FINGER_INDEX = 1
FINGER_MIDDLE = 2
FINGER_RING = 3
FINGER_PINKY = 4

NUM_FINGERS = 5
NUM_EMG_CHANNELS = 8

GESTURE_FINGER_MAP: Dict[GestureType, List[FingerState]] = {
    GestureType.RELAX: [
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.OPEN,
    ],
    GestureType.FIST: [
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
    ],
    GestureType.PINCH: [
        FingerState.HALF,
        FingerState.HALF,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.OPEN,
    ],
    GestureType.OK: [
        FingerState.HALF,
        FingerState.HALF,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.OPEN,
    ],
    GestureType.YE: [
        FingerState.CLOSED,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.CLOSED,
        FingerState.CLOSED,
    ],
    GestureType.SIDEGRIP: [
        FingerState.HALF,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
    ],
    GestureType.TENSE_OPEN: [
        FingerState.HALF,
        FingerState.HALF,
        FingerState.HALF,
        FingerState.HALF,
        FingerState.HALF,
    ],
    GestureType.V_SIGN: [
        FingerState.CLOSED,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.CLOSED,
        FingerState.CLOSED,
    ],
    GestureType.OK_SIGN: [
        FingerState.HALF,
        FingerState.HALF,
        FingerState.OPEN,
        FingerState.OPEN,
        FingerState.OPEN,
    ],
    GestureType.THUMB_UP: [
        FingerState.OPEN,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
    ],
    GestureType.WRIST_CW: [
        FingerState.HALF,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.HALF,
    ],
    GestureType.WRIST_CCW: [
        FingerState.HALF,
        FingerState.HALF,
        FingerState.CLOSED,
        FingerState.CLOSED,
        FingerState.CLOSED,
    ],
}


def validate_gesture_definitions() -> bool:
    for gesture in GestureType:
        if gesture not in GESTURE_FINGER_MAP:
            raise ValueError(f"Gesture {gesture.name} missing finger mapping")
        finger_states = GESTURE_FINGER_MAP[gesture]
        if len(finger_states) != NUM_FINGERS:
            raise ValueError(
                f"Gesture {gesture.name} finger mapping has {len(finger_states)} entries, expected {NUM_FINGERS}"
            )

    mapped_gestures = set(FOLDER_TO_GESTURE.values())
    all_gestures = set(GestureType)
    missing = all_gestures - mapped_gestures
    if missing:
        raise ValueError(f"Missing folder mappings for: {[g.name for g in missing]}")

    return True


def get_finger_angles(
    gesture: GestureType,
    angle_open: float = 0.0,
    angle_half: float = 90.0,
    angle_closed: float = 180.0,
) -> List[float]:
    state_to_angle = {
        FingerState.OPEN: angle_open,
        FingerState.HALF: angle_half,
        FingerState.CLOSED: angle_closed,
    }
    return [state_to_angle[state] for state in GESTURE_FINGER_MAP[gesture]]
