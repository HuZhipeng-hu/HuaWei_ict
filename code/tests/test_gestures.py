"""
Unit tests for gesture definitions and mappings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.gestures import (
    FOLDER_TO_GESTURE,
    GESTURE_DEFINITIONS,
    GESTURE_FINGER_MAP,
    GESTURE_LABEL_MAP,
    LABEL_NAME_MAP,
    GestureType,
    NUM_CLASSES,
    NUM_FINGERS,
    get_finger_angles,
    validate_gesture_definitions,
)


def test_gesture_count():
    assert NUM_CLASSES == 6, f"expected 6 gesture classes, got {NUM_CLASSES}"
    assert len(GestureType) == 6


def test_gesture_values():
    values = [g.value for g in GestureType]
    assert values == list(range(NUM_CLASSES)), f"unexpected enum values: {values}"


def test_gesture_definitions_order_matches_enum():
    assert GESTURE_DEFINITIONS == tuple(GestureType)


def test_finger_map_complete():
    for gesture in GestureType:
        assert gesture in GESTURE_FINGER_MAP, f"{gesture.name} missing finger map"
        assert len(GESTURE_FINGER_MAP[gesture]) == NUM_FINGERS


def test_folder_map_complete():
    mapped = set(FOLDER_TO_GESTURE.values())
    all_gestures = set(GestureType)
    missing = all_gestures - mapped
    assert not missing, f"missing folder map for: {[g.name for g in missing]}"


def test_label_maps_consistent():
    for name, idx in GESTURE_LABEL_MAP.items():
        assert LABEL_NAME_MAP[idx] == name


def test_validate_definitions():
    assert validate_gesture_definitions() is True


def test_get_finger_angles():
    relax = get_finger_angles(GestureType.RELAX)
    assert len(relax) == NUM_FINGERS
    assert all(a == 0.0 for a in relax), f"RELAX expected all open, got {relax}"

    fist = get_finger_angles(GestureType.FIST)
    assert all(a == 180.0 for a in fist), f"FIST expected all closed, got {fist}"


def test_finger_angles_custom():
    angles = get_finger_angles(
        GestureType.YE,
        angle_open=10.0,
        angle_half=80.0,
        angle_closed=160.0,
    )
    # YE: thumb closed, index open, middle open, ring closed, pinky closed
    assert angles[0] == 160.0
    assert angles[1] == 10.0
    assert angles[2] == 10.0
    assert angles[3] == 160.0
    assert angles[4] == 160.0


if __name__ == "__main__":
    tests = [
        test_gesture_count,
        test_gesture_values,
        test_gesture_definitions_order_matches_enum,
        test_finger_map_complete,
        test_folder_map_complete,
        test_label_maps_consistent,
        test_validate_definitions,
        test_get_finger_angles,
        test_finger_angles_custom,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {test.__name__}: {exc}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
