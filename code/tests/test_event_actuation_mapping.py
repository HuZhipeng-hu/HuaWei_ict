from pathlib import Path

from event_onset.actuation_mapping import load_and_validate_actuation_map
from shared.gestures import GestureType


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _assert_raises_value_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_actuation_mapping_loads_valid_mapping(tmp_path: Path):
    mapping = tmp_path / "map.yaml"
    _write_yaml(
        mapping,
        "\n".join(
            [
                "actuation_map:",
                "  RELAX: RELAX",
                "  E1_G01: FIST",
                "  E1_G02: PINCH",
            ]
        ),
    )
    label_to_state, by_name = load_and_validate_actuation_map(
        mapping,
        class_names=["RELAX", "E1_G01", "E1_G02"],
    )

    assert by_name == {"RELAX": "RELAX", "E1_G01": "FIST", "E1_G02": "PINCH"}
    assert label_to_state[0] == GestureType.RELAX
    assert label_to_state[1] == GestureType.FIST
    assert label_to_state[2] == GestureType.PINCH


def test_actuation_mapping_rejects_missing_and_extra_keys(tmp_path: Path):
    mapping = tmp_path / "map.yaml"
    _write_yaml(
        mapping,
        "\n".join(
            [
                "actuation_map:",
                "  RELAX: RELAX",
                "  E1_G01: FIST",
                "  E9_G99: OK",
            ]
        ),
    )
    _assert_raises_value_error(load_and_validate_actuation_map, mapping, class_names=["RELAX", "E1_G01", "E1_G02"])


def test_actuation_mapping_requires_relax_to_relax(tmp_path: Path):
    mapping = tmp_path / "map.yaml"
    _write_yaml(
        mapping,
        "\n".join(
            [
                "actuation_map:",
                "  RELAX: FIST",
                "  E1_G01: FIST",
                "  E1_G02: PINCH",
            ]
        ),
    )
    _assert_raises_value_error(load_and_validate_actuation_map, mapping, class_names=["RELAX", "E1_G01", "E1_G02"])


def test_actuation_mapping_loads_demo_six_action_mapping(tmp_path: Path):
    mapping = tmp_path / "map.yaml"
    _write_yaml(
        mapping,
        "\n".join(
            [
                "actuation_map:",
                "  RELAX: RELAX",
                "  TENSE_OPEN: TENSE_OPEN",
                "  V_SIGN: V_SIGN",
                "  OK_SIGN: OK_SIGN",
                "  THUMB_UP: THUMB_UP",
                "  WRIST_CW: WRIST_CW",
                "  WRIST_CCW: WRIST_CCW",
            ]
        ),
    )
    class_names = ["RELAX", "TENSE_OPEN", "V_SIGN", "OK_SIGN", "THUMB_UP", "WRIST_CW", "WRIST_CCW"]
    label_to_state, by_name = load_and_validate_actuation_map(mapping, class_names=class_names)

    assert by_name["RELAX"] == "RELAX"
    assert by_name["WRIST_CW"] == "WRIST_CW"
    assert by_name["WRIST_CCW"] == "WRIST_CCW"
    assert label_to_state[0] == GestureType.RELAX
    assert label_to_state[5] == GestureType.WRIST_CW
    assert label_to_state[6] == GestureType.WRIST_CCW


def test_actuation_mapping_allows_tense_open_as_release_to_relax(tmp_path: Path):
    mapping = tmp_path / "map.yaml"
    _write_yaml(
        mapping,
        "\n".join(
            [
                "actuation_map:",
                "  RELAX: RELAX",
                "  TENSE_OPEN: RELAX",
                "  THUMB_UP: THUMB_UP",
                "  WRIST_CW: WRIST_CW",
                "  WRIST_CCW: WRIST_CCW",
            ]
        ),
    )
    class_names = ["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW", "WRIST_CCW"]
    label_to_state, by_name = load_and_validate_actuation_map(mapping, class_names=class_names)

    assert by_name["TENSE_OPEN"] == "RELAX"
    assert label_to_state[0] == GestureType.RELAX
    assert label_to_state[1] == GestureType.RELAX
    assert label_to_state[2] == GestureType.THUMB_UP


def test_actuation_mapping_accepts_continue_alias_in_yaml(tmp_path: Path):
    mapping = tmp_path / "map.yaml"
    _write_yaml(
        mapping,
        "\n".join(
            [
                "actuation_map:",
                "  CONTINUE: CONTINUE",
                "  TENSE_OPEN: CONTINUE",
                "  THUMB_UP: THUMB_UP",
            ]
        ),
    )
    label_to_state, by_name = load_and_validate_actuation_map(
        mapping,
        class_names=["RELAX", "TENSE_OPEN", "THUMB_UP"],
    )

    assert by_name["RELAX"] == "RELAX"
    assert by_name["TENSE_OPEN"] == "RELAX"
    assert by_name["THUMB_UP"] == "THUMB_UP"
    assert label_to_state[0] == GestureType.RELAX
    assert label_to_state[1] == GestureType.RELAX
