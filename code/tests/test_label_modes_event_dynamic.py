from shared.label_modes import get_label_mode_spec


def test_event_label_mode_builds_dynamic_class_names():
    spec = get_label_mode_spec("event_onset", ["e1_g01", "E1_G02", "E1_G01", "RELAX"])

    assert spec.class_names == ["RELAX", "E1_G01", "E1_G02"]
    assert spec.gesture_to_idx == {"RELAX": 0, "E1_G01": 1, "E1_G02": 2}


def test_event_label_mode_uses_default_actions_when_none_provided():
    spec = get_label_mode_spec("event_onset", [])
    assert spec.class_names == ["RELAX", "E1_G01", "E1_G02"]
