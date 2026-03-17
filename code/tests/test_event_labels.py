from shared.event_labels import (
    is_continue_label,
    normalize_event_label_input,
    public_event_label,
    public_event_labels,
)


def test_event_label_aliases_normalize_to_internal_relax() -> None:
    assert is_continue_label("relax") is True
    assert is_continue_label("continue") is True
    assert normalize_event_label_input("CONTINUE") == "RELAX"
    assert normalize_event_label_input("RELAX") == "RELAX"


def test_event_label_public_output_prefers_continue() -> None:
    assert public_event_label("RELAX") == "CONTINUE"
    assert public_event_label("continue") == "CONTINUE"
    assert public_event_labels(["RELAX", "THUMB_UP"]) == ["CONTINUE", "THUMB_UP"]
