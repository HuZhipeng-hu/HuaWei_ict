from __future__ import annotations

from scripts.build_s2_train_manifest import (
    _classify_row,
    _parse_target_states,
)


def test_parse_target_states_defaults_when_empty() -> None:
    states = _parse_target_states("")
    assert "RELAX" in states
    assert "TENSE_OPEN" in states


def test_classify_row_keeps_low_risk_warn() -> None:
    row = {"capture_mode": "event_onset"}
    detail = {
        "category": "suspicious",
        "quality_status": "warn",
        "selected_windows": 2,
        "dead_channels": [],
    }
    keep, reasons = _classify_row(
        row,
        detail,
        require_capture_mode="event_onset",
        min_selected_windows=2,
    )
    assert keep is True
    assert reasons == []


def test_classify_row_drops_retake_and_dead_channel() -> None:
    row = {"capture_mode": "event_onset"}
    detail = {
        "category": "retake",
        "quality_status": "warn",
        "selected_windows": 1,
        "dead_channels": [3],
    }
    keep, reasons = _classify_row(
        row,
        detail,
        require_capture_mode="event_onset",
        min_selected_windows=2,
    )
    assert keep is False
    assert "retake_category" in reasons
    assert "selected_windows<2" in reasons
    assert "dead_channels" in reasons


def test_classify_row_drops_when_missing_audit_detail() -> None:
    row = {"capture_mode": "event_onset"}
    keep, reasons = _classify_row(
        row,
        None,
        require_capture_mode="event_onset",
        min_selected_windows=2,
    )
    assert keep is False
    assert reasons == ["missing_audit_detail"]

