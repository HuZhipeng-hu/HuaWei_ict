from __future__ import annotations

from scripts.check_s2_collection_progress import (
    _latest_slice_report_by_state,
    _parse_target_states,
)


def test_parse_target_states_defaults_when_empty() -> None:
    states = _parse_target_states("")
    assert "CONTINUE" in states
    assert "TENSE_OPEN" in states


def test_latest_slice_report_by_state_prefers_newer_file(tmp_path) -> None:
    state_dir = tmp_path / "V_SIGN"
    state_dir.mkdir(parents=True, exist_ok=True)
    older = state_dir / "20260101T000000__slice_report.json"
    newer = state_dir / "20260102T000000__slice_report.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    mapping = _latest_slice_report_by_state(tmp_path)
    assert "V_SIGN" in mapping
    assert mapping["V_SIGN"] == newer
