from pathlib import Path

from scripts.collect_event_data import (
    _build_default_relative_path,
    _estimate_activation_quality,
    _frame_to_rows,
    _normalize_state,
    _resolve_output_paths,
    _resolve_duration_sec,
    _sanitize_token,
)


def test_frame_to_rows_converts_single_frame_with_two_emg_packs():
    parsed = {
        "acc": (1, 2, 3),
        "gyro": (4, 5, 6),
        "angle": (7, 8, 9),
        "emg": [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ],
    }
    rows = _frame_to_rows(parsed)
    assert len(rows) == 2
    assert len(rows[0]) == 17
    assert rows[0][:8] == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    assert rows[0][8:] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_build_default_relative_path_uses_target_state_directory():
    rel = _build_default_relative_path(
        target_state="V_SIGN",
        user_id="user-a",
        session_id="s1",
        device_id="arm-01",
        wearing_state="normal",
        timestamp="20260314T120000",
    )
    assert rel.as_posix().startswith("V_SIGN/")
    assert rel.suffix == ".csv"
    assert "u-user-a" in rel.name


def test_resolve_output_paths_accepts_manual_relpath(tmp_path: Path):
    output_csv, rel = _resolve_output_paths(
        data_dir=tmp_path,
        output_relpath="WRIST_CW/custom_capture.csv",
        target_state="WRIST_CW",
        user_id="u",
        session_id="s",
        device_id="d",
        wearing_state="w",
        timestamp="20260314T120000",
    )
    assert rel == "WRIST_CW/custom_capture.csv"
    assert output_csv == (tmp_path / "WRIST_CW" / "custom_capture.csv").resolve()


def test_state_and_token_normalization():
    assert _normalize_state(" wrist_ccw ") == "WRIST_CCW"
    assert _normalize_state(" continue ") == "CONTINUE"
    assert _sanitize_token(" demo user ") == "demo-user"


def test_resolve_duration_sec_defaults_action_and_relax():
    assert _resolve_duration_sec("V_SIGN", None) == 3.0
    assert _resolve_duration_sec("RELAX", None) == 4.0
    assert _resolve_duration_sec("CONTINUE", None) == 4.0
    assert _resolve_duration_sec("V_SIGN", 2.5) == 2.5


def test_estimate_activation_quality_marks_weak_action_for_retake():
    baseline_row = [10.0] * 8 + [0.0] * 9
    action_row = [10.2] * 8 + [0.0] * 9
    rows = [baseline_row for _ in range(100)] + [action_row for _ in range(300)]
    report = _estimate_activation_quality(
        rows=rows,
        elapsed_sec=3.0,
        target_state="V_SIGN",
        pre_roll_ms=500,
        action_window_sec=1.2,
    )
    assert report["quality_status"] == "retake_recommended"
    assert "weak_activation" in report["quality_reasons"]


def test_estimate_activation_quality_marks_pass_for_clear_action():
    baseline_row = [10.0] * 8 + [0.0] * 9
    action_row = [18.0] * 8 + [0.0] * 9
    rows = [baseline_row for _ in range(100)] + [action_row for _ in range(300)]
    report = _estimate_activation_quality(
        rows=rows,
        elapsed_sec=3.0,
        target_state="THUMB_UP",
        pre_roll_ms=500,
        action_window_sec=1.2,
    )
    assert report["quality_status"] == "pass"
    assert report["activation_ratio"] is not None
    assert report["activation_ratio"] > 1.2
