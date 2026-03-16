from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from scripts.train_event_demo_latch_matrix import (
    _build_invalid_reason_counts,
    _compute_metric_invariant_ok,
    _rank_key,
    _write_summary_csv,
)


def test_rank_key_prefers_safety_then_event_then_control() -> None:
    unsafe_high = {
        "safety_ok": False,
        "event_action_accuracy": 0.95,
        "event_action_macro_f1": 0.90,
        "command_success_rate": 0.95,
        "false_trigger_rate": 0.00,
        "false_release_rate": 0.00,
        "test_accuracy": 0.95,
        "test_macro_f1": 0.95,
    }
    safe_lower = {
        "safety_ok": True,
        "event_action_accuracy": 0.80,
        "event_action_macro_f1": 0.75,
        "command_success_rate": 0.82,
        "false_trigger_rate": 0.05,
        "false_release_rate": 0.04,
        "test_accuracy": 0.70,
        "test_macro_f1": 0.68,
    }
    ranked = sorted([unsafe_high, safe_lower], key=_rank_key, reverse=True)
    assert ranked[0] is safe_lower


def test_write_summary_csv_includes_control_fields() -> None:
    tmp_root = Path(__file__).resolve().parent / ".tmp_testdata" / f"csv_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    out = tmp_root / "summary.csv"
    rows = [
        {
            "run_id": "exp_1",
            "config_tag": "demo_p0",
            "config_path": "configs/training_event_onset_demo_p0.yaml",
            "split_seed": 42,
            "event_action_accuracy": 0.7,
            "event_action_macro_f1": 0.6,
            "test_accuracy": 0.65,
            "test_macro_f1": 0.61,
            "command_success_rate": 0.78,
            "false_trigger_rate": 0.10,
            "false_release_rate": 0.06,
            "best_val_acc": 0.73,
            "best_val_f1": 0.70,
            "relax_action_confusion": 6,
            "safety_ok": True,
            "control_eval_present": True,
            "control_eval_source": "control_eval_summary",
            "metric_invariant_ok": True,
            "control_metric_drift": False,
            "warning_reasons": "",
            "checkpoint_path": "artifacts/runs/x/checkpoints/event_onset_best.ckpt",
            "summary_path": "artifacts/runs/x/offline_summary.json",
            "metrics_path": "artifacts/runs/x/evaluation/test_metrics.json",
        }
    ]
    try:
        _write_summary_csv(out, rows)
        text = out.read_text(encoding="utf-8")
        assert "command_success_rate" in text
        assert "false_trigger_rate" in text
        assert "false_release_rate" in text
        assert "control_eval_present" in text
        assert "warning_reasons" in text
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


def test_compute_metric_invariant_ok() -> None:
    assert _compute_metric_invariant_ok(command_success_rate=0.0, false_trigger_rate=1.0) is True
    assert _compute_metric_invariant_ok(command_success_rate=0.06, false_trigger_rate=0.96) is False


def test_build_invalid_reason_counts() -> None:
    rows = [
        {"status": "invalid", "warning_reasons": "control_eval_invalid;metric_invariant_failed"},
        {"status": "invalid", "warning_reasons": "control_eval_invalid"},
        {"status": "ok", "warning_reasons": "control_eval_invalid"},
    ]
    counts = _build_invalid_reason_counts(rows)
    assert counts["control_eval_invalid"] == 2
    assert counts["metric_invariant_failed"] == 1
