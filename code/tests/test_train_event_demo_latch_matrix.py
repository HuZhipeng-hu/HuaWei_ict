from __future__ import annotations

from pathlib import Path

from scripts.train_event_demo_latch_matrix import _rank_key, _write_summary_csv


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


def test_write_summary_csv_includes_control_fields(tmp_path: Path) -> None:
    out = tmp_path / "summary.csv"
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
            "checkpoint_path": "artifacts/runs/x/checkpoints/event_onset_best.ckpt",
            "summary_path": "artifacts/runs/x/offline_summary.json",
            "metrics_path": "artifacts/runs/x/evaluation/test_metrics.json",
        }
    ]
    _write_summary_csv(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "command_success_rate" in text
    assert "false_trigger_rate" in text
    assert "false_release_rate" in text
