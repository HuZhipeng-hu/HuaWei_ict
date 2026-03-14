from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pretrain_db5_repr_method_matrix import (
    _as_pretrain_row,
    _compute_recommended_budget,
    _parse_float_grid,
    _parse_int_grid,
    _pretrain_rank_key,
    _rank_key,
    _resolve_fewshot_plan,
    _update_plateau_streak,
)


def test_compute_recommended_budget_returns_smallest_budget_within_tolerance() -> None:
    rows = [
        {"budget": 10, "macro_f1_mean": 0.31, "acc_mean": 0.40, "macro_f1_std": 0.03},
        {"budget": 20, "macro_f1_mean": 0.36, "acc_mean": 0.44, "macro_f1_std": 0.02},
        {"budget": 35, "macro_f1_mean": 0.38, "acc_mean": 0.46, "macro_f1_std": 0.02},
        {"budget": 60, "macro_f1_mean": 0.39, "acc_mean": 0.47, "macro_f1_std": 0.03},
    ]
    budget, row = _compute_recommended_budget(rows, tolerance=0.02)
    assert budget == 35
    assert int(row["budget"]) == 35


def test_rank_key_prefers_lower_budget_then_higher_f1_acc_lower_std() -> None:
    row_a = {
        "recommended_budget": 20,
        "recommended_budget_row": {"macro_f1_mean": 0.34, "acc_mean": 0.40, "macro_f1_std": 0.03},
    }
    row_b = {
        "recommended_budget": 35,
        "recommended_budget_row": {"macro_f1_mean": 0.50, "acc_mean": 0.50, "macro_f1_std": 0.01},
    }
    row_c = {
        "recommended_budget": 20,
        "recommended_budget_row": {"macro_f1_mean": 0.36, "acc_mean": 0.39, "macro_f1_std": 0.02},
    }
    ranked = sorted([row_a, row_b, row_c], key=_rank_key)
    assert ranked[0] is row_c
    assert ranked[1] is row_a
    assert ranked[2] is row_b


def test_update_plateau_streak_increments_when_budget_not_improving() -> None:
    assert _update_plateau_streak(previous_budget=None, current_budget=35, current_streak=0) == 0
    assert _update_plateau_streak(previous_budget=35, current_budget=35, current_streak=0) == 1
    assert _update_plateau_streak(previous_budget=35, current_budget=60, current_streak=1) == 2
    assert _update_plateau_streak(previous_budget=35, current_budget=20, current_streak=2) == 0


def test_resolve_fewshot_plan_off_and_auto_without_manifest() -> None:
    enabled, status, reason = _resolve_fewshot_plan(mode="off", recordings_manifest=None)
    assert enabled is False
    assert status == "skipped"
    assert "fewshot_mode=off" in reason

    enabled, status, reason = _resolve_fewshot_plan(mode="auto", recordings_manifest=None)
    assert enabled is False
    assert status == "skipped"
    assert "not provided" in reason


def test_resolve_fewshot_plan_on_requires_existing_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires --recordings_manifest"):
        _resolve_fewshot_plan(mode="on", recordings_manifest=None)

    missing = tmp_path / "missing_manifest.csv"
    with pytest.raises(RuntimeError, match="requires existing recordings_manifest"):
        _resolve_fewshot_plan(mode="on", recordings_manifest=str(missing))

    ok = tmp_path / "recordings_manifest.csv"
    ok.write_text("relative_path\n", encoding="utf-8")
    enabled, status, reason = _resolve_fewshot_plan(mode="on", recordings_manifest=str(ok))
    assert enabled is True
    assert status == "enabled"
    assert reason == ""


def test_pretrain_rank_key_prefers_val_f1_then_val_acc_then_test_f1() -> None:
    rows = [
        {"pretrain_best_val_macro_f1": 0.10, "pretrain_best_val_acc": 0.20, "pretrain_test_macro_f1": 0.50},
        {"pretrain_best_val_macro_f1": 0.10, "pretrain_best_val_acc": 0.21, "pretrain_test_macro_f1": 0.40},
        {"pretrain_best_val_macro_f1": 0.11, "pretrain_best_val_acc": 0.19, "pretrain_test_macro_f1": 0.10},
    ]
    ranked = sorted(rows, key=_pretrain_rank_key)
    assert ranked[0]["pretrain_best_val_macro_f1"] == 0.11


def test_parse_float_grid_parses_comma_values() -> None:
    values = _parse_float_grid("0.05, 0.07,0.10", name="temperature_grid")
    assert values == [0.05, 0.07, 0.10]


def test_parse_int_grid_parses_comma_values() -> None:
    values = _parse_int_grid("3,5,11", name="knn_k_grid")
    assert values == [3, 5, 11]


def test_parse_float_grid_rejects_empty() -> None:
    with pytest.raises(ValueError, match="temperature_grid must contain at least one numeric value"):
        _parse_float_grid(" , ", name="temperature_grid")


def test_parse_int_grid_rejects_empty() -> None:
    with pytest.raises(ValueError, match="knn_k_grid must contain at least one integer value"):
        _parse_int_grid(" , ", name="knn_k_grid")


def test_as_pretrain_row_exports_backward_compatible_aliases() -> None:
    row = _as_pretrain_row(
        phase="run_2",
        candidate="temp_0p1",
        run_id="db5_r2_t_0p1",
        summary={
            "best_val_macro_f1": 0.12,
            "best_val_acc": 0.20,
            "test_macro_f1": 0.11,
            "checkpoint_path": "artifacts/runs/x/checkpoints/best.ckpt",
            "offline_summary_path": "artifacts/runs/x/offline_summary.json",
        },
        temperature=0.1,
        projection_dim=128,
        knn_k=5,
    )
    assert row["phase"] == "run_2"
    assert row["round"] == "run_2"
    assert row["candidate"] == "temp_0p1"
    assert row["name"] == "temp_0p1"
    assert row["run_id"] == "db5_r2_t_0p1"
    assert row["pretrain_run_id"] == "db5_r2_t_0p1"
    assert row["checkpoint_path"] == "artifacts/runs/x/checkpoints/best.ckpt"
    assert row["encoder_checkpoint_path"] == "artifacts/runs/x/checkpoints/best.ckpt"
