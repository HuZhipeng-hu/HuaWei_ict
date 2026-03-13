from __future__ import annotations

from scripts.pretrain_db5_repr_method_matrix import (
    _compute_recommended_budget,
    _rank_key,
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

