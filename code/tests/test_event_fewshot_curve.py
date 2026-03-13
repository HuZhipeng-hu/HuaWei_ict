from __future__ import annotations

from scripts.evaluate_event_fewshot_curve import (
    FewshotRunResult,
    _aggregate_budget_rows,
    _choose_best_budget,
    _choose_elbow_budget,
)


def test_choose_best_budget_prefers_f1_then_acc_then_std() -> None:
    rows = [
        FewshotRunResult(10, 1, "r1", 0.30, 0.40, "a.ckpt", "", "cmd1"),
        FewshotRunResult(10, 2, "r2", 0.40, 0.30, "b.ckpt", "", "cmd2"),
        FewshotRunResult(20, 1, "r3", 0.36, 0.50, "c.ckpt", "", "cmd3"),
        FewshotRunResult(20, 2, "r4", 0.34, 0.50, "d.ckpt", "", "cmd4"),
    ]
    agg = _aggregate_budget_rows(rows)
    best = _choose_best_budget(agg)
    assert int(best["budget"]) == 20


def test_choose_elbow_budget_returns_smallest_budget_within_tolerance() -> None:
    agg = [
        {"budget": 10, "macro_f1_mean": 0.31, "acc_mean": 0.40, "macro_f1_std": 0.02},
        {"budget": 20, "macro_f1_mean": 0.35, "acc_mean": 0.44, "macro_f1_std": 0.02},
        {"budget": 35, "macro_f1_mean": 0.37, "acc_mean": 0.46, "macro_f1_std": 0.03},
        {"budget": 60, "macro_f1_mean": 0.38, "acc_mean": 0.47, "macro_f1_std": 0.03},
    ]
    assert _choose_elbow_budget(agg, tolerance=0.02) == 35
