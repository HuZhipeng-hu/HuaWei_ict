from __future__ import annotations

from scripts.evaluate_event_fewshot_curve import (
    FewshotRunResult,
    _append_recordings_manifest_arg,
    _aggregate_budget_rows,
    _choose_best_budget,
    _choose_elbow_budget,
    _resolve_recordings_manifest_path_for_fewshot,
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


def test_append_recordings_manifest_arg_only_when_path_provided() -> None:
    base = ["python", "scripts/finetune_event_onset.py", "--data_dir", "../data"]
    cmd_with = _append_recordings_manifest_arg(base, "../data/recordings_manifest.csv")
    assert cmd_with[-2:] == ["--recordings_manifest", "../data/recordings_manifest.csv"]

    cmd_without = _append_recordings_manifest_arg(base, None)
    assert cmd_without == base


def test_resolve_recordings_manifest_for_fewshot_prefers_explicit_path(tmp_path) -> None:
    manifest = tmp_path / "explicit_manifest.csv"
    manifest.write_text("relative_path\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    resolved = _resolve_recordings_manifest_path_for_fewshot(
        data_dir=str(tmp_path),
        config_path=str(config),
        manifest_arg=str(manifest),
    )
    assert resolved.endswith("explicit_manifest.csv")


def test_resolve_recordings_manifest_for_fewshot_uses_config_default(tmp_path) -> None:
    manifest = tmp_path / "recordings_manifest.csv"
    manifest.write_text("relative_path\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    resolved = _resolve_recordings_manifest_path_for_fewshot(
        data_dir=str(tmp_path),
        config_path=str(config),
        manifest_arg=None,
    )
    assert resolved.endswith("recordings_manifest.csv")


def test_resolve_recordings_manifest_for_fewshot_missing_raises_clear_error(tmp_path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    try:
        _resolve_recordings_manifest_path_for_fewshot(
            data_dir=str(tmp_path),
            config_path=str(config),
            manifest_arg=None,
        )
    except FileNotFoundError as exc:
        assert "Few-shot evaluation requires recordings manifest" in str(exc)
        return
    raise AssertionError("expected FileNotFoundError")
