from __future__ import annotations

from dataclasses import dataclass

from scripts.pretrain_db5_experiment_matrix import _rank_key, _should_skip_run4
from scripts.pretrain_ninapro_db5 import (
    _best_validation_from_history,
    _build_referee_card_content,
    _build_split_diagnostics,
)


def test_best_validation_from_history_prefers_f1_then_acc() -> None:
    history = {
        "epoch": [1, 2, 3, 4],
        "val_macro_f1": [0.10, 0.12, 0.12, 0.11],
        "val_acc": [0.20, 0.19, 0.21, 0.25],
    }
    best_epoch, best_acc, best_f1 = _best_validation_from_history(history)
    assert best_epoch == 3
    assert best_f1 == 0.12
    assert best_acc == 0.21


def test_matrix_rank_key_uses_validation_metrics() -> None:
    summary = {
        "best_val_macro_f1": 0.031,
        "best_val_acc": 0.044,
        "test_macro_f1": 0.099,  # Should not affect ranking.
        "test_accuracy": 0.099,
    }
    assert _rank_key(summary) == (0.031, 0.044)


def test_should_skip_run4_when_run3_gain_below_threshold() -> None:
    assert _should_skip_run4(run0_val_f1=0.083, run3_val_f1=0.109) is True
    assert _should_skip_run4(run0_val_f1=0.083, run3_val_f1=0.113) is False


@dataclass
class _ManifestStub:
    class_distribution: dict


def test_build_split_diagnostics_reports_empty_classes() -> None:
    manifest = _ManifestStub(
        class_distribution={
            "train": {"A": 10, "B": 5},
            "val": {"A": 3, "B": 0},
            "test": {"A": 2, "B": 1},
        }
    )
    diag = _build_split_diagnostics(manifest, ["A", "B"])
    assert diag["by_split"]["val"]["has_empty_classes"] is True
    assert diag["by_split"]["val"]["empty_classes"] == ["B"]
    assert diag["overall"]["has_any_empty_classes"] is True


def test_build_referee_card_content_contains_repro_statement() -> None:
    summary = {
        "run_id": "db5_sprint_v1_run_3",
        "checkpoint_path": "artifacts/runs/db5_sprint_v1_run_3/checkpoints/db5_pretrain_best.ckpt",
        "best_val_epoch": 27,
        "best_val_macro_f1": 0.111,
        "best_val_acc": 0.128,
        "top_confusion_pair": "E1_G01<->E1_G02:42",
        "split_seed": 42,
    }
    content = _build_referee_card_content(
        summary=summary,
        data_dir="../data_ninaproDB5",
        command="python scripts/pretrain_ninapro_db5.py --config configs/pretrain_ninapro_db5.yaml",
    )
    assert "无需个人校准数据" in content
    assert "best_val_macro_f1" in content
    assert "db5_pretrain_best.ckpt" in content
