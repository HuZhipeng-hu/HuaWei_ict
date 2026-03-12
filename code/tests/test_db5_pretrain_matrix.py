from __future__ import annotations

from scripts.pretrain_db5_experiment_matrix import _rank_key
from scripts.pretrain_ninapro_db5 import _best_validation_from_history


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
