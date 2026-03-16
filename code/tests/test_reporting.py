"""Unit tests for evaluation reporting metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.reporting import compute_classification_report


def test_classification_report_metrics_and_confusion_matrix():
    # 3-class toy example
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    y_pred = np.array([0, 1, 1, 1, 0, 2], dtype=np.int32)

    report = compute_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=3,
        class_names=["c0", "c1", "c2"],
    )

    expected_confusion = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(np.asarray(report["confusion_matrix"]), expected_confusion)

    assert report["num_samples"] == 6
    assert abs(report["accuracy"] - (4 / 6)) < 1e-9

    # Class-level metrics
    assert abs(report["per_class"]["c0"]["precision"] - 0.5) < 1e-9
    assert abs(report["per_class"]["c0"]["recall"] - 0.5) < 1e-9
    assert abs(report["per_class"]["c0"]["f1"] - 0.5) < 1e-9

    assert abs(report["per_class"]["c1"]["precision"] - (2 / 3)) < 1e-9
    assert abs(report["per_class"]["c1"]["recall"] - 1.0) < 1e-9
    assert abs(report["per_class"]["c1"]["f1"] - 0.8) < 1e-9

    assert abs(report["per_class"]["c2"]["precision"] - 1.0) < 1e-9
    assert abs(report["per_class"]["c2"]["recall"] - 0.5) < 1e-9
    assert abs(report["per_class"]["c2"]["f1"] - (2 / 3)) < 1e-9

    expected_macro_precision = (0.5 + (2 / 3) + 1.0) / 3
    expected_macro_recall = (0.5 + 1.0 + 0.5) / 3
    expected_macro_f1 = (0.5 + 0.8 + (2 / 3)) / 3

    assert abs(report["macro_precision"] - expected_macro_precision) < 1e-9
    assert abs(report["macro_recall"] - expected_macro_recall) < 1e-9
    assert abs(report["macro_f1"] - expected_macro_f1) < 1e-9

    # event-action metrics should exclude RELAX when present
    assert "event_action_accuracy" in report
    assert "event_action_macro_f1" in report
