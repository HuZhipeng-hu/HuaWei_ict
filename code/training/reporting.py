"""
Evaluation reporting layer.

Computes and exports reproducible classification metrics independent from training
loop internals.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from shared.gestures import GestureType, NUM_CLASSES


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= int(true_label) < num_classes and 0 <= int(pred_label) < num_classes:
            matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _default_class_names(num_classes: int) -> List[str]:
    if num_classes == NUM_CLASSES:
        return [g.name for g in GestureType]
    return [f"class_{idx}" for idx in range(num_classes)]


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
) -> Dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if num_classes is None:
        max_true = int(np.max(y_true)) if y_true.size > 0 else 0
        max_pred = int(np.max(y_pred)) if y_pred.size > 0 else 0
        num_classes = max(max_true, max_pred) + 1

    if class_names is None:
        class_names = _default_class_names(num_classes)
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes")

    confusion = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    num_samples = int(y_true.size)
    accuracy = _safe_divide(float(np.sum(np.diag(confusion))), float(num_samples))

    per_class: Dict[str, Dict[str, float | int]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for class_idx, class_name in enumerate(class_names):
        tp = float(confusion[class_idx, class_idx])
        fp = float(np.sum(confusion[:, class_idx]) - tp)
        fn = float(np.sum(confusion[class_idx, :]) - tp)
        support = int(np.sum(confusion[class_idx, :]))

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    support_total = float(np.sum(supports)) if supports else 0.0
    weighted_precision = _safe_divide(float(np.dot(precisions, supports)), support_total)
    weighted_recall = _safe_divide(float(np.dot(recalls, supports)), support_total)
    weighted_f1 = _safe_divide(float(np.dot(f1s, supports)), support_total)

    report = {
        "num_samples": num_samples,
        "num_classes": int(num_classes),
        "class_names": list(class_names),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    return report


def save_classification_report(report: Dict, output_dir: str | Path, prefix: str = "test") -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = out_dir / f"{prefix}_metrics.json"
    per_class_csv = out_dir / f"{prefix}_per_class_metrics.csv"
    confusion_csv = out_dir / f"{prefix}_confusion_matrix.csv"

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for class_name in report["class_names"]:
            metrics = report["per_class"][class_name]
            writer.writerow(
                [
                    class_name,
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                    int(metrics["support"]),
                ]
            )

    confusion = np.asarray(report["confusion_matrix"], dtype=np.int64)
    with open(confusion_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *report["class_names"]])
        for row_idx, class_name in enumerate(report["class_names"]):
            writer.writerow([class_name, *confusion[row_idx].tolist()])

    return {
        "metrics_json": str(metrics_json),
        "per_class_csv": str(per_class_csv),
        "confusion_csv": str(confusion_csv),
    }
