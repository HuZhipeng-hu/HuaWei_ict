"""Training/evaluation reporting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def per_class_metrics(cm: np.ndarray, class_names: Sequence[str]) -> List[Dict]:
    metrics = []
    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = int(cm[i, :].sum())
        metrics.append(
            {
                "class_id": i,
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return metrics


def _top_confusion_pairs(cm: np.ndarray, class_names: Sequence[str], top_k: int = 3) -> List[Dict]:
    pairs: List[Dict] = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            count = int(cm[i, j] + cm[j, i])
            if count <= 0:
                continue
            pairs.append(
                {
                    "pair": [class_names[i], class_names[j]],
                    "count": count,
                    "a_to_b": int(cm[i, j]),
                    "b_to_a": int(cm[j, i]),
                }
            )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
    top_k_confusions: int = 3,
) -> Dict:
    if class_names is None:
        if num_classes is None:
            raise ValueError("Either class_names or num_classes must be provided.")
        class_names = [f"class_{i}" for i in range(num_classes)]
    class_names = list(class_names)
    if num_classes is None:
        num_classes = len(class_names)
    if len(class_names) != num_classes:
        raise ValueError("num_classes must match len(class_names)")

    cm = confusion_matrix(y_true, y_pred, num_classes)
    per_class_rows = per_class_metrics(cm, class_names)
    per_class_map = {row["class_name"]: row for row in per_class_rows}
    acc = float((y_true == y_pred).mean())
    macro_p = float(np.mean([m["precision"] for m in per_class_rows])) if per_class_rows else 0.0
    macro_r = float(np.mean([m["recall"] for m in per_class_rows])) if per_class_rows else 0.0
    macro_f1 = float(np.mean([m["f1"] for m in per_class_rows])) if per_class_rows else 0.0

    relax_idx = next(
        (idx for idx, name in enumerate(class_names) if str(name).strip().upper() == "RELAX"),
        None,
    )
    if relax_idx is None:
        action_mask = np.ones_like(y_true, dtype=bool)
        action_indices = list(range(num_classes))
    else:
        action_mask = y_true != int(relax_idx)
        action_indices = [idx for idx in range(num_classes) if idx != int(relax_idx)]

    if int(np.sum(action_mask)) > 0 and action_indices:
        action_true = y_true[action_mask]
        action_pred = y_pred[action_mask]
        action_cm = confusion_matrix(action_true, action_pred, num_classes)
        action_rows = per_class_metrics(action_cm, class_names)
        action_rows = [row for row in action_rows if int(row["class_id"]) in action_indices]
        action_accuracy = float((action_true == action_pred).mean())
        action_macro_f1 = float(np.mean([row["f1"] for row in action_rows])) if action_rows else 0.0
        action_macro_recall = float(np.mean([row["recall"] for row in action_rows])) if action_rows else 0.0
    else:
        action_accuracy = 0.0
        action_macro_f1 = 0.0
        action_macro_recall = 0.0

    return {
        "num_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "event_action_num_samples": int(np.sum(action_mask)),
        "event_action_accuracy": action_accuracy,
        "event_action_macro_recall": action_macro_recall,
        "event_action_macro_f1": action_macro_f1,
        "per_class": per_class_map,
        "per_class_rows": per_class_rows,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": _top_confusion_pairs(cm, class_names, top_k=top_k_confusions),
    }


def save_classification_report(report: Dict, out_dir: str | Path, prefix: str = "test") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_json = out / f"{prefix}_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    rows = report.get("per_class_rows")
    if rows is None:
        raw_per_class = report.get("per_class", {})
        rows = list(raw_per_class.values()) if isinstance(raw_per_class, dict) else list(raw_per_class)

    per_class_csv = out / f"{prefix}_per_class_metrics.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "class_name", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    confusion_csv = out / f"{prefix}_confusion_matrix.csv"
    with open(confusion_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        class_names = [row["class_name"] for row in rows]
        writer.writerow(["true\\pred"] + class_names)
        cm = np.asarray(report.get("confusion_matrix", []), dtype=int)
        for i, name in enumerate(class_names):
            writer.writerow([name] + cm[i].tolist())

    return {
        "metrics_json": str(metrics_json),
        "per_class_csv": str(per_class_csv),
        "confusion_csv": str(confusion_csv),
    }
