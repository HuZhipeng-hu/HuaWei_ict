"""
Model evaluation utilities.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

try:
    import mindspore.ops as ops
    from mindspore import Tensor, load_checkpoint, load_param_into_net

    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

from shared.gestures import GestureType
from training.reporting import compute_classification_report

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> Dict:
    """Evaluate model and return detailed metrics."""
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required for evaluation")

    model.set_train(False)

    all_preds = []
    all_labels = []

    num_samples = len(samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_data = Tensor(samples[start:end].astype(np.float32))
        batch_labels = labels[start:end]

        logits = model(batch_data)
        preds = ops.Argmax(axis=1)(logits).asnumpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())

    all_preds_np = np.asarray(all_preds, dtype=np.int32)
    all_labels_np = np.asarray(all_labels, dtype=np.int32)

    class_names = [g.name for g in GestureType]
    report = compute_classification_report(
        y_true=all_labels_np,
        y_pred=all_preds_np,
        num_classes=len(class_names),
        class_names=class_names,
    )

    results = {
        "accuracy": report["accuracy"],
        "per_class_accuracy": {
            class_name: report["per_class"][class_name]["recall"]
            for class_name in class_names
        },
        "confusion_matrix": np.asarray(report["confusion_matrix"], dtype=np.int64),
        "num_samples": report["num_samples"],
        "macro_precision": report["macro_precision"],
        "macro_recall": report["macro_recall"],
        "macro_f1": report["macro_f1"],
        "report": report,
        "predictions": all_preds,
        "labels": all_labels,
    }

    _print_results(results)
    return results


def _print_results(results: Dict) -> None:
    logger.info("\n%s", "=" * 64)
    logger.info("Evaluation result (%s samples)", results["num_samples"])
    logger.info("%s", "=" * 64)
    logger.info("Accuracy: %.4f", results["accuracy"])
    logger.info(
        "Macro P/R/F1: %.4f / %.4f / %.4f",
        results["macro_precision"],
        results["macro_recall"],
        results["macro_f1"],
    )

    logger.info("\nPer-class recall:")
    for name, recall in results["per_class_accuracy"].items():
        logger.info("  %-10s %.4f", name, recall)

    logger.info("\nConfusion matrix:")
    class_names = list(results["per_class_accuracy"].keys())
    header = "true\\pred " + " ".join([f"{n[:6]:>6}" for n in class_names])
    logger.info(header)
    confusion = results["confusion_matrix"]
    for idx, class_name in enumerate(class_names):
        row_str = " ".join([f"{int(v):>6}" for v in confusion[idx]])
        logger.info("%8s %s", class_name[:6], row_str)


def load_and_evaluate(
    model,
    checkpoint_path: str,
    samples: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Load checkpoint and evaluate."""
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required for evaluation")

    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(model, param_dict)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    return evaluate_model(model, samples, labels)
