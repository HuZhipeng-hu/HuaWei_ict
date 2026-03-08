"""Checkpoint evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, context, load_checkpoint, load_param_into_net
except Exception:  # pragma: no cover
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore
    load_checkpoint = None  # type: ignore
    load_param_into_net = None  # type: ignore

from training.model import NeuroGripNet
from training.reporting import compute_classification_report

logger = logging.getLogger(__name__)


def _set_device(mode: str = "graph", target: str = "CPU", device_id: int = 0) -> None:
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    mode_map = {"graph": context.GRAPH_MODE, "pynative": context.PYNATIVE_MODE}
    context.set_context(mode=mode_map.get(mode, context.GRAPH_MODE))
    context.set_context(device_target=target)
    if target.upper() == "GPU":
        context.set_context(device_id=device_id)


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    in_channels: int,
    num_classes: int,
    dropout_rate: float,
    hidden_dim: int = 64,
    num_layers: int = 2,
) -> NeuroGripNet:
    if ms is None:
        raise RuntimeError("MindSpore is not available")

    model = NeuroGripNet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    params = load_checkpoint(str(ckpt_path))
    load_param_into_net(model, params)
    model.set_train(False)
    return model


def evaluate_model(
    model: NeuroGripNet,
    samples: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> Dict:
    if ms is None:
        raise RuntimeError("MindSpore is not available")

    logits = model(Tensor(samples, ms.float32)).asnumpy()
    preds = np.argmax(logits, axis=1).astype(np.int32)
    return compute_classification_report(labels.astype(np.int32), preds, class_names=class_names)


def load_and_evaluate(
    ckpt_path: str | Path,
    samples: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    in_channels: int,
    num_classes: int,
    dropout_rate: float,
    hidden_dim: int = 64,
    num_layers: int = 2,
    device_target: str = "CPU",
    device_id: int = 0,
) -> Dict:
    _set_device(target=device_target, device_id=device_id)
    model = load_model_from_checkpoint(
        ckpt_path=ckpt_path,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    logger.info("Loaded checkpoint: %s", ckpt_path)
    report = evaluate_model(model, samples, labels, class_names)
    _print_report(report)
    return report


def _print_report(report: Dict) -> None:
    rows = report.get("per_class_rows", list(report.get("per_class", {}).values()))
    logger.info("\n================================================================")
    logger.info("Evaluation result (%d samples)", int(report.get("num_samples", 0)))
    logger.info("================================================================")
    logger.info("Accuracy: %.4f", report["accuracy"])
    logger.info(
        "Macro P/R/F1: %.4f / %.4f / %.4f",
        report["macro_precision"],
        report["macro_recall"],
        report["macro_f1"],
    )

    logger.info("\nPer-class recall:")
    for row in rows:
        logger.info("  %-10s %.4f", row["class_name"], row["recall"])

    if report.get("top_confusion_pairs"):
        logger.info("\nTop confusion pairs:")
        for pair in report["top_confusion_pairs"]:
            a, b = pair["pair"]
            logger.info(
                "  %-10s <-> %-10s total=%d (%d/%d)",
                a,
                b,
                pair["count"],
                pair["a_to_b"],
                pair["b_to_a"],
            )

    cm = np.asarray(report["confusion_matrix"], dtype=int)
    names = [x["class_name"] for x in rows]
    logger.info("\nConfusion matrix:")
    header = "true\\pred  " + " ".join([f"{name[:6]:>6}" for name in names])
    logger.info(header)
    for i, name in enumerate(names):
        row = " ".join([f"{int(v):>6d}" for v in cm[i]])
        logger.info("%10s %s", name[:10], row)
