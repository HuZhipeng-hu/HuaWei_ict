"""Evaluation helpers for DB5 pretraining."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from experimental.ninapro_db5.config import DB5PretrainConfig
from experimental.ninapro_db5.model import build_db5_pretrain_model

try:
    import mindspore as ms
    from mindspore import Tensor, context, load_checkpoint, load_param_into_net
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore
    load_checkpoint = None  # type: ignore
    load_param_into_net = None  # type: ignore

from training.reporting import compute_classification_report

logger = logging.getLogger(__name__)


def _set_device(mode: str = "graph", target: str = "CPU", device_id: int = 0) -> None:
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    mode_map = {"graph": context.GRAPH_MODE, "pynative": context.PYNATIVE_MODE}
    context.set_context(mode=mode_map.get(mode, context.GRAPH_MODE))
    context.set_context(device_target=target)
    if target.upper() in {"GPU", "ASCEND"}:
        context.set_context(device_id=device_id)


def load_db5_model_from_checkpoint(ckpt_path: str | Path, config: DB5PretrainConfig, num_classes: int):
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    model = build_db5_pretrain_model(config, num_classes=num_classes)
    params = load_checkpoint(str(ckpt_path))
    load_param_into_net(model, params)
    model.set_train(False)
    return model


def evaluate_db5_model(model, samples: np.ndarray, labels: np.ndarray, class_names: Sequence[str]) -> Dict:
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    logits = model(Tensor(samples, ms.float32)).asnumpy()
    preds = np.argmax(logits, axis=1).astype(np.int32)
    return compute_classification_report(labels.astype(np.int32), preds, class_names)
