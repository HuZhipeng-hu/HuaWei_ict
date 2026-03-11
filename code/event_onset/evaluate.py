"""Evaluation helpers for event-onset checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from event_onset.config import EventModelConfig
from event_onset.model import build_event_model

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
    if target.upper() == "GPU":
        context.set_context(device_id=device_id)


def load_event_model_from_checkpoint(
    ckpt_path: str | Path,
    model_config: EventModelConfig,
):
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    model = build_event_model(model_config)
    params = load_checkpoint(str(ckpt_path))
    load_param_into_net(model, params)
    model.set_train(False)
    return model


def evaluate_event_model(
    model,
    emg_samples: np.ndarray,
    imu_samples: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    logits = model(Tensor(emg_samples, ms.float32), Tensor(imu_samples, ms.float32)).asnumpy()
    preds = np.argmax(logits, axis=1).astype(np.int32)
    return compute_classification_report(labels.astype(np.int32), preds, class_names=class_names)


def load_and_evaluate_event(
    ckpt_path: str | Path,
    emg_samples: np.ndarray,
    imu_samples: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    model_config: EventModelConfig,
    device_target: str = "CPU",
    device_id: int = 0,
) -> Dict[str, Any]:
    _set_device(target=device_target, device_id=device_id)
    model = load_event_model_from_checkpoint(ckpt_path=ckpt_path, model_config=model_config)
    logger.info("Loaded event checkpoint: %s", ckpt_path)
    return evaluate_event_model(model, emg_samples, imu_samples, labels, class_names)
