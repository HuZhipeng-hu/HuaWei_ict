"""MindSpore inference engine with strict input-shape validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, context
    from mindspore_lite import Model, Context as LiteContext
except Exception:  # pragma: no cover
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore
    Model = None  # type: ignore
    LiteContext = None  # type: ignore

from shared.gestures import GESTURE_DEFINITIONS

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        use_lite: bool = True,
        device_target: str = "CPU",
        device_id: int = 0,
        expected_input_shape: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.model_path = Path(model_path)
        self.use_lite = use_lite
        self.device_target = device_target
        self.device_id = device_id
        self.expected_input_shape = expected_input_shape
        self.num_classes = len(GESTURE_DEFINITIONS)

        self._lite_model = None
        self._ms_graph = None
        self._input_shape: Optional[Tuple[int, ...]] = None

    def load(self) -> None:
        if not self.model_path.exists():
            logger.warning("Model file not found: %s, fallback to mock mode.", self.model_path)
            self._lite_model = None
            self._ms_graph = None
            return

        if self.use_lite and Model is not None:
            self._load_lite()
            return

        self._load_mindspore_graph()

    def _load_lite(self) -> None:
        assert Model is not None and LiteContext is not None
        ctx = LiteContext()
        model = Model()
        model.build_from_file(str(self.model_path), model_type=0, context=ctx)
        self._lite_model = model
        self._ms_graph = None
        try:
            input_tensor = self._lite_model.get_inputs()[0]
            self._input_shape = tuple(int(x) for x in input_tensor.shape)
        except Exception:
            self._input_shape = None
        logger.info("Loaded Lite model: %s", self.model_path)

    def _load_mindspore_graph(self) -> None:
        if ms is None:
            logger.warning("MindSpore unavailable, fallback to mock mode.")
            self._lite_model = None
            self._ms_graph = None
            return
        context.set_context(mode=context.GRAPH_MODE, device_target=self.device_target)
        if self.device_target.upper() == "GPU":
            context.set_context(device_id=self.device_id)
        graph = ms.load(str(self.model_path))
        self._ms_graph = graph
        self._lite_model = None
        self._input_shape = None
        logger.info("Loaded MindSpore graph: %s", self.model_path)

    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        if self._input_shape is not None:
            return tuple(self._input_shape)
        if self.expected_input_shape is not None:
            return tuple(self.expected_input_shape)
        return None

    def _validate_input(self, spectrogram: np.ndarray) -> np.ndarray:
        if spectrogram.ndim == 3:
            batched = np.expand_dims(spectrogram, axis=0)
        elif spectrogram.ndim == 4:
            batched = spectrogram
        else:
            raise ValueError(f"Invalid spectrogram ndim={spectrogram.ndim}, expect 3 or 4.")

        expected = self.get_input_shape()
        if expected is not None and tuple(batched.shape) != tuple(expected):
            raise ValueError(
                f"Model input shape mismatch: got {tuple(batched.shape)}, expected {tuple(expected)}. "
                "Please re-export model with matching preprocess/training settings."
            )
        return batched.astype(np.float32)

    def predict_proba(self, spectrogram: np.ndarray) -> np.ndarray:
        batched = self._validate_input(spectrogram)

        if self._lite_model is not None:
            input_tensor = self._lite_model.get_inputs()[0]
            input_tensor.set_data_from_numpy(batched)
            outputs = self._lite_model.predict([input_tensor])
            logits = outputs[0].get_data_to_numpy()
        elif self._ms_graph is not None and Tensor is not None:
            logits = self._ms_graph(Tensor(batched)).asnumpy()
        else:
            logits = np.random.randn(batched.shape[0], self.num_classes).astype(np.float32)

        logits = logits[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs.astype(np.float32)

    def predict(self, spectrogram: np.ndarray) -> Tuple[int, float]:
        probs = self.predict_proba(spectrogram)
        gesture_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return gesture_id, confidence
