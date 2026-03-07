"""
MindSpore Lite inference engine wrapper.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mindspore_lite as mslite

    MSLITE_AVAILABLE = True
except ImportError:
    MSLITE_AVAILABLE = False


class InferenceEngine:
    _DEVICE_ALIASES = {"CPU": "CPU", "GPU": "GPU", "ASCEND": "ASCEND", "NPU": "ASCEND"}
    _TARGET_MAP = {"CPU": "cpu", "GPU": "gpu", "ASCEND": "ascend"}

    @classmethod
    def _normalize_device(cls, device: str) -> str:
        key = str(device).strip().upper()
        normalized = cls._DEVICE_ALIASES.get(key)
        if normalized is None:
            supported = ", ".join(sorted(cls._DEVICE_ALIASES))
            raise ValueError(f"Unsupported inference device: {device!r}. Expected one of: {supported}.")
        return normalized

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        num_threads: int = 4,
        latency_window: int = 100,
    ):
        self.model_path = model_path
        self.device = self._normalize_device(device)
        self._latencies = deque(maxlen=latency_window)
        self._inference_count = 0
        self._input_shape: Optional[Tuple[int, ...]] = None

        if not MSLITE_AVAILABLE:
            logger.warning("mindspore_lite not installed, inference engine running in mock mode.")
            self._model = None
            self._mock_mode = True
            return

        self._mock_mode = False
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        context = mslite.Context()
        context.target = [self._TARGET_MAP[self.device]]
        if self.device == "CPU":
            context.cpu.thread_num = int(num_threads)

        self._model = mslite.Model()
        try:
            self._model.build_from_file(str(model_file), mslite.ModelType.MINDIR, context)
        except Exception as exc:
            raise RuntimeError(
                f"build_from_file failed for model={model_file}, device={self.device}. "
                "Please confirm model/runtime compatibility and backend support."
            ) from exc

        inputs = self._model.get_inputs()
        if inputs:
            self._input_shape = tuple(int(v) for v in inputs[0].shape)

        logger.info("Inference engine ready: model=%s, device=%s", model_path, self.device)

    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        return self._input_shape

    def predict(self, spectrogram: np.ndarray) -> Tuple[int, float]:
        if spectrogram.ndim == 3:
            spectrogram = spectrogram[np.newaxis, ...]
        spectrogram = spectrogram.astype(np.float32)

        start = time.perf_counter()
        if self._mock_mode:
            probs = self._mock_predict()
        else:
            probs = self._real_predict(spectrogram)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._latencies.append(elapsed_ms)
        self._inference_count += 1

        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])
        return gesture_id, confidence

    def _real_predict(self, input_data: np.ndarray) -> np.ndarray:
        inputs = self._model.get_inputs()
        inputs[0].set_data_from_numpy(input_data)
        outputs = self._model.predict(inputs)
        logits = outputs[0].get_data_to_numpy()
        return self._softmax(logits[0])

    @staticmethod
    def _mock_predict() -> np.ndarray:
        from shared.gestures import NUM_CLASSES

        time.sleep(0.015)
        logits = np.random.randn(NUM_CLASSES).astype(np.float32)
        return InferenceEngine._softmax(logits)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def get_latency_stats(self) -> Dict[str, float]:
        if not self._latencies:
            return {"count": 0}
        latencies = np.array(self._latencies)
        return {
            "count": self._inference_count,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
