"""Runtime control loop."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from runtime.hardware.factory import create_actuator, create_sensor
from runtime.inference import InferenceEngine, PredictionScheduler, TemporalVoter
from shared.config import RuntimeConfig
from shared.gestures import GestureType
from shared.preprocessing import PreprocessPipeline

logger = logging.getLogger(__name__)


class RuntimeController:
    def __init__(
        self,
        config: RuntimeConfig,
        sensor=None,
        actuator=None,
        preprocess: Optional[PreprocessPipeline] = None,
        engine: Optional[InferenceEngine] = None,
    ):
        self.config = config
        self.preprocess = preprocess or PreprocessPipeline(config.preprocess)
        expected_shape = (1,) + tuple(self.preprocess.get_output_shape())
        self.engine = engine or InferenceEngine(
            model_path=config.model_path,
            use_lite=config.inference.use_lite,
            device_target=config.device.target,
            device_id=config.device.id,
            expected_input_shape=expected_shape,
        )
        self.sensor = sensor or create_sensor(config.hardware)
        self.actuator = actuator or create_actuator(config.hardware)

        self._stop = False
        self._recent_predictions: Deque[tuple[float, int, float]] = deque(maxlen=100)
        infer_rate_hz = float(config.inference.infer_rate_hz)
        interval_ms = 0 if infer_rate_hz <= 0 else int(round(1000.0 / infer_rate_hz))
        self.scheduler = PredictionScheduler(inference_interval_ms=interval_ms)
        self.voter = TemporalVoter(
            history_window_ms=self.config.inference.smoothing_window_ms,
            hysteresis_count=self.config.inference.hysteresis_count,
        )

        self._tta_offsets = list(self.config.inference.tta_offsets or [0.0])
        self._num_channels = int(self.config.preprocess.num_channels)
        self._base_window_size = self.preprocess.get_required_window_size()
        self._stride = self.preprocess.get_required_window_stride()
        self._read_window_size = self._calc_read_window_size(self._base_window_size, self._stride, self._tta_offsets)

        self.engine.load()
        self._validate_model_shape()

    @staticmethod
    def _calc_read_window_size(base_window: int, stride: int, offsets: List[float]) -> int:
        if not offsets:
            return base_window
        max_offset = max(0.0, max(offsets))
        return int(base_window + round(max_offset * stride))

    @staticmethod
    def _slice_tta_windows(window: np.ndarray, base_window: int, stride: int, offsets: List[float]) -> List[np.ndarray]:
        if not offsets:
            offsets = [0.0]
        slices: List[np.ndarray] = []
        for offset in offsets:
            start = int(round(max(0.0, offset) * stride))
            end = start + base_window
            if end > window.shape[0]:
                continue
            slices.append(window[start:end])
        if not slices and window.shape[0] >= base_window:
            slices.append(window[-base_window:])
        return slices

    def _validate_model_shape(self) -> None:
        dummy = np.zeros((self._base_window_size, self._num_channels), dtype=np.float32)
        expected_feature_shape = self.preprocess.process_window(dummy).shape
        expected_input_shape = (1,) + tuple(expected_feature_shape)
        model_shape = self.engine.get_input_shape()
        if model_shape is not None and tuple(model_shape) != expected_input_shape:
            raise ValueError(
                f"Runtime shape mismatch: preprocess expects {expected_input_shape}, "
                f"but model expects {tuple(model_shape)}. Please re-convert model."
            )
        logger.info("Runtime shape check passed: expected input=%s", expected_input_shape)

    def _ensure_connected(self) -> None:
        if hasattr(self.sensor, "is_connected") and not self.sensor.is_connected():
            if not self.sensor.connect():
                raise RuntimeError("Failed to connect sensor")
        if hasattr(self.actuator, "is_connected") and not self.actuator.is_connected():
            if not self.actuator.connect():
                raise RuntimeError("Failed to connect actuator")

    def start(self, max_cycles: Optional[int] = None) -> None:
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be > 0 when specified.")
        self._ensure_connected()
        logger.info("Runtime started. Press Ctrl+C to stop.")
        cycle_interval = 0.0 if self.config.control_rate_hz <= 0 else 1.0 / float(self.config.control_rate_hz)
        cycles = 0
        try:
            while not self._stop and (max_cycles is None or cycles < max_cycles):
                started = time.perf_counter()
                raw_window = self.sensor.read_window(window_size=self._read_window_size)
                if raw_window is not None:
                    self._control_step(raw_window)
                cycles += 1
                if cycle_interval > 0:
                    elapsed = time.perf_counter() - started
                    sleep_time = cycle_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt. Stopping runtime.")
        finally:
            self.shutdown()

    def run(self) -> None:
        self.start(max_cycles=None)

    def _control_step(self, raw_window: np.ndarray) -> None:
        now = time.time()
        if not self.scheduler.should_run(now):
            return

        slices = self._slice_tta_windows(raw_window, self._base_window_size, self._stride, self._tta_offsets)
        if not slices:
            return

        probs: List[np.ndarray] = []
        for segment in slices:
            feature = self.preprocess.process_window(segment)
            probs.append(self.engine.predict_proba(feature))
        mean_prob = np.mean(np.stack(probs, axis=0), axis=0)

        gesture_id = int(np.argmax(mean_prob))
        confidence = float(np.max(mean_prob))
        stable_gesture = self.voter.update(gesture_id, confidence, now)

        self._recent_predictions.append((now, gesture_id, confidence))
        if stable_gesture is not None and confidence >= self.config.inference.confidence_threshold:
            self.actuator.execute_gesture(GestureType(int(stable_gesture)))
            logger.info("Stable gesture=%s, confidence=%.3f", GestureType(int(stable_gesture)).name, confidence)

    def shutdown(self) -> None:
        self._stop = True
        try:
            self.actuator.disconnect()
        except Exception:
            pass
        try:
            self.sensor.disconnect()
        except Exception:
            pass
        logger.info("Runtime stopped.")


class ProsthesisController(RuntimeController):
    pass
