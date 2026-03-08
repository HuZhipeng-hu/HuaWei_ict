"""
Inference scheduling helpers.
"""

from __future__ import annotations

import time
from typing import Optional


class InferenceRateScheduler:
    """Simple frequency limiter for inference calls."""

    def __init__(self, infer_rate_hz: float = 0.0):
        self._last_infer_ts: Optional[float] = None
        self.set_rate_hz(infer_rate_hz)

    def set_rate_hz(self, infer_rate_hz: float) -> None:
        self.infer_rate_hz = float(max(0.0, infer_rate_hz))
        self._interval_sec = 0.0 if self.infer_rate_hz <= 0 else 1.0 / self.infer_rate_hz

    @property
    def interval_sec(self) -> float:
        return self._interval_sec

    def reset(self) -> None:
        self._last_infer_ts = None

    def should_run(self, now: Optional[float] = None) -> bool:
        """
        Return True if inference should run now according to configured rate.
        """
        if self._interval_sec <= 0.0:
            return True

        ts = time.perf_counter() if now is None else float(now)
        if self._last_infer_ts is None or (ts - self._last_infer_ts) >= self._interval_sec:
            self._last_infer_ts = ts
            return True
        return False


class PredictionScheduler(InferenceRateScheduler):
    """Compatibility wrapper that accepts interval in milliseconds."""

    def __init__(self, inference_interval_ms: int = 0):
        infer_rate_hz = 0.0 if inference_interval_ms <= 0 else 1000.0 / float(inference_interval_ms)
        super().__init__(infer_rate_hz=infer_rate_hz)
