"""Inference post-processing helpers."""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import Optional

from shared.gestures import GestureType, NUM_CLASSES

logger = logging.getLogger(__name__)


class SlidingWindowVoter:
    """Majority vote smoother kept for backward compatibility."""

    def __init__(
        self,
        window_size: int = 5,
        min_count: int = 3,
        confidence_threshold: float = 0.5,
    ):
        if min_count > window_size:
            raise ValueError(f"min_count ({min_count}) cannot exceed window_size ({window_size})")
        self.window_size = int(window_size)
        self.min_count = int(min_count)
        self.confidence_threshold = float(confidence_threshold)
        self._window: deque[int] = deque(maxlen=window_size)
        self._last_output: Optional[GestureType] = None

    def update(self, gesture_id: int, confidence: float) -> Optional[GestureType]:
        if confidence < self.confidence_threshold:
            return self._last_output
        self._window.append(int(gesture_id))
        if len(self._window) < self.window_size:
            if len(set(self._window)) == 1:
                self._last_output = GestureType(int(gesture_id))
            return self._last_output
        counter = Counter(self._window)
        most_common_id, most_common_count = counter.most_common(1)[0]
        if most_common_count >= self.min_count:
            self._last_output = GestureType(int(most_common_id))
        return self._last_output

    def reset(self) -> None:
        self._window.clear()
        self._last_output = None

    @property
    def current_gesture(self) -> Optional[GestureType]:
        return self._last_output

    @property
    def window_state(self) -> list[int]:
        return list(self._window)


class TemporalVoter:
    """Time-window voter used by the new runtime controller."""

    def __init__(
        self,
        history_window_ms: int = 400,
        hysteresis_count: int = 3,
        num_classes: int = NUM_CLASSES,
    ):
        del num_classes
        self.history_window_sec = max(0.0, float(history_window_ms) / 1000.0)
        self.hysteresis_count = max(1, int(hysteresis_count))
        self._window: deque[tuple[float, int]] = deque()
        self._last_output: Optional[int] = None

    def update(self, gesture_id: int, confidence: float, now: float) -> Optional[int]:
        del confidence
        self._window.append((float(now), int(gesture_id)))
        cutoff = float(now) - self.history_window_sec
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        if not self._window:
            return self._last_output
        counts = Counter(label for _, label in self._window)
        most_common_id, most_common_count = counts.most_common(1)[0]
        if most_common_count >= self.hysteresis_count:
            self._last_output = int(most_common_id)
        return self._last_output
