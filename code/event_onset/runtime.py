"""Runtime state machine and feature helpers for event-onset control."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np

from event_onset.config import EventDataConfig, EventInferenceConfig, EventRuntimeBehaviorConfig
from shared.preprocessing import PreprocessPipeline
from shared.gestures import GestureType


@dataclass
class RuntimeDecision:
    state: GestureType
    changed: bool
    emitted_label: int
    voted_label: int


@dataclass
class ControllerStep:
    sample_index: int
    now_ms: float
    energy: float
    predicted_label: int
    confidence: float
    decision: RuntimeDecision


class EventRuntimeStateMachine:
    """Event-onset runtime controller with vote, latching and idle release."""

    def __init__(
        self,
        inference_config: EventInferenceConfig,
        runtime_config: EventRuntimeBehaviorConfig,
        *,
        low_energy_threshold: float,
    ):
        self.inference_config = inference_config
        self.runtime_config = runtime_config
        self.low_energy_threshold = float(low_energy_threshold)
        self.current_state = GestureType.RELAX
        self._vote_window: Deque[int] = deque(maxlen=max(1, int(inference_config.vote_window)))
        self._idle_since_ms: Optional[float] = None
        self._last_transition_ms: float = -1e9

    def _class_threshold(self, label: int) -> float:
        if label == 1:
            return float(self.inference_config.fist_confidence_threshold)
        if label == 2:
            return float(self.inference_config.pinch_confidence_threshold)
        return float(self.inference_config.confidence_threshold)

    def _vote(self) -> int:
        if not self._vote_window:
            return 0
        counts = Counter(self._vote_window)
        label, count = counts.most_common(1)[0]
        if count >= int(self.inference_config.vote_min_count):
            return int(label)
        return 0

    def _accept_action_event(
        self,
        predicted_label: int,
        confidence: float,
        *,
        current_state_confidence: float,
        top2_confidence: float,
        now_ms: float,
    ) -> bool:
        if predicted_label not in {1, 2}:
            return False

        current_state_label = int(self.current_state)
        switching = current_state_label in {1, 2} and predicted_label != current_state_label
        threshold = self._class_threshold(predicted_label)
        if switching:
            threshold += float(self.inference_config.switch_confidence_boost)
        if confidence < threshold:
            return False

        margin_to_top2 = float(confidence) - float(top2_confidence)
        if margin_to_top2 < float(self.inference_config.activation_margin_threshold):
            return False

        if switching:
            if (now_ms - self._last_transition_ms) < float(self.runtime_config.post_transition_lock_ms):
                return False
            margin_to_current = float(confidence) - float(current_state_confidence)
            if margin_to_current < float(self.inference_config.switch_margin_threshold):
                return False
        return True

    def update(
        self,
        predicted_label: int,
        confidence: float,
        energy: float,
        now_ms: float,
        *,
        current_state_confidence: float = 0.0,
        top2_confidence: float = 0.0,
    ) -> RuntimeDecision:
        now_ms = float(now_ms)
        active_event = self._accept_action_event(
            predicted_label,
            float(confidence),
            current_state_confidence=float(current_state_confidence),
            top2_confidence=float(top2_confidence),
            now_ms=now_ms,
        )
        if active_event:
            last_active = next((label for label in reversed(self._vote_window) if label in {1, 2}), None)
            if last_active is not None and last_active != int(predicted_label):
                self._vote_window.clear()
            self._vote_window.append(int(predicted_label))
            voted_label = self._vote()
            self._idle_since_ms = None
            if voted_label in {1, 2} and (now_ms - self._last_transition_ms) >= float(self.runtime_config.min_transition_gap_ms):
                target_state = GestureType.FIST if voted_label == 1 else GestureType.PINCH
                changed = target_state != self.current_state
                self.current_state = target_state
                self._last_transition_ms = now_ms
                return RuntimeDecision(state=self.current_state, changed=changed, emitted_label=voted_label, voted_label=voted_label)
            return RuntimeDecision(state=self.current_state, changed=False, emitted_label=0, voted_label=voted_label)

        self._vote_window.append(0)
        if energy <= self.low_energy_threshold:
            if self._idle_since_ms is None:
                self._idle_since_ms = now_ms
        else:
            self._idle_since_ms = None

        if self._idle_since_ms is not None and (now_ms - self._idle_since_ms) >= float(self.runtime_config.idle_release_hold_ms):
            changed = self.current_state != GestureType.RELAX
            self.current_state = GestureType.RELAX
            self._last_transition_ms = now_ms if changed else self._last_transition_ms
            return RuntimeDecision(state=self.current_state, changed=changed, emitted_label=0, voted_label=0)

        return RuntimeDecision(state=self.current_state, changed=False, emitted_label=0, voted_label=0)


class EventFeatureExtractor:
    """Build aligned EMG and IMU runtime inputs from causal windows."""

    def __init__(self, data_config: EventDataConfig):
        self.data_config = data_config
        self._stft_pipeline = PreprocessPipeline(
            {
                "sampling_rate": data_config.device_sampling_rate_hz,
                "num_channels": 8,
                "target_length": data_config.context_samples,
                "segment_length": data_config.context_samples,
                "segment_stride": data_config.window_step_samples,
                "stft_window": data_config.feature.emg_stft_window,
                "stft_hop": data_config.feature.emg_stft_hop,
                "n_fft": data_config.feature.emg_n_fft,
                "freq_bins_out": data_config.feature.emg_freq_bins,
                "normalize": "log",
                "clip_min": 0.0,
                "clip_max": 10.0,
                "dual_branch": {"enabled": False},
            }
        )

    def _resample_imu(self, imu_window: np.ndarray) -> np.ndarray:
        target_steps = int(self.data_config.feature.imu_resample_steps)
        if imu_window.shape[0] == target_steps:
            resampled = imu_window
        else:
            src = np.linspace(0.0, 1.0, imu_window.shape[0], dtype=np.float32)
            dst = np.linspace(0.0, 1.0, target_steps, dtype=np.float32)
            resampled = np.empty((target_steps, imu_window.shape[1]), dtype=np.float32)
            for channel_idx in range(imu_window.shape[1]):
                resampled[:, channel_idx] = np.interp(dst, src, imu_window[:, channel_idx]).astype(np.float32)
        centered = resampled - np.mean(resampled, axis=0, keepdims=True)
        std = np.std(centered, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (centered / std).T.astype(np.float32)

    def build_inputs(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        emg_window = matrix[:, :8]
        imu_window = matrix[:, 8:14]
        energy = float(np.mean(np.abs(emg_window)))
        emg_feature = self._stft_pipeline.process_window(emg_window)
        imu_feature = self._resample_imu(imu_window)
        return emg_feature, imu_feature, energy


def build_runtime_inputs(matrix: np.ndarray, data_config: EventDataConfig) -> tuple[np.ndarray, np.ndarray, float]:
    extractor = EventFeatureExtractor(data_config)
    return extractor.build_inputs(matrix)


class EventOnsetController:
    """Reusable event-onset runtime controller for live and replay flows."""

    def __init__(
        self,
        *,
        data_config: EventDataConfig,
        inference_config: EventInferenceConfig,
        runtime_config: EventRuntimeBehaviorConfig,
        predict_proba: Callable[[np.ndarray, np.ndarray], np.ndarray],
        actuator=None,
    ):
        self.data_config = data_config
        self.predict_proba = predict_proba
        self.actuator = actuator
        self.extractor = EventFeatureExtractor(data_config)
        self.state_machine = EventRuntimeStateMachine(
            inference_config=inference_config,
            runtime_config=runtime_config,
            low_energy_threshold=float(runtime_config.low_energy_release_threshold or data_config.quality_filter.energy_min),
        )
        self._buffer: Deque[np.ndarray] = deque(maxlen=int(data_config.context_samples))
        self._processed_samples = 0
        self._next_infer_sample = int(data_config.context_samples)

    @property
    def current_state(self) -> GestureType:
        return self.state_machine.current_state

    def ingest_rows(self, rows: np.ndarray) -> list[ControllerStep]:
        if rows.size == 0:
            return []
        if rows.ndim != 2 or rows.shape[1] < 14:
            raise ValueError(f"Expected input rows shaped (N, >=14), got {rows.shape}")
        steps: list[ControllerStep] = []
        for row in np.asarray(rows[:, :14], dtype=np.float32):
            self._buffer.append(row.copy())
            self._processed_samples += 1
            if len(self._buffer) >= int(self.data_config.context_samples) and self._processed_samples >= self._next_infer_sample:
                step = self._process_current()
                if step is not None:
                    steps.append(step)
                self._next_infer_sample += int(self.data_config.window_step_samples)
        return steps

    def _process_current(self) -> ControllerStep | None:
        context = int(self.data_config.context_samples)
        if len(self._buffer) < context:
            return None
        matrix = np.asarray(self._buffer, dtype=np.float32)[-context:]
        emg_feature, imu_feature, energy = self.extractor.build_inputs(matrix)
        probs = np.asarray(self.predict_proba(emg_feature, imu_feature), dtype=np.float32)
        predicted_label = int(np.argmax(probs))
        confidence = float(np.max(probs))
        sorted_probs = np.sort(probs)[::-1]
        top2_confidence = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
        current_state_confidence = float(probs[int(self.state_machine.current_state)]) if int(self.state_machine.current_state) < probs.shape[0] else 0.0
        now_ms = float(self._processed_samples) * 1000.0 / float(self.data_config.device_sampling_rate_hz)
        decision = self.state_machine.update(
            predicted_label,
            confidence,
            energy,
            now_ms,
            current_state_confidence=current_state_confidence,
            top2_confidence=top2_confidence,
        )
        if decision.changed and self.actuator is not None:
            self.actuator.execute_gesture(decision.state)
        return ControllerStep(
            sample_index=self._processed_samples,
            now_ms=now_ms,
            energy=energy,
            predicted_label=predicted_label,
            confidence=confidence,
            decision=decision,
        )
