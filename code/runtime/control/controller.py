"""
Main runtime controller.

Control loop:
    sensor.read_window -> preprocess -> inference -> vote -> actuator
"""

import logging
import signal
import time
from pathlib import Path
from typing import Optional

from runtime.control.state_machine import SystemState, SystemStateMachine
from runtime.hardware.base import ActuatorBase, SensorBase
from runtime.hardware.factory import create_actuator, create_sensor
from runtime.inference import InferenceEngine, SlidingWindowVoter
from shared.config import RuntimeConfig
from shared.gestures import GestureType
from shared.preprocessing import PreprocessPipeline

logger = logging.getLogger(__name__)


class ProsthesisController:
    """
    Prosthesis runtime controller.

    It owns hardware lifecycle, control loop lifecycle, and state transitions.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._running = False

        self._cycle_count = 0
        self._last_gesture: Optional[GestureType] = None
        self.state_machine = SystemStateMachine()

        self._validate_config_consistency()

        pp_cfg = config.preprocess
        self.pipeline = PreprocessPipeline(
            sampling_rate=pp_cfg.sampling_rate,
            num_channels=pp_cfg.num_channels,
            lowcut=pp_cfg.lowcut,
            highcut=pp_cfg.highcut,
            filter_order=pp_cfg.filter_order,
            stft_window_size=pp_cfg.stft_window_size,
            stft_hop_size=pp_cfg.stft_hop_size,
            stft_n_fft=pp_cfg.stft_n_fft,
        )

        inf_cfg = config.inference
        self.engine = InferenceEngine(
            model_path=inf_cfg.model_path,
            device=inf_cfg.device,
            num_threads=inf_cfg.num_threads,
        )

        self.voter = SlidingWindowVoter(
            window_size=config.vote_window_size,
            min_count=config.vote_min_count,
            confidence_threshold=config.confidence_threshold,
        )

        hw_cfg = config.hardware
        self.sensor: SensorBase = create_sensor(hw_cfg)
        self.actuator: ActuatorBase = create_actuator(hw_cfg)

    def _expected_model_input_shape(self) -> tuple[int, int, int, int]:
        pp = self.config.preprocess
        freq_bins = pp.stft_n_fft // 2 + 1
        time_frames = max(1, (pp.segment_length - pp.stft_window_size) // pp.stft_hop_size + 1)
        return (1, pp.num_channels, freq_bins, time_frames)

    def _validate_config_consistency(self) -> None:
        cfg = self.config
        pp = cfg.preprocess

        if cfg.control_rate_hz <= 0:
            raise ValueError(f"control_rate_hz must be > 0, got {cfg.control_rate_hz}")
        if pp.stft_hop_size <= 0:
            raise ValueError(f"stft_hop_size must be > 0, got {pp.stft_hop_size}")
        if pp.stft_window_size <= 0:
            raise ValueError(f"stft_window_size must be > 0, got {pp.stft_window_size}")
        if pp.segment_length <= 0:
            raise ValueError(f"segment_length must be > 0, got {pp.segment_length}")
        if pp.segment_stride <= 0:
            raise ValueError(f"segment_stride must be > 0, got {pp.segment_stride}")

        if cfg.vote_min_count > cfg.vote_window_size:
            raise ValueError(
                f"vote_min_count ({cfg.vote_min_count}) cannot exceed "
                f"vote_window_size ({cfg.vote_window_size})."
            )

        if pp.segment_length < pp.stft_window_size:
            logger.warning(
                "segment_length (%s) < stft_window_size (%s). "
                "This works via zero-padding, but may degrade stability.",
                pp.segment_length,
                pp.stft_window_size,
            )
        if pp.segment_stride > pp.segment_length:
            logger.warning(
                "segment_stride (%s) > segment_length (%s). "
                "This can reduce effective sample overlap.",
                pp.segment_stride,
                pp.segment_length,
            )

        if cfg.model.in_channels != pp.num_channels:
            logger.warning(
                "Model in_channels (%s) != preprocess.num_channels (%s). "
                "This may cause shape mismatch at inference.",
                cfg.model.in_channels,
                pp.num_channels,
            )

        if cfg.hardware.sensor_mode == "armband" and int(pp.sampling_rate) != 200:
            logger.warning(
                "preprocess.sampling_rate=%s, but ArmbandSensor factory currently uses "
                "target_sampling_rate=200. Keep them aligned to avoid train/runtime drift.",
                pp.sampling_rate,
            )

        model_path = Path(cfg.inference.model_path)
        if not model_path.exists():
            logger.warning(
                "Inference model file not found yet: %s. "
                "This is expected before conversion, but runtime on device needs it.",
                model_path,
            )

        logger.info("Expected runtime model input shape: %s", self._expected_model_input_shape())

    def start(self, max_cycles: Optional[int] = None) -> None:
        """
        Start runtime loop.

        Args:
            max_cycles: Optional max number of control cycles for smoke tests.
        """
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be > 0 when specified.")

        logger.info("=" * 60)
        logger.info("NeuroGrip Pro V2 - Runtime Controller Start")
        logger.info("=" * 60)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.state_machine.transition_to(SystemState.CALIBRATING)

        if not self.sensor.connect():
            self.state_machine.set_error("Failed to connect sensor.")
            raise RuntimeError("Failed to connect sensor.")
        if not self.actuator.connect():
            self.sensor.disconnect()
            self.state_machine.set_error("Failed to connect actuator.")
            raise RuntimeError("Failed to connect actuator.")

        logger.info("Hardware ready.")
        logger.info("  Sensor: %s", self.sensor.get_info())
        logger.info("  Actuator: %s", self.actuator.get_info())
        logger.info("  Control rate: %s Hz", self.config.control_rate_hz)
        if max_cycles is not None:
            logger.info("  Max cycles: %s", max_cycles)

        self.state_machine.transition_to(SystemState.RUNNING)
        self._running = True
        try:
            self._main_loop(max_cycles=max_cycles)
        except Exception as exc:
            logger.error("Control loop error: %s", exc)
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.state_machine.transition_to(SystemState.STOPPING)

        logger.info("Stopping runtime controller...")

        self.actuator.disconnect()
        self.sensor.disconnect()

        latency = self.engine.get_latency_stats()
        logger.info("Runtime stats:")
        logger.info("  Total cycles: %s", self._cycle_count)
        logger.info("  Inference latency: %s", latency)

        logger.info("Runtime controller stopped.")
        self.state_machine.transition_to(SystemState.IDLE)

    def _main_loop(self, max_cycles: Optional[int] = None) -> None:
        cycle_interval = 1.0 / self.config.control_rate_hz
        window_size = self.config.preprocess.segment_length

        logger.info(
            "Control loop started (window=%s, interval=%.1fms).",
            window_size,
            cycle_interval * 1000,
        )
        logger.info("Press Ctrl+C to stop.")

        while self._running:
            cycle_start = time.perf_counter()
            try:
                self._control_step(window_size)
            except Exception as exc:
                logger.debug("Control step error: %s", exc)

            elapsed = time.perf_counter() - cycle_start
            sleep_time = cycle_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._cycle_count += 1
            if max_cycles is not None and self._cycle_count >= max_cycles:
                logger.info("Reached max_cycles=%s, exiting loop.", max_cycles)
                self._running = False

    def _control_step(self, window_size: int) -> None:
        window = self.sensor.read_window(window_size)
        if window is None:
            return

        spectrogram = self.pipeline.process_window(window)
        gesture_id, confidence = self.engine.predict(spectrogram)
        stable_gesture = self.voter.update(gesture_id, confidence)

        if stable_gesture is not None and stable_gesture != self._last_gesture:
            self.actuator.execute_gesture(stable_gesture)
            self._last_gesture = stable_gesture

            if self._cycle_count % 30 == 0:
                logger.info(
                    "[Cycle %s] gesture=%s confidence=%.3f",
                    self._cycle_count,
                    stable_gesture.name,
                    confidence,
                )

    def _signal_handler(self, signum, frame):
        del frame
        logger.info("Received exit signal: %s", signum)
        self._running = False
