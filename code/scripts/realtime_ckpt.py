"""Manual realtime checkpoint debugging for the fixed 8-channel protocol."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

try:
    import emg_armband as emg

    EMG_IMPORT_ERROR = None
except ImportError as exc:
    emg = None
    EMG_IMPORT_ERROR = exc

from runtime.control.controller import RuntimeController
from runtime.inference import InferenceRateScheduler, TemporalVoter
from shared.config import (
    RuntimeConfig,
    get_protocol_input_shape,
    get_protocol_num_channels,
    load_runtime_config,
    load_training_config,
    normalize_model_config_channels,
)
from shared.gestures import GestureType, LABEL_NAME_MAP, NUM_CLASSES
from shared.preprocessing import PreprocessPipeline
from training.model import build_model_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realtime_ckpt")

try:
    import mindspore as ms
    from mindspore import Tensor, load_checkpoint, load_param_into_net

    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    logger.warning("MindSpore is not installed. Falling back to mock inference mode.")

EMG_CENTER_VALUE = 128.0

GESTURE_DISPLAY = {
    "relax": "RELAX",
    "fist": "FIST",
    "pinch": "PINCH",
    "ok": "OK",
    "ye": "YE",
    "sidegrip": "SIDEGRIP",
    "unknown": "UNKNOWN",
}


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def _resolve_existing_path(path_value: str | os.PathLike[str]) -> Path:
    candidate = Path(path_value)
    if candidate.exists():
        return candidate.resolve()
    alt = (_CODE_DIR / candidate).resolve()
    if alt.exists():
        return alt
    return candidate


class NeuroGripRealtimeEngine:
    """Realtime inference engine using the current dual-branch 8-channel protocol."""

    def __init__(
        self,
        ckpt_path: str | os.PathLike[str],
        *,
        model_config,
        runtime_config: RuntimeConfig,
        threshold: float,
        infer_rate_hz: float,
        device_target: str,
        force_mock: bool = False,
    ):
        self.ckpt_path = str(ckpt_path)
        self.runtime_config = runtime_config
        self.model_config = normalize_model_config_channels(
            model_config,
            runtime_config.preprocess,
            logger=logger,
            context="realtime ckpt debug protocol",
        )
        self.threshold = float(threshold)
        self.preprocess = PreprocessPipeline(runtime_config.preprocess)
        self.expected_input_shape = get_protocol_input_shape(runtime_config.preprocess)
        self.num_channels = get_protocol_num_channels(runtime_config.preprocess)
        self.base_window_size = int(self.preprocess.get_required_window_size())
        self.stride = int(self.preprocess.get_required_window_stride())
        self.tta_offsets = list(runtime_config.inference.tta_offsets or [0.0])
        self.read_window_size = RuntimeController._calc_read_window_size(
            self.base_window_size,
            self.stride,
            self.tta_offsets,
        )

        self.model = None
        self._mock_mode = bool(force_mock or not MINDSPORE_AVAILABLE)
        self._load_model(device_target)

        self._sample_buffer: deque[np.ndarray] = deque(maxlen=self.read_window_size * 2)
        self._voter = TemporalVoter(
            history_window_ms=runtime_config.inference.smoothing_window_ms,
            hysteresis_count=runtime_config.inference.hysteresis_count,
        )
        self._infer_scheduler = InferenceRateScheduler(infer_rate_hz)
        self._rng = np.random.default_rng(42)

        self.inference_count = 0
        self.total_frames = 0
        self._last_infer_ms = 0.0
        self._last_gesture_id = int(GestureType.RELAX)
        self._last_confidence = 0.0

    def _load_model(self, device_target: str) -> None:
        if self._mock_mode:
            logger.warning("Realtime checkpoint debug is running in mock inference mode.")
            return

        try:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target)
        except Exception:
            ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
            logger.info("Fallback to CPU device")

        self.model = build_model_from_config(self.model_config, dropout_rate=self.model_config.dropout_rate)
        self.model.set_train(False)

        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        param_dict = load_checkpoint(str(ckpt_path))
        not_loaded, _ = load_param_into_net(self.model, param_dict)
        if not_loaded:
            logger.warning("Some checkpoint params were not loaded: %s", not_loaded)

        total_params = sum(param.size for param in self.model.trainable_params())
        logger.info(
            "Loaded model type=%s params=%s device=%s expected_input_shape=%s",
            self.model_config.model_type,
            f"{total_params:,}",
            device_target,
            self.expected_input_shape,
        )

    def feed_frame(self, frame_event) -> Optional[Tuple[int, str, float]]:
        self.total_frames += 1

        emg_packs = getattr(frame_event.emg_event, "emg", [])
        for pack in emg_packs:
            sample = np.asarray(pack[: self.num_channels], dtype=np.float32)
            sample -= EMG_CENTER_VALUE
            self._sample_buffer.append(sample)

        if len(self._sample_buffer) < self.read_window_size:
            return None

        if not self._infer_scheduler.should_run():
            return None

        raw_window = np.asarray(list(self._sample_buffer)[-self.read_window_size :], dtype=np.float32)
        slices = RuntimeController._slice_tta_windows(
            raw_window,
            self.base_window_size,
            self.stride,
            self.tta_offsets,
        )
        if not slices:
            return None

        probs = []
        t0 = time.perf_counter()
        for segment in slices:
            feature = self.preprocess.process_window(segment)
            feature_shape = (1,) + tuple(feature.shape)
            if feature_shape != self.expected_input_shape:
                raise ValueError(
                    f"Realtime debug feature shape mismatch: {feature_shape} != {self.expected_input_shape}"
                )
            probs.append(self._predict_probs(feature))
        self._last_infer_ms = (time.perf_counter() - t0) * 1000.0

        mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
        raw_gesture_id = int(np.argmax(mean_prob))
        confidence = float(np.max(mean_prob))
        stable_gesture = self._voter.update(raw_gesture_id, confidence, now=time.time())
        if stable_gesture is not None and confidence >= self.threshold:
            emitted_id = int(stable_gesture)
        else:
            emitted_id = int(GestureType.RELAX)

        self._last_gesture_id = emitted_id
        self._last_confidence = confidence
        self.inference_count += 1
        return emitted_id, LABEL_NAME_MAP.get(emitted_id, "unknown"), confidence

    def _predict_probs(self, feature: np.ndarray) -> np.ndarray:
        if self._mock_mode:
            logits = self._rng.standard_normal(NUM_CLASSES).astype(np.float32)
            return _softmax(logits).astype(np.float32)

        input_tensor = Tensor(feature[np.newaxis, ...].astype(np.float32), ms.float32)
        logits = self.model(input_tensor).asnumpy()[0]
        return _softmax(logits).astype(np.float32)

    def get_buffer_progress(self) -> float:
        return min(1.0, len(self._sample_buffer) / max(1, self.read_window_size))

    def reset(self) -> None:
        self._sample_buffer.clear()
        self._voter = TemporalVoter(
            history_window_ms=self.runtime_config.inference.smoothing_window_ms,
            hysteresis_count=self.runtime_config.inference.hysteresis_count,
        )
        self._infer_scheduler.reset()
        self._last_gesture_id = int(GestureType.RELAX)
        self._last_confidence = 0.0

    @property
    def last_gesture(self) -> str:
        return LABEL_NAME_MAP.get(self._last_gesture_id, "unknown")

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def last_latency_ms(self) -> float:
        return self._last_infer_ms


class RealtimeCkptListener((emg.DeviceListener if emg is not None else object)):
    def __init__(self, engine: NeuroGripRealtimeEngine, print_interval: float = 0.25, on_gesture_change=None):
        self.engine = engine
        self.print_interval = print_interval
        self.on_gesture_change = on_gesture_change
        self._last_print_time = 0.0
        self._last_gesture = None

    def on_connected(self, device):
        logger.info("Armband connected: %s", getattr(device, "port", "unknown"))

    def on_disconnected(self, device):
        del device
        logger.info("Armband disconnected")

    def on_frame(self, event):
        result = self.engine.feed_frame(event)
        now = time.time()

        if result is None:
            if now - self._last_print_time > 0.5:
                progress = self.engine.get_buffer_progress()
                print(
                    f"\r  Buffering: {progress * 100:.0f}% (needs {self.engine.read_window_size} raw samples)",
                    end="",
                    flush=True,
                )
                self._last_print_time = now
            return

        gesture_id, gesture_name, confidence = result
        if gesture_name != self._last_gesture:
            self._last_gesture = gesture_name
            if self.on_gesture_change:
                self.on_gesture_change(gesture_id, gesture_name, confidence)

        if now - self._last_print_time >= self.print_interval:
            display = GESTURE_DISPLAY.get(gesture_name, gesture_name.upper())
            bar_len = int(confidence * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            latency = self.engine.last_latency_ms
            print(
                f"\r  Gesture: {display:<10s} [{bar}] {confidence:5.1%} "
                f"Infer: {latency:5.1f}ms Frames:{self.engine.total_frames:6d} "
                f"Calls:{self.engine.inference_count:5d}",
                end="",
                flush=True,
            )
            self._last_print_time = now


def _load_runtime_and_model_configs(args: argparse.Namespace):
    runtime_cfg_path = _resolve_existing_path(args.runtime_config)
    training_cfg_path = _resolve_existing_path(args.training_config)

    runtime_cfg = load_runtime_config(runtime_cfg_path)
    model_cfg, preprocess_cfg, _, _ = load_training_config(training_cfg_path)
    model_cfg = normalize_model_config_channels(
        model_cfg,
        preprocess_cfg,
        logger=logger,
        context="training protocol",
    )

    runtime_shape = get_protocol_input_shape(runtime_cfg.preprocess)
    training_shape = get_protocol_input_shape(preprocess_cfg)
    if runtime_shape != training_shape:
        raise ValueError(
            f"Runtime/training preprocess mismatch: runtime={runtime_shape}, training={training_shape}"
        )

    return runtime_cfg_path, training_cfg_path, runtime_cfg, model_cfg


def run_realtime(args: argparse.Namespace) -> None:
    if emg is None:
        raise ImportError(
            "emg_armband dependency is missing. Install pyserial and vendor libs first."
        ) from EMG_IMPORT_ERROR

    runtime_cfg_path, training_cfg_path, runtime_cfg, model_cfg = _load_runtime_and_model_configs(args)
    logger.info("Loaded runtime config: %s", runtime_cfg_path)
    logger.info("Loaded training config: %s", training_cfg_path)

    threshold = (
        runtime_cfg.inference.confidence_threshold if args.threshold is None else float(args.threshold)
    )
    infer_rate_hz = (
        runtime_cfg.inference.infer_rate_hz if args.infer_rate_hz is None else float(args.infer_rate_hz)
    )
    hysteresis_count = (
        runtime_cfg.inference.hysteresis_count
        if args.hysteresis_count is None
        else int(args.hysteresis_count)
    )
    runtime_cfg.inference.confidence_threshold = float(threshold)
    runtime_cfg.inference.infer_rate_hz = float(infer_rate_hz)
    runtime_cfg.inference.hysteresis_count = int(hysteresis_count)
    runtime_cfg.infer_rate_hz = float(infer_rate_hz)

    port = args.port or runtime_cfg.hardware.sensor_port or "COM4"
    runtime_cfg.hardware.sensor_port = port

    device_target = args.device or runtime_cfg.device.target
    device_target = {
        "NPU": "Ascend",
        "GPU": "GPU",
        "CPU": "CPU",
        "ASCEND": "Ascend",
        "ascend": "Ascend",
    }.get(device_target, device_target)
    runtime_cfg.device.target = device_target

    ckpt_path = _resolve_existing_path(args.ckpt)
    if not ckpt_path.exists() and not args.mock:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print()
    print("+" + "-" * 72 + "+")
    print("|" + " NeuroGrip Realtime Checkpoint Debug ".center(72) + "|")
    print("+" + "-" * 72 + "+")
    print(f"  Runtime config:  {runtime_cfg_path}")
    print(f"  Training config: {training_cfg_path}")
    print(f"  Checkpoint:      {ckpt_path}")
    print(f"  Model type:      {model_cfg.model_type}")
    print(f"  Device:          {device_target}")
    print(f"  Serial port:     {port}")
    print(f"  Threshold:       {threshold}")
    print(f"  Hysteresis:      {hysteresis_count}")
    print(f"  Infer rate Hz:   {infer_rate_hz} (0 means no limit)")
    print(f"  Input shape:     {get_protocol_input_shape(runtime_cfg.preprocess)}")
    print()

    engine = NeuroGripRealtimeEngine(
        ckpt_path=ckpt_path,
        model_config=model_cfg,
        runtime_config=runtime_cfg,
        threshold=float(threshold),
        infer_rate_hz=float(infer_rate_hz),
        device_target=device_target,
        force_mock=bool(args.mock),
    )

    def on_gesture_change(_gesture_id, gesture_name, confidence):
        if gesture_name != "relax":
            print(f"\n  >>> Gesture changed: {gesture_name.upper()} (confidence={confidence:.1%})")

    listener = RealtimeCkptListener(engine=engine, print_interval=0.2, on_gesture_change=on_gesture_change)
    hub = emg.Hub(port=port)

    print("  " + "-" * 60)
    print("  Manual debug only. Use runtime.run or benchmark_realtime_ckpt.py for acceptance.")
    print("  Press Ctrl+C to stop")
    print("  " + "-" * 60)
    print()

    try:
        hub.run(listener)
    except KeyboardInterrupt:
        print("\n")
    finally:
        hub.stop()
        print()
        print("  Runtime stats:")
        print(f"  Total frames:      {engine.total_frames}")
        print(f"  Inference calls:   {engine.inference_count}")
        print(f"  Last gesture:      {engine.last_gesture} ({engine.last_confidence:.1%})")
        print(f"  Last latency:      {engine.last_latency_ms:.1f} ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NeuroGrip realtime checkpoint debug using the current 8-channel protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/realtime_ckpt.py
  python scripts/realtime_ckpt.py --ckpt checkpoints/neurogrip_best.ckpt
  python scripts/realtime_ckpt.py --port COM3 --threshold 0.6
  python scripts/realtime_ckpt.py --infer_rate_hz 20 --hysteresis_count 3
  python scripts/realtime_ckpt.py --runtime_config configs/runtime.yaml --training_config configs/training.yaml
        """,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/neurogrip_best.ckpt",
        help="MindSpore .ckpt path",
    )
    parser.add_argument(
        "--runtime_config",
        "--config",
        dest="runtime_config",
        type=str,
        default="configs/runtime.yaml",
        help="Runtime YAML config path",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training.yaml",
        help="Training YAML config path",
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port")
    parser.add_argument("--device", type=str, default=None, help="Inference device: CPU/GPU/Ascend")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    parser.add_argument(
        "--hysteresis_count",
        type=int,
        default=None,
        help="Temporal voter hysteresis override",
    )
    parser.add_argument(
        "--infer_rate_hz",
        type=float,
        default=None,
        help="Inference frequency limit in Hz (0 = no limit)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock inference mode even when MindSpore is available.",
    )
    return parser


def main() -> None:
    logger.info(
        "Manual realtime checkpoint debug follows the current 8-channel dual-branch protocol. "
        "Use runtime.run or scripts/benchmark_realtime_ckpt.py for acceptance checks."
    )
    args = build_parser().parse_args()
    run_realtime(args)


if __name__ == "__main__":
    main()
