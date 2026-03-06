# -*- coding: utf-8 -*-
"""
Realtime gesture inference directly from a MindSpore .ckpt checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter, deque
from typing import Optional, Tuple

import numpy as np

# Ensure `code/` is importable when script is run as `python scripts/realtime_ckpt.py`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_SCRIPT_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

try:
    import emg_armband as emg

    EMG_IMPORT_ERROR = None
except ImportError as exc:
    emg = None
    EMG_IMPORT_ERROR = exc

from runtime.inference.scheduler import InferenceRateScheduler
from shared.gestures import LABEL_NAME_MAP, NUM_CLASSES
from shared.preprocessing import PreprocessPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realtime_ckpt")

try:
    import mindspore as ms
    from mindspore import Tensor, load_checkpoint, load_param_into_net

    from shared.models import create_model

    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    logger.warning("MindSpore is not installed. Fallback to mock inference mode.")

DEVICE_SAMPLING_RATE = 1000
TARGET_SAMPLING_RATE = 200
DECIMATE_RATIO = DEVICE_SAMPLING_RATE // TARGET_SAMPLING_RATE
NUM_USE_CHANNELS = 6
EMG_CENTER_VALUE = 128.0
SEGMENT_LENGTH = 84

DEFAULT_THRESHOLD = 0.5
DEFAULT_VOTE_WINDOW = 5
DEFAULT_VOTE_MIN_COUNT = 3
DEFAULT_INFER_RATE_HZ = 0.0

GESTURE_DISPLAY = {
    "relax": "RELAX",
    "fist": "FIST",
    "pinch": "PINCH",
    "ok": "OK",
    "ye": "YE",
    "sidegrip": "SIDEGRIP",
    "unknown": "UNKNOWN",
}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class NeuroGripRealtimeEngine:
    """Realtime inference engine for armband stream -> checkpoint model."""

    def __init__(
        self,
        ckpt_path: str,
        model_config: Optional[dict] = None,
        preprocess_config: Optional[dict] = None,
        threshold: float = DEFAULT_THRESHOLD,
        vote_window: int = DEFAULT_VOTE_WINDOW,
        vote_min_count: int = DEFAULT_VOTE_MIN_COUNT,
        infer_rate_hz: float = DEFAULT_INFER_RATE_HZ,
        device: str = "CPU",
    ):
        self.ckpt_path = ckpt_path
        self.threshold = threshold
        self.vote_window = vote_window
        self.vote_min_count = vote_min_count

        if model_config is None:
            model_config = {
                "model_type": "standard",
                "in_channels": NUM_USE_CHANNELS,
                "num_classes": NUM_CLASSES,
                "base_channels": 16,
                "use_se": True,
                "dropout_rate": 0.0,
            }
        self.model_config = model_config

        if preprocess_config is None:
            preprocess_config = {
                "sampling_rate": TARGET_SAMPLING_RATE,
                "num_channels": NUM_USE_CHANNELS,
                "lowcut": 20.0,
                "highcut": 90.0,
                "filter_order": 4,
                "stft_window_size": 24,
                "stft_hop_size": 12,
                "stft_n_fft": 46,
            }
        self.preprocess_config = preprocess_config

        self.pipeline = PreprocessPipeline(
            sampling_rate=preprocess_config["sampling_rate"],
            num_channels=preprocess_config["num_channels"],
            lowcut=preprocess_config["lowcut"],
            highcut=preprocess_config["highcut"],
            filter_order=preprocess_config["filter_order"],
            stft_window_size=preprocess_config["stft_window_size"],
            stft_hop_size=preprocess_config["stft_hop_size"],
            stft_n_fft=preprocess_config["stft_n_fft"],
        )

        self.model = None
        self._mock_mode = False
        self._load_model(ckpt_path, model_config, device)

        self._sample_buffer: deque = deque(maxlen=SEGMENT_LENGTH * 2)
        self._decimate_counter = 0

        self._vote_buffer: deque = deque(maxlen=vote_window)
        self._last_gesture_id = 0
        self._last_confidence = 0.0

        self._infer_scheduler = InferenceRateScheduler(infer_rate_hz)

        self.inference_count = 0
        self.total_frames = 0
        self._last_infer_ms = 0.0

    def _load_model(self, ckpt_path: str, config: dict, device: str):
        if not MINDSPORE_AVAILABLE:
            self._mock_mode = True
            return

        try:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=device)
        except Exception:
            ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
            logger.info("Fallback to CPU device")

        self.model = create_model(config)
        self.model.set_train(False)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        param_dict = load_checkpoint(ckpt_path)
        not_loaded, _ = load_param_into_net(self.model, param_dict)
        if not_loaded:
            logger.warning("Some params are not loaded: %s", not_loaded)

        total_params = sum(p.size for p in self.model.trainable_params())
        logger.info(
            "Loaded model type=%s params=%s device=%s",
            config["model_type"],
            f"{total_params:,}",
            device,
        )

    def feed_frame(self, frame_event) -> Optional[Tuple[int, str, float]]:
        self.total_frames += 1

        emg_packs = frame_event.emg_event.emg
        for pack in emg_packs:
            self._decimate_counter += 1
            if self._decimate_counter % DECIMATE_RATIO != 0:
                continue
            sample = np.asarray(pack[:NUM_USE_CHANNELS], dtype=np.float32)
            sample -= EMG_CENTER_VALUE
            self._sample_buffer.append(sample)

        if len(self._sample_buffer) < SEGMENT_LENGTH:
            return None

        if not self._infer_scheduler.should_run():
            return None

        window = np.asarray(list(self._sample_buffer)[-SEGMENT_LENGTH:], dtype=np.float32)

        try:
            spectrogram = self.pipeline.process_window(window)
        except Exception as exc:
            logger.debug("Preprocess failed: %s", exc)
            return None

        gesture_id, confidence = self._infer(spectrogram)

        if confidence >= self.threshold:
            self._vote_buffer.append(gesture_id)
        else:
            self._vote_buffer.append(0)

        voted_id = self._majority_vote()
        voted_name = LABEL_NAME_MAP.get(voted_id, "unknown")

        self._last_gesture_id = voted_id
        self._last_confidence = confidence
        self.inference_count += 1

        return voted_id, voted_name, confidence

    def _infer(self, spectrogram: np.ndarray) -> Tuple[int, float]:
        if self._mock_mode:
            return self._mock_infer()

        x = spectrogram[np.newaxis, ...].astype(np.float32)
        t0 = time.perf_counter()
        input_tensor = Tensor(x, ms.float32)
        logits = self.model(input_tensor)
        logits_np = logits.asnumpy()[0]
        self._last_infer_ms = (time.perf_counter() - t0) * 1000

        probs = softmax(logits_np)
        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])
        return gesture_id, confidence

    def _mock_infer(self) -> Tuple[int, float]:
        time.sleep(0.01)
        gesture_id = np.random.randint(0, NUM_CLASSES)
        confidence = np.random.uniform(0.3, 0.95)
        self._last_infer_ms = 10.0
        return int(gesture_id), float(confidence)

    def _majority_vote(self) -> int:
        if not self._vote_buffer:
            return 0
        counts = Counter(self._vote_buffer)
        most_common_id, most_common_count = counts.most_common(1)[0]
        if most_common_count >= self.vote_min_count:
            return int(most_common_id)
        return self._last_gesture_id

    def get_buffer_progress(self) -> float:
        return min(1.0, len(self._sample_buffer) / SEGMENT_LENGTH)

    def reset(self):
        self._sample_buffer.clear()
        self._vote_buffer.clear()
        self._decimate_counter = 0
        self._last_gesture_id = 0
        self._last_confidence = 0.0
        self._infer_scheduler.reset()

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
        logger.info("Armband connected: %s", device.port)

    def on_disconnected(self, device):
        logger.info("Armband disconnected")

    def on_frame(self, event):
        result = self.engine.feed_frame(event)
        now = time.time()

        if result is None:
            if now - self._last_print_time > 0.5:
                progress = self.engine.get_buffer_progress()
                print(
                    f"\r  Buffering: {progress * 100:.0f}% (needs {SEGMENT_LENGTH} samples at 200Hz)",
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
            display = GESTURE_DISPLAY.get(gesture_name, gesture_name)
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


def load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using defaults")
        return {}

    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_realtime(args):
    if emg is None:
        raise ImportError(
            "emg_armband dependency is missing. Install pyserial and vendor libs first."
        ) from EMG_IMPORT_ERROR

    yaml_cfg = {}
    if args.config and os.path.exists(args.config):
        yaml_cfg = load_yaml_config(args.config)
        logger.info("Loaded config: %s", args.config)

    model_cfg = yaml_cfg.get("model", {})
    model_config = {
        "model_type": model_cfg.get("model_type", "standard"),
        "in_channels": model_cfg.get("in_channels", NUM_USE_CHANNELS),
        "num_classes": model_cfg.get("num_classes", NUM_CLASSES),
        "base_channels": model_cfg.get("base_channels", 16),
        "use_se": model_cfg.get("use_se", True),
        "dropout_rate": 0.0,
    }

    prep_cfg = yaml_cfg.get("preprocess", {})
    preprocess_config = {
        "sampling_rate": prep_cfg.get("sampling_rate", TARGET_SAMPLING_RATE),
        "num_channels": prep_cfg.get("num_channels", NUM_USE_CHANNELS),
        "lowcut": prep_cfg.get("lowcut", 20.0),
        "highcut": prep_cfg.get("highcut", 90.0),
        "filter_order": prep_cfg.get("filter_order", 4),
        "stft_window_size": prep_cfg.get("stft_window_size", 24),
        "stft_hop_size": prep_cfg.get("stft_hop_size", 12),
        "stft_n_fft": prep_cfg.get("stft_n_fft", 46),
    }

    runtime_cfg = yaml_cfg.get("runtime", {}) if isinstance(yaml_cfg.get("runtime", {}), dict) else {}

    threshold = args.threshold if args.threshold is not None else yaml_cfg.get("confidence_threshold", DEFAULT_THRESHOLD)
    vote_window = args.vote_window if args.vote_window is not None else yaml_cfg.get("vote_window_size", DEFAULT_VOTE_WINDOW)
    vote_min = args.vote_min if args.vote_min is not None else yaml_cfg.get("vote_min_count", DEFAULT_VOTE_MIN_COUNT)

    infer_rate_hz = (
        args.infer_rate_hz
        if args.infer_rate_hz is not None
        else runtime_cfg.get("infer_rate_hz", yaml_cfg.get("infer_rate_hz", DEFAULT_INFER_RATE_HZ))
    )

    port = args.port
    if port is None:
        hw_cfg = yaml_cfg.get("hardware", {})
        port = hw_cfg.get("sensor_port", None) or "COM4"

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        alt = os.path.join(_CODE_DIR, ckpt_path)
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            print(f"Checkpoint file not found: {ckpt_path}")
            sys.exit(1)

    device = args.device or yaml_cfg.get("inference", {}).get("device", "CPU")
    device_map = {
        "NPU": "Ascend",
        "GPU": "GPU",
        "CPU": "CPU",
        "ASCEND": "Ascend",
        "ascend": "Ascend",
    }
    device = device_map.get(device, device)

    print()
    print("+" + "-" * 64 + "+")
    print("|" + " NeuroGrip Realtime Inference (.ckpt) ".center(64) + "|")
    print("+" + "-" * 64 + "+")
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"  Model type:     {model_config['model_type']}")
    print(f"  Device:         {device}")
    print(f"  Serial port:    {port}")
    print(f"  Threshold:      {threshold}")
    print(f"  Vote window:    {vote_window} (min={vote_min})")
    print(f"  Infer rate Hz:  {infer_rate_hz} (0 means no limit)")
    print()

    engine = NeuroGripRealtimeEngine(
        ckpt_path=ckpt_path,
        model_config=model_config,
        preprocess_config=preprocess_config,
        threshold=threshold,
        vote_window=vote_window,
        vote_min_count=vote_min,
        infer_rate_hz=float(infer_rate_hz),
        device=device,
    )

    def on_gesture_change(_gesture_id, gesture_name, confidence):
        if gesture_name != "relax":
            print(f"\n  >>> Gesture changed: {gesture_name} (confidence={confidence:.1%})")

    listener = RealtimeCkptListener(engine=engine, print_interval=0.2, on_gesture_change=on_gesture_change)
    hub = emg.Hub(port=port)

    print("  " + "-" * 56)
    print("  Press Ctrl+C to stop")
    print("  " + "-" * 56)
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


def main():
    parser = argparse.ArgumentParser(
        description="NeuroGrip realtime gesture inference (.ckpt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/realtime_ckpt.py
  python scripts/realtime_ckpt.py --ckpt checkpoints/neurogrip_best.ckpt
  python scripts/realtime_ckpt.py --port COM3 --threshold 0.6
  python scripts/realtime_ckpt.py --infer_rate_hz 20
  python scripts/realtime_ckpt.py --config configs/runtime.yaml --device CPU
        """,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/neurogrip_best.ckpt",
        help="MindSpore .ckpt path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/runtime.yaml",
        help="Runtime YAML config path",
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port")
    parser.add_argument("--device", type=str, default=None, help="Inference device: CPU/GPU/Ascend")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--vote-window", type=int, default=None, help="Vote window size")
    parser.add_argument("--vote-min", type=int, default=None, help="Minimum votes to emit gesture")
    parser.add_argument(
        "--infer_rate_hz",
        type=float,
        default=None,
        help="Inference frequency limit in Hz (0 = no limit)",
    )

    args = parser.parse_args()
    run_realtime(args)


if __name__ == "__main__":
    main()
