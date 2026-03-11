"""Run the event-onset runtime controller using a checkpoint model."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.evaluate import load_event_model_from_checkpoint
from event_onset.runtime import EventOnsetController
from runtime.hardware.factory import create_actuator
from scripts.collection_utils import STANDARD_CSV_HEADERS

try:
    import mindspore as ms
    from mindspore import Tensor, context
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run event-onset runtime controller")
    parser.add_argument("--config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--device", default=None, choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--source_csv", default=None, help="Replay a standardized CSV instead of reading the live device.")
    parser.add_argument("--duration_sec", type=float, default=0.0, help="Optional live duration. 0 means until Ctrl+C.")
    parser.add_argument("--standalone", action="store_true", help="Use standalone actuator mock.")
    return parser


def _load_standardized_matrix(path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {path}")
        for row in reader:
            rows.append([float(row[field]) for field in STANDARD_CSV_HEADERS[:14]])
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix[:, :8].min(initial=0.0) >= 0.0 and matrix[:, :8].max(initial=0.0) > 64.0:
        matrix[:, :8] -= 128.0
    return matrix


def _build_predictor(checkpoint_path: str | Path, model_config, device_target: str):
    if ms is None or Tensor is None or context is None:
        raise RuntimeError("MindSpore is not available")
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    model = load_event_model_from_checkpoint(checkpoint_path, model_config)

    def _predict(emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
        logits = model(Tensor(emg_feature[np.newaxis, ...], ms.float32), Tensor(imu_feature[np.newaxis, ...], ms.float32)).asnumpy()[0]
        exp_logits = np.exp(logits - np.max(logits))
        return (exp_logits / np.sum(exp_logits)).astype(np.float32)

    return _predict


def _rows_from_frame(parsed: dict) -> np.ndarray:
    acc = np.asarray(parsed["acc"], dtype=np.float32)
    gyro = np.asarray(parsed["gyro"], dtype=np.float32)
    imu_row = np.concatenate([acc, gyro], axis=0)
    rows = []
    for pack in parsed.get("emg") or []:
        emg = np.asarray(pack[:8], dtype=np.float32)
        if emg.min(initial=0.0) >= 0.0 and emg.max(initial=0.0) > 64.0:
            emg = emg - 128.0
        rows.append(np.concatenate([emg, imu_row], axis=0))
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    runtime_cfg = load_event_runtime_config(args.config)
    if args.checkpoint:
        runtime_cfg.checkpoint_path = args.checkpoint
    if args.port:
        runtime_cfg.hardware.sensor_port = args.port
    if args.device:
        runtime_cfg.device.target = args.device
    if args.standalone:
        runtime_cfg.hardware.actuator_mode = "standalone"

    model_cfg, _, _, _ = load_event_training_config(runtime_cfg.training_config)
    predictor = _build_predictor(runtime_cfg.checkpoint_path, model_cfg, runtime_cfg.device.target)
    actuator = create_actuator(runtime_cfg.hardware)
    if hasattr(actuator, "connect"):
        actuator.connect()
    controller = EventOnsetController(
        data_config=runtime_cfg.data,
        inference_config=runtime_cfg.inference,
        runtime_config=runtime_cfg.runtime,
        predict_proba=predictor,
        actuator=actuator,
    )

    try:
        if args.source_csv:
            matrix = _load_standardized_matrix(args.source_csv)
            for step in controller.ingest_rows(matrix):
                if step.decision.changed:
                    logging.getLogger("event_runtime").info(
                        "state=%s confidence=%.3f energy=%.3f now_ms=%.1f",
                        step.decision.state.name,
                        step.confidence,
                        step.energy,
                        step.now_ms,
                    )
            return

        from scripts.emg_armband import Device

        device = Device(port=runtime_cfg.hardware.sensor_port or "COM4", baudrate=runtime_cfg.hardware.sensor_baudrate, timeout=0.5)
        device.connect()
        start = time.monotonic()
        try:
            while True:
                if args.duration_sec > 0 and (time.monotonic() - start) >= args.duration_sec:
                    break
                frames = device.read_frames()
                if not frames:
                    time.sleep(float(runtime_cfg.runtime.poll_interval_ms) / 1000.0)
                    continue
                for parsed in frames:
                    rows = _rows_from_frame(parsed)
                    for step in controller.ingest_rows(rows):
                        if step.decision.changed:
                            logging.getLogger("event_runtime").info(
                                "state=%s confidence=%.3f energy=%.3f now_ms=%.1f",
                                step.decision.state.name,
                                step.confidence,
                                step.energy,
                                step.now_ms,
                            )
        finally:
            device.disconnect()
    finally:
        if hasattr(actuator, "disconnect"):
            actuator.disconnect()


if __name__ == "__main__":
    main()
