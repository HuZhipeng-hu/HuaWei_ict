"""Run event-onset runtime controller (MindIR Lite by default, CKPT for debug)."""

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
from event_onset.inference import EventPredictor
from event_onset.runtime import EventOnsetController
from runtime.hardware.factory import create_actuator
from scripts.collection_utils import STANDARD_CSV_HEADERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run event-onset runtime controller")
    parser.add_argument("--config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--backend", default="lite", choices=["lite", "ckpt"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_path", default=None, help="MindIR model path for --backend lite")
    parser.add_argument("--model_metadata", default=None, help="Model metadata json path for --backend lite")
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
    logger = logging.getLogger("event_runtime")
    args = build_parser().parse_args()

    runtime_cfg = load_event_runtime_config(args.config)
    if args.checkpoint:
        runtime_cfg.checkpoint_path = args.checkpoint
    if args.model_path:
        runtime_cfg.model_path = args.model_path
    if args.model_metadata:
        runtime_cfg.model_metadata_path = args.model_metadata
    if args.port:
        runtime_cfg.hardware.sensor_port = args.port
    if args.device:
        runtime_cfg.device.target = args.device
    if args.standalone:
        runtime_cfg.hardware.actuator_mode = "standalone"

    model_cfg, _, _, _ = load_event_training_config(runtime_cfg.training_config)
    predictor = EventPredictor(
        backend=args.backend,
        model_config=model_cfg,
        device_target=runtime_cfg.device.target,
        checkpoint_path=runtime_cfg.checkpoint_path,
        model_path=runtime_cfg.model_path,
        model_metadata_path=runtime_cfg.model_metadata_path,
    )

    actuator = create_actuator(runtime_cfg.hardware)
    if hasattr(actuator, "connect"):
        actuator.connect()

    logger.info(
        "Event runtime started: backend=%s device=%s checkpoint=%s model=%s",
        args.backend,
        runtime_cfg.device.target,
        runtime_cfg.checkpoint_path,
        runtime_cfg.model_path,
    )
    if args.backend == "ckpt":
        logger.warning("CKPT backend is intended for debugging only. Use --backend lite for production deployment.")

    controller = EventOnsetController(
        data_config=runtime_cfg.data,
        inference_config=runtime_cfg.inference,
        runtime_config=runtime_cfg.runtime,
        predict_proba=predictor.predict_proba,
        actuator=actuator,
    )

    try:
        if args.source_csv:
            matrix = _load_standardized_matrix(args.source_csv)
            for step in controller.ingest_rows(matrix):
                if step.decision.changed:
                    logger.info(
                        "state=%s confidence=%.3f energy=%.3f now_ms=%.1f",
                        step.decision.state.name,
                        step.confidence,
                        step.energy,
                        step.now_ms,
                    )
            return

        from scripts.emg_armband import Device

        device = Device(
            port=runtime_cfg.hardware.sensor_port or "COM4",
            baudrate=runtime_cfg.hardware.sensor_baudrate,
            timeout=0.5,
        )
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
                            logger.info(
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
