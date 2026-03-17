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
from event_onset.algo import EventAlgoPredictor
from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.runtime import EventOnsetController
from runtime.hardware.factory import create_actuator
from scripts.collection_utils import STANDARD_CSV_HEADERS
from shared.event_labels import normalize_event_label_input, public_event_labels
from shared.label_modes import get_label_mode_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run event-onset runtime controller")
    parser.add_argument(
        "--recognizer_backend",
        default="model",
        choices=["model", "algo"],
        help="Choose recognizer backend once at startup.",
    )
    parser.add_argument("--config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--backend", default="lite", choices=["lite", "ckpt"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_path", default=None, help="MindIR model path for --backend lite")
    parser.add_argument("--model_metadata", default=None, help="Model metadata json path for --backend lite")
    parser.add_argument(
        "--algo_model_path",
        default=None,
        help="Algorithm model json path when --recognizer_backend=algo.",
    )
    parser.add_argument("--actuation_mapping", default=None, help="Class-to-actuator mapping YAML path.")
    parser.add_argument(
        "--target_db5_keys",
        default=None,
        help=(
            "Comma-separated action keys to override runtime config, "
            "e.g. TENSE_OPEN,V_SIGN,OK_SIGN,THUMB_UP,WRIST_CW,WRIST_CCW."
        ),
    )
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


def _validate_runtime_class_contract(
    *,
    recognizer_backend: str,
    model_backend: str,
    expected_class_names: list[str],
    mapping_by_name: dict[str, str],
    model_num_classes: int | None,
    metadata_class_names: list[str] | None,
    recognizer_class_names: list[str] | None = None,
) -> None:
    normalized_expected = [normalize_event_label_input(name) for name in expected_class_names]

    mapping_keys = sorted(normalize_event_label_input(key) for key in mapping_by_name.keys())
    if mapping_keys != sorted(normalized_expected):
        raise ValueError(
            f"Actuation mapping keys mismatch expected classes. mapping_keys={public_event_labels(mapping_keys)}, "
            f"expected={public_event_labels(sorted(normalized_expected))}"
        )

    if recognizer_backend == "algo":
        names = [normalize_event_label_input(item) for item in (recognizer_class_names or [])]
        if not names:
            raise ValueError("Algorithm backend must provide non-empty class_names.")
        if names != normalized_expected:
            raise ValueError(
                "Runtime class order mismatch between config and algo model: "
                f"config={public_event_labels(normalized_expected)}, algo={public_event_labels(names)}"
            )
        return

    if model_num_classes is None:
        raise ValueError("model_num_classes is required for recognizer_backend=model.")
    if int(model_num_classes) != len(normalized_expected):
        raise ValueError(
            f"model.num_classes={model_num_classes} mismatches expected labels={len(normalized_expected)} "
            f"({normalized_expected})"
        )

    if metadata_class_names is None:
        if model_backend == "lite":
            raise ValueError("Lite backend requires model metadata with class_names for strict runtime validation.")
        return

    if not metadata_class_names:
        if model_backend == "lite":
            raise ValueError("Lite backend metadata must include non-empty class_names.")
        return
    normalized_metadata = [normalize_event_label_input(name) for name in metadata_class_names]
    if normalized_metadata != normalized_expected:
        raise ValueError(
            "Runtime class order mismatch between config and model metadata: "
            f"config={public_event_labels(normalized_expected)}, metadata={public_event_labels(normalized_metadata)}"
        )


def _validate_release_contract(
    *,
    release_mode: str,
    class_names: list[str],
    mapping_by_name: dict[str, str],
) -> None:
    mode = str(release_mode).strip().lower()
    if mode != "command_only":
        return
    normalized = [str(name).strip().upper() for name in class_names]
    if "TENSE_OPEN" not in normalized:
        raise ValueError(
            "release_mode=command_only requires class TENSE_OPEN in runtime label set."
        )
    mapped = normalize_event_label_input(mapping_by_name.get("TENSE_OPEN", ""))
    if mapped != "RELAX":
        raise ValueError(
            "release_mode=command_only requires mapping TENSE_OPEN -> CONTINUE."
        )


def _ensure_file_exists(path: str | Path, *, desc: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{desc} not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{desc} is not a file: {resolved}")
    return resolved


def _validate_startup_artifacts(
    *,
    recognizer_backend: str,
    model_backend: str,
    checkpoint_path: str | Path,
    model_path: str | Path,
    model_metadata_path: str | Path,
    algo_model_path: str | Path | None,
) -> dict[str, str]:
    if recognizer_backend == "algo":
        if not str(algo_model_path or "").strip():
            raise ValueError(
                "--algo_model_path is required when --recognizer_backend=algo."
            )
        algo_path = _ensure_file_exists(str(algo_model_path), desc="Algorithm model")
        return {"algo_model_path": str(algo_path)}

    if model_backend == "ckpt":
        ckpt = _ensure_file_exists(checkpoint_path, desc="Model checkpoint")
        return {"checkpoint_path": str(ckpt)}
    if model_backend == "lite":
        model = _ensure_file_exists(model_path, desc="MindIR model")
        metadata = _ensure_file_exists(model_metadata_path, desc="Model metadata")
        return {"model_path": str(model), "model_metadata_path": str(metadata)}
    raise ValueError(f"Unsupported model backend: {model_backend}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_runtime")
    args = build_parser().parse_args()

    runtime_cfg = load_event_runtime_config(args.config)
    recognizer_backend = str(args.recognizer_backend).strip().lower()
    model_backend = str(args.backend).strip().lower()
    if args.checkpoint:
        runtime_cfg.checkpoint_path = args.checkpoint
    if args.model_path:
        runtime_cfg.model_path = args.model_path
    if args.model_metadata:
        runtime_cfg.model_metadata_path = args.model_metadata
    if args.actuation_mapping:
        runtime_cfg.actuation_mapping_path = args.actuation_mapping
    if args.target_db5_keys:
        keys = [item.strip().upper() for item in str(args.target_db5_keys).split(",") if item.strip()]
        if not keys:
            raise ValueError("--target_db5_keys provided but no valid keys parsed.")
        runtime_cfg.data.target_db5_keys = keys
    if args.port:
        runtime_cfg.hardware.sensor_port = args.port
    if args.device:
        runtime_cfg.device.target = args.device
    if args.standalone:
        runtime_cfg.hardware.actuator_mode = "standalone"

    model_cfg, _, _, _ = load_event_training_config(runtime_cfg.training_config)
    label_spec = get_label_mode_spec(runtime_cfg.data.label_mode, runtime_cfg.data.target_db5_keys)
    model_cfg.num_classes = int(len(label_spec.class_names))
    label_to_state, mapping_by_name = load_and_validate_actuation_map(
        runtime_cfg.actuation_mapping_path,
        class_names=label_spec.class_names,
    )
    startup_artifacts = _validate_startup_artifacts(
        recognizer_backend=recognizer_backend,
        model_backend=model_backend,
        checkpoint_path=runtime_cfg.checkpoint_path,
        model_path=runtime_cfg.model_path,
        model_metadata_path=runtime_cfg.model_metadata_path,
        algo_model_path=args.algo_model_path,
    )

    predictor = None
    predict_proba = None
    metadata_class_names: list[str] | None = None
    recognizer_class_names: list[str] | None = None
    if recognizer_backend == "model":
        predictor = EventPredictor(
            backend=model_backend,
            model_config=model_cfg,
            device_target=runtime_cfg.device.target,
            checkpoint_path=runtime_cfg.checkpoint_path,
            model_path=runtime_cfg.model_path,
            model_metadata_path=runtime_cfg.model_metadata_path,
        )
        predict_proba = predictor.predict_proba
        if predictor.metadata is not None and predictor.metadata.class_names:
            metadata_class_names = [str(name).strip().upper() for name in predictor.metadata.class_names]
    else:
        algo_predictor = EventAlgoPredictor(model_path=str(startup_artifacts["algo_model_path"]))
        predict_proba = algo_predictor.predict_proba
        recognizer_class_names = [str(name).strip().upper() for name in algo_predictor.class_names]

    _validate_runtime_class_contract(
        recognizer_backend=recognizer_backend,
        model_backend=model_backend,
        expected_class_names=list(label_spec.class_names),
        model_num_classes=int(model_cfg.num_classes) if recognizer_backend == "model" else None,
        mapping_by_name=mapping_by_name,
        metadata_class_names=metadata_class_names,
        recognizer_class_names=recognizer_class_names,
    )
    _validate_release_contract(
        release_mode=runtime_cfg.runtime.release_mode,
        class_names=list(label_spec.class_names),
        mapping_by_name=mapping_by_name,
    )

    actuator = create_actuator(runtime_cfg.hardware)
    if hasattr(actuator, "connect"):
        actuator.connect()

    logger.info(
        "Event runtime started: recognizer_backend=%s model_backend=%s device=%s actuation_mapping=%s",
        recognizer_backend,
        model_backend if recognizer_backend == "model" else "n/a",
        runtime_cfg.device.target,
        runtime_cfg.actuation_mapping_path,
    )
    if recognizer_backend == "model":
        logger.info(
            "Model artifacts: checkpoint=%s model=%s metadata=%s",
            runtime_cfg.checkpoint_path,
            runtime_cfg.model_path,
            runtime_cfg.model_metadata_path,
        )
    else:
        logger.info("Algo artifact: algo_model_path=%s", startup_artifacts["algo_model_path"])
    logger.info("Class order: %s", list(label_spec.class_names))
    logger.info("Class mapping: %s", mapping_by_name)
    logger.info("Release mode: %s", runtime_cfg.runtime.release_mode)
    if recognizer_backend == "model" and model_backend == "ckpt":
        logger.warning("CKPT backend is intended for debugging only. Use --backend lite for production deployment.")

    controller = EventOnsetController(
        data_config=runtime_cfg.data,
        inference_config=runtime_cfg.inference,
        runtime_config=runtime_cfg.runtime,
        class_names=label_spec.class_names,
        label_to_state=label_to_state,
        predict_proba=predict_proba,
        actuator=actuator,
    )

    try:
        if args.source_csv:
            matrix = _load_standardized_matrix(args.source_csv)
            for step in controller.ingest_rows(matrix):
                if step.decision.changed:
                    logger.info(
                        "state=%s class=%s confidence=%.3f energy=%.3f now_ms=%.1f",
                        step.decision.state.name,
                        step.decision.emitted_class_name,
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
                                "state=%s class=%s confidence=%.3f energy=%.3f now_ms=%.1f",
                                step.decision.state.name,
                                step.decision.emitted_class_name,
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
