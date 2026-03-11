"""Collect short event-onset clips for the 3-state experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.manifest import upsert_event_manifest
from scripts.collection_utils import (
    build_relative_recording_path,
    build_timestamp,
    ensure_unique_path,
    frame_to_standard_rows,
    validate_metadata,
    write_standard_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect event-onset clips for RELAX/FIST/PINCH")
    parser.add_argument("--data_dir", default="../data_event_onset")
    parser.add_argument("--start_state", required=True, choices=["RELAX", "FIST", "PINCH"])
    parser.add_argument("--target_state", required=True, choices=["RELAX", "FIST", "PINCH"])
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--wearing_state", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--countdown_sec", type=int, default=3)
    parser.add_argument("--rest_seconds", type=float, default=1.0)
    parser.add_argument("--training_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument(
        "--source_csv",
        action="append",
        default=None,
        help="Optional standardized CSV(s) for local testing. Can be passed multiple times.",
    )
    return parser


def _print_countdown(seconds: int, *, recording_index: int, total_count: int, start_state: str, target_state: str) -> None:
    if seconds <= 0:
        return
    prompt = "stay relaxed" if start_state == target_state == "RELAX" else f"start at {start_state}, then snap to {target_state}"
    for remaining in range(int(seconds), 0, -1):
        print(f"[clip {recording_index}/{total_count}] {prompt} in {remaining}s")
        time.sleep(1.0)


def _collect_rows_from_device(*, port: str, baudrate: int, timeout: float, duration_sec: float) -> np.ndarray:
    from scripts.emg_armband import Device

    device = Device(port=port, baudrate=baudrate, timeout=timeout)
    device.connect()
    rows: list[list[float]] = []
    try:
        deadline = time.monotonic() + float(duration_sec)
        while time.monotonic() < deadline:
            frames = device.read_frames()
            if not frames:
                time.sleep(0.01)
                continue
            for frame in frames:
                rows.extend(frame_to_standard_rows(frame))
    finally:
        device.disconnect()

    if not rows:
        raise RuntimeError("No device data captured. Check port, baudrate, or armband connection.")
    return np.asarray(rows, dtype=np.float32)


def _read_standard_csv(path: str | Path) -> np.ndarray:
    from scripts.collection_utils import STANDARD_CSV_HEADERS

    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {path}")
        for row in reader:
            rows.append([float(row[field]) for field in STANDARD_CSV_HEADERS])
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix[:, :8].min(initial=0.0) >= 0.0 and matrix[:, :8].max(initial=0.0) > 64.0:
        matrix[:, :8] -= 128.0
    return matrix


def _read_written_matrix(path: str | Path) -> np.ndarray:
    return _read_standard_csv(path)


def _acquire_clip(
    *,
    recording_index: int,
    source_csvs: Sequence[str | Path] | None,
    port: str,
    baudrate: int,
    timeout: float,
    duration_sec: float,
) -> tuple[np.ndarray, str]:
    if source_csvs:
        source_path = Path(source_csvs[(recording_index - 1) % len(source_csvs)])
        return _read_standard_csv(source_path), f"simulated_csv:{source_path.name}"
    return _collect_rows_from_device(port=port, baudrate=baudrate, timeout=timeout, duration_sec=duration_sec), "armband_serial"


def run_collection_batch(
    *,
    data_dir: str | Path,
    start_state: str,
    target_state: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
    count: int = 1,
    countdown_sec: int = 3,
    rest_seconds: float = 1.0,
    training_config: str | Path = "configs/training_event_onset.yaml",
    manifest_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    port: str = "COM4",
    baudrate: int = 115200,
    timeout: float = 0.5,
    source_csvs: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    if target_state == "RELAX" and start_state != "RELAX":
        raise ValueError("Pure RELAX clips must use start_state=RELAX and target_state=RELAX")

    metadata = validate_metadata(
        gesture=target_state,
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        wearing_state=wearing_state,
    )
    _, data_cfg, _, _ = load_event_training_config(training_config)
    clip_duration_ms = 2000 if start_state == target_state == "RELAX" else int(data_cfg.clip_duration_ms)
    duration_sec = float(clip_duration_ms) / 1000.0

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path) if manifest_path else data_root / "recordings_manifest.csv"
    report_root = Path(report_dir) if report_dir else data_root / "collection_reports"
    report_root.mkdir(parents=True, exist_ok=True)

    quality_loader = EventClipDatasetLoader(data_root, data_cfg, recordings_manifest_path=manifest_file)
    records: list[dict[str, Any]] = []

    for recording_index in range(1, int(count) + 1):
        _print_countdown(
            countdown_sec,
            recording_index=recording_index,
            total_count=int(count),
            start_state=start_state,
            target_state=target_state,
        )
        matrix, source_origin = _acquire_clip(
            recording_index=recording_index,
            source_csvs=source_csvs,
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            duration_sec=duration_sec,
        )
        timestamp = build_timestamp()
        relative_path = build_relative_recording_path(metadata, timestamp=timestamp, recording_index=recording_index)
        destination = ensure_unique_path(data_root / relative_path)
        write_standard_csv(destination, matrix)

        clip_meta = {
            "relative_path": relative_path.as_posix(),
            "target_state": target_state,
            "start_state": start_state,
            "capture_mode": "event_onset",
        }
        normalized_matrix = _read_written_matrix(destination)
        selected_windows = quality_loader._build_event_windows(normalized_matrix, clip_meta)  # internal use for collection QC
        quality_status = "pass" if selected_windows else "retake_recommended"
        quality_reasons = [] if selected_windows else ["no_valid_event_window"]

        manifest_row = {
            "relative_path": relative_path.as_posix(),
            "gesture": target_state,
            "capture_mode": "event_onset",
            "start_state": start_state,
            "target_state": target_state,
            "user_id": metadata.user_id,
            "session_id": metadata.session_id,
            "device_id": metadata.device_id,
            "timestamp": timestamp,
            "wearing_state": metadata.wearing_state,
            "recording_id": destination.stem,
            "sample_count": int(matrix.shape[0]),
            "clip_duration_ms": clip_duration_ms,
            "pre_roll_ms": 0 if target_state == "RELAX" else int(data_cfg.pre_roll_ms),
            "device_sampling_rate_hz": int(data_cfg.device_sampling_rate_hz),
            "imu_sampling_rate_hz": int(data_cfg.imu_sampling_rate_hz),
            "quality_status": quality_status,
            "quality_reasons": "|".join(quality_reasons),
            "source_origin": source_origin,
        }
        upsert_event_manifest(manifest_file, manifest_row)

        record = {
            "absolute_path": str(destination.resolve()),
            "relative_path": relative_path.as_posix(),
            "start_state": start_state,
            "target_state": target_state,
            "timestamp": timestamp,
            "source_origin": source_origin,
            "row_count": int(matrix.shape[0]),
            "selected_windows": len(selected_windows),
            "quality_status": quality_status,
            "quality_reasons": quality_reasons,
        }
        records.append(record)
        print(f"[clip {recording_index}/{count}] {destination.name}: selected_windows={len(selected_windows)} status={quality_status}")
        if recording_index < int(count) and float(rest_seconds) > 0:
            time.sleep(float(rest_seconds))

    payload = {
        "mode": "event_onset_collect",
        "training_config": str(training_config),
        "manifest_path": str(manifest_file.resolve()),
        "data_dir": str(data_root.resolve()),
        "records": records,
    }
    report_path = report_root / f"{build_timestamp()}_{target_state.lower()}_{metadata.session_id}_event_collect.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_file.resolve()),
        "report_path": str(report_path.resolve()),
        "records": records,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_collection_batch(
        data_dir=args.data_dir,
        start_state=args.start_state,
        target_state=args.target_state,
        user_id=args.user_id,
        session_id=args.session_id,
        device_id=args.device_id,
        wearing_state=args.wearing_state,
        count=args.count,
        countdown_sec=args.countdown_sec,
        rest_seconds=args.rest_seconds,
        training_config=args.training_config,
        manifest_path=args.manifest_path,
        report_dir=args.report_dir,
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        source_csvs=args.source_csv,
    )
    print(f"Saved report: {result['report_path']}")
    print(f"Updated manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
