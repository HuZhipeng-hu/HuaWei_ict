"""Collect standardized EMG recordings via serial device or source CSV replay."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.collection_utils import (
    build_manifest_row,
    build_quality_console_line,
    build_relative_recording_path,
    build_timestamp,
    ensure_unique_path,
    evaluate_recording_quality,
    frame_to_standard_rows,
    load_collection_protocol,
    normalize_relative_path,
    read_source_csv,
    resolve_manifest_path,
    resolve_report_dir,
    upsert_recordings_manifest,
    validate_metadata,
    write_json_report,
    write_standard_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a batch of standardized EMG CSV files")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gesture", required=True)
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--wearing_state", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--countdown_sec", type=int, default=3)
    parser.add_argument("--record_seconds", type=float, default=3.0)
    parser.add_argument("--rest_seconds", type=float, default=1.0)
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument(
        "--source_csv",
        action="append",
        default=None,
        help="Optional replay CSV for local testing. Can be passed multiple times.",
    )
    return parser


def _print_countdown(seconds: int, *, recording_index: int, total_count: int) -> None:
    if seconds <= 0:
        return
    for remaining in range(int(seconds), 0, -1):
        print(f"[record {recording_index}/{total_count}] start in {remaining}s")
        time.sleep(1.0)


def _collect_rows_from_device(
    *,
    port: str,
    baudrate: int,
    timeout: float,
    record_seconds: float,
) -> np.ndarray:
    from scripts.emg_armband import Device

    device = Device(port=port, baudrate=baudrate, timeout=timeout)
    device.connect()
    rows: list[list[float]] = []
    try:
        deadline = time.monotonic() + float(record_seconds)
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


def _acquire_recording_matrix(
    *,
    recording_index: int,
    source_csvs: Sequence[str | Path] | None,
    port: str,
    baudrate: int,
    timeout: float,
    record_seconds: float,
) -> tuple[np.ndarray, str]:
    if source_csvs:
        source_path = Path(source_csvs[(recording_index - 1) % len(source_csvs)])
        return read_source_csv(source_path), f"simulated_csv:{source_path.name}"
    return _collect_rows_from_device(
        port=port,
        baudrate=baudrate,
        timeout=timeout,
        record_seconds=record_seconds,
    ), "armband_serial"


def run_collection_batch(
    *,
    data_dir: str | Path,
    gesture: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
    count: int = 1,
    countdown_sec: int = 3,
    record_seconds: float = 3.0,
    rest_seconds: float = 1.0,
    training_config: str | Path = "configs/training.yaml",
    manifest_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    port: str = "COM4",
    baudrate: int = 115200,
    timeout: float = 0.5,
    source_csvs: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    metadata = validate_metadata(
        gesture=gesture,
        user_id=user_id,
        session_id=session_id,
        device_id=device_id,
        wearing_state=wearing_state,
    )
    preprocess_cfg, quality_filter = load_collection_protocol(training_config)

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_file = resolve_manifest_path(data_root, manifest_path)
    report_root = resolve_report_dir(data_root, report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    print("Training currently consumes all 8 EMG channels under the 16x24x6 dual-branch protocol.")

    recordings: list[dict[str, Any]] = []
    for recording_index in range(1, int(count) + 1):
        _print_countdown(countdown_sec, recording_index=recording_index, total_count=int(count))
        matrix, source_origin = _acquire_recording_matrix(
            recording_index=recording_index,
            source_csvs=source_csvs,
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            record_seconds=record_seconds,
        )

        timestamp = build_timestamp()
        relative_path = build_relative_recording_path(
            metadata,
            timestamp=timestamp,
            recording_index=recording_index,
        )
        destination = ensure_unique_path(data_root / relative_path)
        write_standard_csv(destination, matrix)

        quality_report = evaluate_recording_quality(
            matrix,
            preprocess_config=preprocess_cfg,
            quality_filter=quality_filter,
        )
        rel_for_manifest = normalize_relative_path(destination.relative_to(data_root))
        manifest_row = build_manifest_row(
            relative_path=rel_for_manifest,
            metadata=metadata,
            timestamp=timestamp,
            sample_count=int(matrix.shape[0]),
            quality_report=quality_report,
            source_origin=source_origin,
        )
        upsert_recordings_manifest(manifest_file, manifest_row)

        record = {
            "absolute_path": str(destination.resolve()),
            "relative_path": rel_for_manifest,
            "gesture": metadata.gesture,
            "timestamp": timestamp,
            "source_origin": source_origin,
            **quality_report,
        }
        recordings.append(record)
        print(f"[record {recording_index}/{count}] {destination.name}: {build_quality_console_line(record)}")

        if recording_index < int(count) and float(rest_seconds) > 0:
            time.sleep(float(rest_seconds))

    session_payload = {
        "mode": "collect",
        "training_config": str(training_config),
        "manifest_path": str(Path(manifest_file).resolve()),
        "data_dir": str(data_root.resolve()),
        "records": recordings,
    }
    report_path = report_root / f"{build_timestamp()}_{metadata.gesture.lower()}_{metadata.session_id}_collect.json"
    write_json_report(report_path, session_payload)

    return {
        "manifest_path": str(Path(manifest_file).resolve()),
        "report_path": str(Path(report_path).resolve()),
        "records": recordings,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_collection_batch(
        data_dir=args.data_dir,
        gesture=args.gesture,
        user_id=args.user_id,
        session_id=args.session_id,
        device_id=args.device_id,
        wearing_state=args.wearing_state,
        count=args.count,
        countdown_sec=args.countdown_sec,
        record_seconds=args.record_seconds,
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
