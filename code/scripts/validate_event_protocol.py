"""Validate effective EMG/IMU rates from raw armband frames."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate effective EMG/IMU protocol rates from raw timestamps")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--duration_sec", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    return parser


def estimate_rates_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    frame_count = len(frames)
    emg_pack_count = sum(len(frame.get("emg") or []) for frame in frames)
    timestamps = [int(frame["timestamp"]) for frame in frames if "timestamp" in frame]
    if len(timestamps) < 2:
        return {
            "frame_count": frame_count,
            "emg_pack_count": emg_pack_count,
            "timestamp_frame_rate_hz": None,
            "timestamp_emg_rate_hz": None,
            "timestamp_median_delta": None,
        }
    deltas = [curr - prev for prev, curr in zip(timestamps, timestamps[1:]) if curr > prev]
    if not deltas:
        return {
            "frame_count": frame_count,
            "emg_pack_count": emg_pack_count,
            "timestamp_frame_rate_hz": None,
            "timestamp_emg_rate_hz": None,
            "timestamp_median_delta": None,
        }
    median_delta = float(statistics.median(deltas))
    frame_rate = 1000.0 / median_delta
    avg_pack_per_frame = float(emg_pack_count) / float(frame_count) if frame_count else 0.0
    return {
        "frame_count": frame_count,
        "emg_pack_count": emg_pack_count,
        "timestamp_frame_rate_hz": round(frame_rate, 3),
        "timestamp_emg_rate_hz": round(frame_rate * avg_pack_per_frame, 3),
        "timestamp_median_delta": median_delta,
    }


def collect_protocol_sample(port: str, baudrate: int, timeout: float, duration_sec: float) -> dict[str, Any]:
    from scripts.emg_armband import Device

    device = Device(port=port, baudrate=baudrate, timeout=timeout)
    frames: list[dict[str, Any]] = []
    device.connect()
    started = time.perf_counter()
    try:
        while (time.perf_counter() - started) < float(duration_sec):
            frames.extend(device.read_frames())
            time.sleep(0.01)
    finally:
        wall_elapsed = time.perf_counter() - started
        stats = dict(device.stats)
        device.disconnect()

    estimated = estimate_rates_from_frames(frames)
    estimated.update(
        {
            "wall_clock_sec": round(wall_elapsed, 3),
            "wall_frame_rate_hz": round(len(frames) / wall_elapsed, 3) if wall_elapsed > 0 else None,
            "wall_emg_rate_hz": round(sum(len(frame.get("emg") or []) for frame in frames) / wall_elapsed, 3) if wall_elapsed > 0 else None,
            "sync_errors": int(stats.get("sync_errors", 0)),
            "frames_failed": int(stats.get("frames_failed", 0)),
            "frames_parsed": int(stats.get("frames_parsed", 0)),
            "port": port,
            "baudrate": baudrate,
        }
    )
    return estimated


def main() -> None:
    args = build_parser().parse_args()
    report = collect_protocol_sample(args.port, args.baudrate, args.timeout, args.duration_sec)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Saved report: {Path(args.output).resolve()}")
    print(text)


if __name__ == "__main__":
    main()
