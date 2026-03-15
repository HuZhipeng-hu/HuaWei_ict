"""Collect live armband data into event-onset CSV + optional manifest row."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.manifest import upsert_event_manifest

STANDARD_CSV_HEADERS = [
    "emg1",
    "emg2",
    "emg3",
    "emg4",
    "emg5",
    "emg6",
    "emg7",
    "emg8",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "angle_pitch",
    "angle_roll",
    "angle_yaw",
]


def _resolve_duration_sec(target_state: str, duration_sec: float | None) -> float:
    if duration_sec is not None:
        value = float(duration_sec)
        if value <= 0:
            raise ValueError("duration_sec must be > 0.")
        return value
    return 4.0 if target_state == "RELAX" else 3.0


def _mean_abs_emg(rows: list[list[float]], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    segment = rows[start:end]
    if not segment:
        return 0.0
    emg_values = [float(v) for row in segment for v in row[:8]]
    if not emg_values:
        return 0.0
    min_v = min(emg_values)
    max_v = max(emg_values)
    offset = 128.0 if (min_v >= 0.0 and max_v > 64.0) else 0.0
    total = 0.0
    count = 0
    for row in segment:
        for value in row[:8]:
            total += abs(float(value) - offset)
            count += 1
    return (total / float(count)) if count else 0.0


def _estimate_activation_quality(
    *,
    rows: list[list[float]],
    elapsed_sec: float,
    target_state: str,
    pre_roll_ms: int,
    action_window_sec: float,
) -> dict[str, Any]:
    estimated_rate_hz = (float(len(rows)) / float(elapsed_sec)) if elapsed_sec > 0 else 0.0
    quality_status = "raw"
    quality_reasons: list[str] = []
    activation_ratio: float | None = None
    baseline_mean_abs = None
    action_mean_abs = None

    if target_state != "RELAX":
        pre_roll_sec = max(0.0, float(pre_roll_ms) / 1000.0)
        pre_end = min(len(rows), max(1, int(round(estimated_rate_hz * pre_roll_sec))))
        action_start = pre_end
        action_count = max(1, int(round(estimated_rate_hz * max(0.2, float(action_window_sec)))))
        action_end = min(len(rows), action_start + action_count)

        baseline_mean_abs = _mean_abs_emg(rows, 0, pre_end)
        action_mean_abs = _mean_abs_emg(rows, action_start, action_end)
        if baseline_mean_abs > 1e-6:
            activation_ratio = action_mean_abs / baseline_mean_abs

        if action_end <= action_start:
            quality_status = "warn"
            quality_reasons.append("insufficient_action_window")
        elif activation_ratio is None:
            quality_status = "warn"
            quality_reasons.append("invalid_activation_ratio")
        elif activation_ratio < 1.08:
            quality_status = "retake_recommended"
            quality_reasons.append("weak_activation")
        elif activation_ratio < 1.20:
            quality_status = "warn"
            quality_reasons.append("low_activation")
        else:
            quality_status = "pass"
    else:
        quality_status = "pass"

    return {
        "estimated_emg_rate_hz": round(estimated_rate_hz, 3) if estimated_rate_hz > 0 else None,
        "baseline_mean_abs": round(float(baseline_mean_abs), 4) if baseline_mean_abs is not None else None,
        "action_mean_abs": round(float(action_mean_abs), 4) if action_mean_abs is not None else None,
        "activation_ratio": round(float(activation_ratio), 4) if activation_ratio is not None else None,
        "quality_status": quality_status,
        "quality_reasons": quality_reasons,
    }


def _sanitize_token(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "unknown"


def _normalize_state(value: str) -> str:
    state = str(value or "").strip().upper()
    if not state:
        raise ValueError("State cannot be empty.")
    return state


def _frame_to_rows(parsed: dict[str, Any]) -> list[list[float]]:
    acc = parsed.get("acc") or (0, 0, 0)
    gyro = parsed.get("gyro") or (0, 0, 0)
    angle = parsed.get("angle") or (0, 0, 0)
    emg_packs = parsed.get("emg") or []
    rows: list[list[float]] = []
    for pack in emg_packs:
        if len(pack) < 8:
            continue
        rows.append(
            [
                float(pack[0]),
                float(pack[1]),
                float(pack[2]),
                float(pack[3]),
                float(pack[4]),
                float(pack[5]),
                float(pack[6]),
                float(pack[7]),
                float(acc[0]),
                float(acc[1]),
                float(acc[2]),
                float(gyro[0]),
                float(gyro[1]),
                float(gyro[2]),
                float(angle[0]),
                float(angle[1]),
                float(angle[2]),
            ]
        )
    return rows


def _build_default_relative_path(
    *,
    target_state: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
    timestamp: str,
) -> Path:
    stem = (
        f"{timestamp}"
        f"__u-{_sanitize_token(user_id)}"
        f"__s-{_sanitize_token(session_id)}"
        f"__d-{_sanitize_token(device_id)}"
        f"__w-{_sanitize_token(wearing_state)}"
    )
    return Path(target_state) / f"{stem}.csv"


def _resolve_output_paths(
    *,
    data_dir: Path,
    output_relpath: str | None,
    target_state: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
    timestamp: str,
) -> tuple[Path, str]:
    if str(output_relpath or "").strip():
        rel = Path(str(output_relpath).replace("\\", "/"))
    else:
        rel = _build_default_relative_path(
            target_state=target_state,
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            wearing_state=wearing_state,
            timestamp=timestamp,
        )
    rel_text = rel.as_posix()
    abs_path = (data_dir / rel).resolve()
    return abs_path, rel_text


def _write_standard_csv(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STANDARD_CSV_HEADERS)
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect live armband data for event-onset training.")
    parser.add_argument("--data_dir", default="../data", help="Output data root.")
    parser.add_argument("--recordings_manifest", default="recordings_manifest.csv")
    parser.add_argument("--target_state", required=True, help="Target action label, e.g. V_SIGN / WRIST_CW.")
    parser.add_argument("--start_state", default="RELAX", help="Start state label.")
    parser.add_argument("--capture_mode", default="event_onset")
    parser.add_argument("--user_id", default="demo_user")
    parser.add_argument("--session_id", default="s1")
    parser.add_argument("--device_id", default="armband01")
    parser.add_argument("--wearing_state", default="normal")
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=None,
        help="Total clip length. Default: action=3s, RELAX=4s.",
    )
    parser.add_argument("--pre_roll_ms", type=int, default=400)
    parser.add_argument(
        "--action_window_sec",
        type=float,
        default=1.2,
        help="Expected strong activation window after pre-roll (action labels only).",
    )
    parser.add_argument("--device_sampling_rate_hz", type=int, default=500)
    parser.add_argument("--imu_sampling_rate_hz", type=int, default=50)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--poll_interval_ms", type=int, default=10)
    parser.add_argument("--output_relpath", default=None, help="Optional relative csv path under data_dir.")
    parser.add_argument("--no_manifest", action="store_true", help="Do not update recordings_manifest.csv.")
    parser.add_argument("--report_json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from scripts.emg_armband import Device
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "collect_event_data.py requires pyserial. Install it first: pip install pyserial"
        ) from exc

    data_dir = Path(args.data_dir).resolve()
    target_state = _normalize_state(args.target_state)
    start_state = _normalize_state(args.start_state)
    duration_sec = _resolve_duration_sec(target_state, args.duration_sec)
    capture_mode = str(args.capture_mode or "").strip() or "event_onset"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    output_csv, relative_path = _resolve_output_paths(
        data_dir=data_dir,
        output_relpath=args.output_relpath,
        target_state=target_state,
        user_id=args.user_id,
        session_id=args.session_id,
        device_id=args.device_id,
        wearing_state=args.wearing_state,
        timestamp=timestamp,
    )
    manifest_path = Path(args.recordings_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (data_dir / manifest_path).resolve()

    rows: list[list[float]] = []
    frame_count = 0
    pack_count = 0
    started = time.perf_counter()
    device = Device(port=args.port, baudrate=int(args.baudrate), timeout=float(args.timeout))
    pre_roll_sec = max(0.0, float(args.pre_roll_ms) / 1000.0)
    action_window_sec = max(0.2, float(args.action_window_sec))
    action_cue_sent = False
    release_cue_sent = False

    print(
        f"[COLLECT] start port={args.port} baudrate={int(args.baudrate)} "
        f"duration_sec={float(duration_sec):.2f} target_state={target_state}"
    )
    if target_state == "RELAX":
        print("[COLLECT] protocol: keep RELAX for the full clip.")
    else:
        print(
            f"[COLLECT] protocol: keep RELAX for {pre_roll_sec:.2f}s, "
            f"then activate {target_state} strongly for ~{action_window_sec:.2f}s, "
            "then relax naturally until clip end."
        )
    device.connect()
    try:
        while (time.perf_counter() - started) < float(duration_sec):
            elapsed = time.perf_counter() - started
            if target_state != "RELAX" and (not action_cue_sent) and elapsed >= pre_roll_sec:
                print(f"[COLLECT] cue: ACTION NOW -> {target_state}")
                action_cue_sent = True
            if (
                target_state != "RELAX"
                and action_cue_sent
                and (not release_cue_sent)
                and elapsed >= (pre_roll_sec + action_window_sec)
            ):
                print("[COLLECT] cue: RELEASE -> return to RELAX")
                release_cue_sent = True
            frames = device.read_frames()
            if not frames:
                time.sleep(float(args.poll_interval_ms) / 1000.0)
                continue
            for parsed in frames:
                frame_count += 1
                current_rows = _frame_to_rows(parsed)
                pack_count += len(current_rows)
                rows.extend(current_rows)
    finally:
        device.disconnect()
    elapsed_sec = time.perf_counter() - started

    if not rows:
        raise RuntimeError(
            "No rows captured from armband. Check port/baudrate/device mode and retry."
        )

    _write_standard_csv(output_csv, rows)
    quality_report = _estimate_activation_quality(
        rows=rows,
        elapsed_sec=elapsed_sec,
        target_state=target_state,
        pre_roll_ms=int(args.pre_roll_ms),
        action_window_sec=action_window_sec,
    )

    recording_id = Path(relative_path).stem
    manifest_updated = None
    if not bool(args.no_manifest):
        manifest_row = {
            "relative_path": relative_path,
            "gesture": target_state,
            "capture_mode": capture_mode,
            "start_state": start_state,
            "target_state": target_state,
            "user_id": _sanitize_token(args.user_id),
            "session_id": _sanitize_token(args.session_id),
            "device_id": _sanitize_token(args.device_id),
            "timestamp": timestamp,
            "wearing_state": _sanitize_token(args.wearing_state),
            "recording_id": recording_id,
            "sample_count": int(len(rows)),
            "clip_duration_ms": int(round(float(duration_sec) * 1000.0)),
            "pre_roll_ms": int(args.pre_roll_ms),
            "device_sampling_rate_hz": int(args.device_sampling_rate_hz),
            "imu_sampling_rate_hz": int(args.imu_sampling_rate_hz),
            "quality_status": str(quality_report.get("quality_status", "raw")),
            "quality_reasons": "|".join(quality_report.get("quality_reasons", [])),
            "source_origin": "armband_live_capture",
        }
        manifest_updated = str(upsert_event_manifest(manifest_path, manifest_row))

    report = {
        "status": "ok",
        "output_csv": str(output_csv),
        "relative_path": relative_path,
        "recordings_manifest": manifest_updated,
        "rows": int(len(rows)),
        "frames": int(frame_count),
        "emg_packs": int(pack_count),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "target_state": target_state,
        "start_state": start_state,
        "duration_sec": float(duration_sec),
        "pre_roll_sec": pre_roll_sec,
        "action_window_sec": action_window_sec,
    }
    report.update(quality_report)

    if str(args.report_json or "").strip():
        report_path = Path(args.report_json)
        if not report_path.is_absolute():
            report_path = (data_dir / report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_json"] = str(report_path)

    print(f"[COLLECT] output_csv={report['output_csv']}")
    if manifest_updated:
        print(f"[COLLECT] recordings_manifest={manifest_updated}")
    print(
        f"[COLLECT] rows={report['rows']} frames={report['frames']} "
        f"emg_packs={report['emg_packs']} elapsed_sec={report['elapsed_sec']}"
    )
    if target_state != "RELAX":
        ratio = report.get("activation_ratio")
        status = report.get("quality_status")
        reasons = ",".join(report.get("quality_reasons") or []) or "none"
        print(
            f"[COLLECT] quality status={status} activation_ratio={ratio} reasons={reasons}"
        )
        if status == "retake_recommended":
            print(
                "[COLLECT][WARN] Activation looks weak for this action clip; "
                "recommend retake with faster, stronger contraction near ACTION cue."
            )


if __name__ == "__main__":
    main()
