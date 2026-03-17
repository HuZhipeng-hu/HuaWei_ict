"""Collect continuous armband stream and auto-slice clean event clips."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.manifest import upsert_event_manifest
from shared.event_labels import normalize_event_label_input, public_event_label
from scripts.collection_utils import evaluate_recording_quality, load_collection_protocol, write_standard_csv


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


def _sanitize_token(value: Any) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    return cleaned or "unknown"


def _normalize_state(value: str) -> str:
    state = str(value or "").strip().upper()
    if not state:
        raise ValueError("State cannot be empty.")
    return public_event_label(state)


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


def _normalize_emg_domain(emg: np.ndarray) -> np.ndarray:
    normalized = np.asarray(emg, dtype=np.float32).copy()
    if normalized.size and float(np.min(normalized)) >= 0.0 and float(np.max(normalized)) > 64.0:
        normalized -= 128.0
    return normalized


def _moving_average(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return values.astype(np.float32, copy=False)
    kernel = np.ones((int(win),), dtype=np.float32) / float(win)
    return np.convolve(values.astype(np.float32, copy=False), kernel, mode="same")


def _find_active_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        if (not flag) and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(mask.shape[0])))
    return runs


def detect_onsets(
    emg_matrix: np.ndarray,
    *,
    sample_rate_hz: int,
    smooth_ms: int,
    q_low: float,
    q_high: float,
    threshold_alpha: float,
    min_active_sec: float,
    min_gap_sec: float,
) -> dict[str, Any]:
    emg = _normalize_emg_domain(np.asarray(emg_matrix, dtype=np.float32))
    if emg.ndim != 2 or emg.shape[1] < 1:
        raise ValueError("emg_matrix must be 2D with channel dimension.")
    envelope = np.mean(np.abs(emg[:, :8]), axis=1)
    smooth_win = max(1, int(round(float(sample_rate_hz) * float(smooth_ms) / 1000.0)))
    env_smooth = _moving_average(envelope, smooth_win)

    ql = float(np.quantile(env_smooth, max(0.0, min(1.0, float(q_low)))))
    qh = float(np.quantile(env_smooth, max(0.0, min(1.0, float(q_high)))))
    threshold = ql + float(threshold_alpha) * max(0.0, (qh - ql))
    threshold = max(threshold, ql + 1e-6)

    active = env_smooth >= threshold
    min_active = max(1, int(round(float(min_active_sec) * float(sample_rate_hz))))
    min_gap = max(0, int(round(float(min_gap_sec) * float(sample_rate_hz))))

    onsets: list[int] = []
    last_onset = -10**9
    runs = _find_active_runs(active)
    for start, end in runs:
        if (end - start) < min_active:
            continue
        if (start - last_onset) < min_gap:
            continue
        onsets.append(int(start))
        last_onset = int(start)

    return {
        "onsets": onsets,
        "threshold": round(float(threshold), 4),
        "q_low": round(float(ql), 4),
        "q_high": round(float(qh), 4),
        "smooth_window": int(smooth_win),
    }


def _build_clip_relative_path(
    *,
    target_state: str,
    user_id: str,
    session_id: str,
    device_id: str,
    wearing_state: str,
    timestamp: str,
    clip_index: int,
) -> Path:
    stem = (
        f"{timestamp}"
        f"__u-{_sanitize_token(user_id)}"
        f"__s-{_sanitize_token(session_id)}"
        f"__d-{_sanitize_token(device_id)}"
        f"__w-{_sanitize_token(wearing_state)}"
        f"__seg-{int(clip_index):03d}"
    )
    return Path(target_state) / f"{stem}.csv"


def _parse_keep_quality(values: str) -> set[str]:
    tokens = {str(item).strip().lower() for item in str(values).split(",")}
    valid = {"pass", "warn", "retake_recommended"}
    cleaned = {item for item in tokens if item in valid}
    if not cleaned:
        raise ValueError(f"Invalid keep_quality={values!r}. Valid: pass,warn,retake_recommended")
    return cleaned


def _iter_clip_rows(
    matrix: np.ndarray,
    onsets: Iterable[int],
    *,
    pre_roll_samples: int,
    clip_samples: int,
    max_clips: int,
) -> list[tuple[int, int, int]]:
    rows = int(matrix.shape[0])
    clips: list[tuple[int, int, int]] = []
    for onset in onsets:
        start = int(onset) - int(pre_roll_samples)
        end = start + int(clip_samples)
        if start < 0 or end > rows:
            continue
        clips.append((int(onset), int(start), int(end)))
        if max_clips > 0 and len(clips) >= int(max_clips):
            break
    return clips


def _evaluate_collection_gate(
    *,
    rows: int,
    slice_candidate_count: int,
    accepted_clip_count: int,
    min_rows_gate: int,
    min_candidates_gate: int,
    min_accepted_gate: int,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if int(rows) < int(min_rows_gate):
        failures.append(f"rows<{int(min_rows_gate)}")
    if int(slice_candidate_count) < int(min_candidates_gate):
        failures.append(f"slice_candidate_count<{int(min_candidates_gate)}")
    if int(accepted_clip_count) < int(min_accepted_gate):
        failures.append(f"accepted_clip_count<{int(min_accepted_gate)}")
    return (len(failures) == 0), failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous capture + auto-slice clean event clips.")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default="recordings_manifest.csv")
    parser.add_argument("--target_state", required=True)
    parser.add_argument("--start_state", default="CONTINUE")
    parser.add_argument(
        "--capture_mode",
        default="event_onset",
        help="Must match training capture_mode_filter to be trainable (default: event_onset).",
    )
    parser.add_argument("--user_id", default="demo_user")
    parser.add_argument("--session_id", default="s1")
    parser.add_argument("--device_id", default="armband01")
    parser.add_argument("--wearing_state", default="normal")
    parser.add_argument("--duration_sec", type=float, default=45.0)
    parser.add_argument("--clip_duration_sec", type=float, default=3.0)
    parser.add_argument("--pre_roll_ms", type=int, default=500)
    parser.add_argument("--min_active_sec", type=float, default=0.28)
    parser.add_argument("--min_gap_sec", type=float, default=1.0)
    parser.add_argument("--smooth_ms", type=int, default=80)
    parser.add_argument("--q_low", type=float, default=0.20)
    parser.add_argument("--q_high", type=float, default=0.90)
    parser.add_argument("--threshold_alpha", type=float, default=0.35)
    parser.add_argument("--keep_quality", default="pass,warn")
    parser.add_argument("--max_clips", type=int, default=0)
    parser.add_argument("--save_stream_csv", action="store_true")
    parser.add_argument("--min_rows_gate", type=int, default=15000)
    parser.add_argument("--min_candidates_gate", type=int, default=4)
    parser.add_argument("--min_accepted_gate", type=int, default=2)
    parser.add_argument(
        "--enforce_collection_gate",
        action="store_true",
        help="Exit non-zero when rows/candidates/accepted do not meet gate thresholds.",
    )
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--poll_interval_ms", type=int, default=10)
    parser.add_argument("--report_json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from scripts.emg_armband import Device
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "collect_event_data_continuous.py requires pyserial. Install it first: pip install pyserial"
        ) from exc

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    target_state = _normalize_state(args.target_state)
    start_state = _normalize_state(args.start_state)
    if normalize_event_label_input(target_state) == "RELAX":
        raise ValueError("Continuous auto-slice mode is intended for action labels; use target_state != CONTINUE.")

    keep_quality = _parse_keep_quality(args.keep_quality)
    preprocess_cfg, quality_filter = load_collection_protocol(args.config)
    sample_rate_hz = int(preprocess_cfg.sampling_rate)
    pre_roll_samples = int(round(float(args.pre_roll_ms) / 1000.0 * float(sample_rate_hz)))
    clip_samples = int(round(float(args.clip_duration_sec) * float(sample_rate_hz)))
    if clip_samples <= 0:
        raise ValueError("clip_duration_sec must be > 0.")

    manifest_path = Path(args.recordings_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (data_dir / manifest_path).resolve()

    rows: list[list[float]] = []
    frame_count = 0
    pack_count = 0
    started = time.perf_counter()
    device = Device(port=args.port, baudrate=int(args.baudrate), timeout=float(args.timeout))

    print(
        f"[STREAM] start port={args.port} baudrate={int(args.baudrate)} "
        f"duration_sec={float(args.duration_sec):.2f} target_state={target_state}"
    )
    print(
        "[STREAM] protocol: repeat CONTINUE -> fast strong action burst -> CONTINUE. "
        "Do not switch to other action labels in this run."
    )
    device.connect()
    try:
        while (time.perf_counter() - started) < float(args.duration_sec):
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
        raise RuntimeError("No rows captured from armband. Check port/baudrate/device mode and retry.")

    matrix = np.asarray(rows, dtype=np.float32)
    stream_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    stream_csv_path = None
    if bool(args.save_stream_csv):
        stream_rel = Path("_streams") / target_state / (
            f"{stream_timestamp}__u-{_sanitize_token(args.user_id)}"
            f"__s-{_sanitize_token(args.session_id)}"
            f"__d-{_sanitize_token(args.device_id)}"
            f"__w-{_sanitize_token(args.wearing_state)}.csv"
        )
        stream_csv_path = (data_dir / stream_rel).resolve()
        write_standard_csv(stream_csv_path, matrix)
        print(f"[STREAM] stream_csv={stream_csv_path}")

    det = detect_onsets(
        matrix[:, :8],
        sample_rate_hz=sample_rate_hz,
        smooth_ms=int(args.smooth_ms),
        q_low=float(args.q_low),
        q_high=float(args.q_high),
        threshold_alpha=float(args.threshold_alpha),
        min_active_sec=float(args.min_active_sec),
        min_gap_sec=float(args.min_gap_sec),
    )
    clips = _iter_clip_rows(
        matrix,
        det["onsets"],
        pre_roll_samples=pre_roll_samples,
        clip_samples=clip_samples,
        max_clips=int(args.max_clips),
    )

    accepted = 0
    rejected = 0
    clip_reports: list[dict[str, Any]] = []
    for clip_index, (onset, start, end) in enumerate(clips, start=1):
        clip_matrix = matrix[start:end, :]
        quality = evaluate_recording_quality(
            clip_matrix,
            preprocess_config=preprocess_cfg,
            quality_filter=quality_filter,
        )
        status = str(quality.get("quality_status", "warn")).lower()
        keep = status in keep_quality
        rel = _build_clip_relative_path(
            target_state=target_state,
            user_id=args.user_id,
            session_id=args.session_id,
            device_id=args.device_id,
            wearing_state=args.wearing_state,
            timestamp=stream_timestamp,
            clip_index=clip_index,
        )
        abs_path = (data_dir / rel).resolve()

        clip_report = {
            "clip_index": int(clip_index),
            "onset_sample": int(onset),
            "start_sample": int(start),
            "end_sample": int(end),
            "relative_path": rel.as_posix(),
            "quality_status": str(quality.get("quality_status")),
            "quality_reasons": list(quality.get("quality_reasons", [])),
            "kept": bool(keep),
            "selected_windows": int(quality.get("kept_segments", 0)),
            "total_windows": int(quality.get("total_segments", 0)),
        }

        if keep:
            write_standard_csv(abs_path, clip_matrix)
            manifest_row = {
                "relative_path": rel.as_posix(),
                "gesture": target_state,
                "capture_mode": str(args.capture_mode),
                "start_state": start_state,
                "target_state": target_state,
                "user_id": _sanitize_token(args.user_id),
                "session_id": _sanitize_token(args.session_id),
                "device_id": _sanitize_token(args.device_id),
                "timestamp": stream_timestamp,
                "wearing_state": _sanitize_token(args.wearing_state),
                "recording_id": rel.stem,
                "sample_count": int(clip_matrix.shape[0]),
                "clip_duration_ms": int(round(float(args.clip_duration_sec) * 1000.0)),
                "pre_roll_ms": int(args.pre_roll_ms),
                "device_sampling_rate_hz": int(sample_rate_hz),
                "imu_sampling_rate_hz": 50,
                "quality_status": str(quality.get("quality_status", "warn")),
                "quality_reasons": "|".join(quality.get("quality_reasons", [])),
                "source_origin": "armband_stream_auto_slice",
            }
            upsert_event_manifest(manifest_path, manifest_row)
            accepted += 1
            print(
                f"[SLICE] keep clip={clip_index} status={manifest_row['quality_status']} "
                f"windows={clip_report['selected_windows']}/{clip_report['total_windows']} path={abs_path}"
            )
        else:
            rejected += 1
            print(
                f"[SLICE] drop clip={clip_index} status={clip_report['quality_status']} "
                f"reasons={','.join(clip_report['quality_reasons']) or 'none'}"
            )
        clip_reports.append(clip_report)

    report = {
        "status": "ok",
        "target_state": target_state,
        "start_state": start_state,
        "rows": int(matrix.shape[0]),
        "frames": int(frame_count),
        "emg_packs": int(pack_count),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "estimated_emg_rate_hz": round(float(matrix.shape[0]) / max(float(elapsed_sec), 1e-6), 3),
        "stream_csv_path": str(stream_csv_path) if stream_csv_path is not None else None,
        "manifest_path": str(manifest_path),
        "sample_rate_hz": int(sample_rate_hz),
        "onset_count": int(len(det["onsets"])),
        "slice_candidate_count": int(len(clips)),
        "accepted_clip_count": int(accepted),
        "rejected_clip_count": int(rejected),
        "detection": det,
        "keep_quality": sorted(keep_quality),
        "clips": clip_reports,
    }
    gate_passed, gate_failures = _evaluate_collection_gate(
        rows=int(report["rows"]),
        slice_candidate_count=int(report["slice_candidate_count"]),
        accepted_clip_count=int(report["accepted_clip_count"]),
        min_rows_gate=int(args.min_rows_gate),
        min_candidates_gate=int(args.min_candidates_gate),
        min_accepted_gate=int(args.min_accepted_gate),
    )
    report["collection_gate"] = {
        "passed": bool(gate_passed),
        "failures": gate_failures,
        "thresholds": {
            "min_rows_gate": int(args.min_rows_gate),
            "min_candidates_gate": int(args.min_candidates_gate),
            "min_accepted_gate": int(args.min_accepted_gate),
        },
    }

    if str(args.report_json or "").strip():
        report_path = Path(args.report_json)
        if not report_path.is_absolute():
            report_path = (data_dir / report_path).resolve()
    else:
        report_path = (data_dir / "_streams" / target_state / f"{stream_timestamp}__slice_report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[STREAM] rows={report['rows']} onsets={report['onset_count']} "
        f"candidates={report['slice_candidate_count']} accepted={accepted} rejected={rejected}"
    )
    if gate_passed:
        print("[STREAM] gate=PASS")
    else:
        print(f"[STREAM][WARN] gate=FAIL failures={','.join(gate_failures)}")
    print(f"[STREAM] report_json={report_path}")
    if bool(args.enforce_collection_gate) and (not gate_passed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
