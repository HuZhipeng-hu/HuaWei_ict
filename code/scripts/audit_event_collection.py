"""Audit locally collected event-onset data and classify clip quality."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from scripts.collection_utils import STANDARD_CSV_HEADERS, evaluate_recording_quality
from shared.config import PreprocessConfig


def _read_standard_matrix(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in STANDARD_CSV_HEADERS if name not in fields]
        if missing:
            raise ValueError(f"missing standardized columns: {missing}")
        rows = [[float(row[name]) for name in STANDARD_CSV_HEADERS] for row in reader]
    if not rows:
        raise ValueError("empty data rows")
    return np.asarray(rows, dtype=np.float32)


def _build_preprocess_config(data_cfg) -> PreprocessConfig:
    payload = {
        "sampling_rate": int(data_cfg.device_sampling_rate_hz),
        "num_channels": 8,
        "target_length": int(data_cfg.context_samples),
        "overlap": max(
            0.0,
            min(0.99, 1.0 - float(data_cfg.window_step_samples) / float(data_cfg.context_samples)),
        ),
        "stft_window": int(data_cfg.feature.emg_stft_window),
        "stft_hop": int(data_cfg.feature.emg_stft_hop),
        "n_fft": int(data_cfg.feature.emg_n_fft),
        "freq_bins_out": int(data_cfg.feature.emg_freq_bins),
        "dual_branch": {"enabled": False},
    }
    return PreprocessConfig(**payload)


def _categorize_clip(item: dict[str, Any]) -> str:
    if item.get("quality_status") == "retake_recommended":
        return "retake"
    if item.get("dead_channels"):
        return "retake"
    if item.get("quality_status") == "warn":
        return "suspicious"
    if item.get("weak_channels") or item.get("dominant_channels"):
        return "suspicious"
    if item.get("extra_flags"):
        return "suspicious"
    return "usable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit event-onset collected clips.")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default="recordings_manifest.csv")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_id", default="collection_audit_latest")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    code_root = Path(".").resolve()
    data_dir = Path(args.data_dir).resolve()
    manifest_path = Path(args.recordings_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (data_dir / manifest_path).resolve()

    _, data_cfg, train_cfg, _ = load_event_training_config(Path(args.config))
    pre_cfg = _build_preprocess_config(data_cfg)
    quality_filter = train_cfg.quality_filter

    if not manifest_path.exists():
        raise FileNotFoundError(f"recordings manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    details: list[dict[str, Any]] = []
    missing_files: list[str] = []
    parse_fail: list[dict[str, str]] = []
    for row in manifest_rows:
        rel = str(row.get("relative_path", "")).replace("\\", "/").strip()
        if not rel:
            continue
        csv_path = (data_dir / rel).resolve()
        target_state = str(row.get("target_state", "")).strip().upper()
        if not csv_path.exists():
            missing_files.append(rel)
            continue
        try:
            matrix = _read_standard_matrix(csv_path)
            report = evaluate_recording_quality(
                matrix,
                preprocess_config=pre_cfg,
                quality_filter=quality_filter,
            )
            row_count = int(report.get("row_count", matrix.shape[0]))
            duration_sec_est = float(row_count) / float(data_cfg.device_sampling_rate_hz)
            expected_duration_sec = float(int(row.get("clip_duration_ms", "0") or 0)) / 1000.0
            extra_flags: list[str] = []
            if expected_duration_sec > 0 and abs(duration_sec_est - expected_duration_sec) > 2.5:
                extra_flags.append(
                    f"duration_mismatch(expected={expected_duration_sec:.1f}s,actual={duration_sec_est:.1f}s)"
                )
            if duration_sec_est < 2.0:
                extra_flags.append(f"too_short({duration_sec_est:.1f}s)")
            if duration_sec_est > 20.0:
                extra_flags.append(f"very_long({duration_sec_est:.1f}s)")

            anomaly = report.get("channel_anomaly") or {}
            item = {
                "relative_path": rel,
                "target_state": target_state,
                "row_count": row_count,
                "duration_sec_est": round(duration_sec_est, 3),
                "quality_status": report.get("quality_status"),
                "quality_reasons": report.get("quality_reasons", []),
                "kept_ratio": float(report.get("kept_ratio", 0.0)),
                "dead_channels": list(anomaly.get("dead_channels", [])),
                "weak_channels": list(anomaly.get("weak_channels", [])),
                "dominant_channels": list(anomaly.get("dominant_channels", [])),
                "extra_flags": extra_flags,
            }
            item["category"] = _categorize_clip(item)
            details.append(item)
        except Exception as exc:
            parse_fail.append({"relative_path": rel, "error": str(exc)})

    # Window-level reachability report from current loader config
    selected_by_clip: Counter[str] = Counter()
    loader_error = None
    try:
        loader = EventClipDatasetLoader(data_dir, data_cfg, recordings_manifest_path=manifest_path)
        _, _, _, source_ids, _ = loader.load_all_with_sources(return_metadata=True)
        selected_by_clip = Counter([str(item) for item in source_ids.tolist()])
    except Exception as exc:
        loader_error = str(exc)

    for item in details:
        item["selected_windows"] = int(selected_by_clip.get(item["relative_path"], 0))

    categories = Counter([item["category"] for item in details])
    by_class_total = Counter([item["target_state"] for item in details])
    by_class_usable = Counter([item["target_state"] for item in details if item["category"] == "usable"])
    by_class_susp = Counter([item["target_state"] for item in details if item["category"] == "suspicious"])
    by_class_retake = Counter([item["target_state"] for item in details if item["category"] == "retake"])
    zero_selected = [item["relative_path"] for item in details if int(item.get("selected_windows", 0)) == 0]

    summary = {
        "config": str(Path(args.config).resolve()),
        "data_dir": str(data_dir),
        "recordings_manifest": str(manifest_path),
        "label_set_required": ["RELAX", *list(data_cfg.target_db5_keys)],
        "total_manifest_rows": len(manifest_rows),
        "checked_rows": len(details),
        "missing_files_count": len(missing_files),
        "parse_fail_count": len(parse_fail),
        "usable_count": int(categories.get("usable", 0)),
        "suspicious_count": int(categories.get("suspicious", 0)),
        "retake_count": int(categories.get("retake", 0)),
        "zero_selected_window_clips": len(zero_selected),
        "loader_error": loader_error,
        "by_class_total": dict(by_class_total),
        "by_class_usable": dict(by_class_usable),
        "by_class_suspicious": dict(by_class_susp),
        "by_class_retake": dict(by_class_retake),
    }

    out_dir = Path(args.run_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "collection_audit_summary.json"
    details_json = out_dir / "collection_audit_details.json"
    summary_md = out_dir / "collection_audit_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    details_json.write_text(
        json.dumps(
            {
                "details": details,
                "missing_files": missing_files,
                "parse_fail": parse_fail,
                "selected_windows_by_clip": dict(selected_by_clip),
                "zero_selected_window_clips": zero_selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Collection Audit Summary",
        "",
        f"- total_manifest_rows: `{summary['total_manifest_rows']}`",
        f"- checked_rows: `{summary['checked_rows']}`",
        f"- usable_count: `{summary['usable_count']}`",
        f"- suspicious_count: `{summary['suspicious_count']}`",
        f"- retake_count: `{summary['retake_count']}`",
        f"- zero_selected_window_clips: `{summary['zero_selected_window_clips']}`",
        "",
        "## By Class",
        f"- total: `{json.dumps(summary['by_class_total'], ensure_ascii=False)}`",
        f"- usable: `{json.dumps(summary['by_class_usable'], ensure_ascii=False)}`",
        f"- suspicious: `{json.dumps(summary['by_class_suspicious'], ensure_ascii=False)}`",
        f"- retake: `{json.dumps(summary['by_class_retake'], ensure_ascii=False)}`",
        "",
        f"- summary_json: `{summary_json}`",
        f"- details_json: `{details_json}`",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[AUDIT] summary={summary_json}")
    print(f"[AUDIT] details={details_json}")
    print(f"[AUDIT] markdown={summary_md}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
