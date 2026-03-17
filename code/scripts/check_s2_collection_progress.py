"""Check s2 collection coverage and per-round gate status."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.collect_event_data_continuous import _evaluate_collection_gate
from shared.event_labels import normalize_event_label_input, public_event_label

DEFAULT_TARGET_STATES = [
    "CONTINUE",
    "TENSE_OPEN",
    "V_SIGN",
    "OK_SIGN",
    "THUMB_UP",
    "WRIST_CW",
    "WRIST_CCW",
]


def _parse_target_states(raw: str | None) -> list[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_TARGET_STATES)
    parsed = [normalize_event_label_input(item) for item in str(raw).split(",") if item.strip()]
    return public_event_labels(parsed) if parsed else list(DEFAULT_TARGET_STATES)


def public_event_labels(values: list[str]) -> list[str]:
    return [public_event_label(value) for value in values]


def _latest_slice_report_by_state(stream_root: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    if not stream_root.exists():
        return latest
    reports = sorted(
        stream_root.rglob("*__slice_report.json"),
        key=lambda p: (p.stat().st_mtime, str(p)),
        reverse=True,
    )
    for path in reports:
        state = path.parent.name.upper()
        latest.setdefault(state, path)
    return latest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check s2 collection gaps and gate status.")
    parser.add_argument("--recordings_manifest", default="../data/recordings_manifest.csv")
    parser.add_argument("--session_id", default="s2")
    parser.add_argument("--streams_root", default="../data/_streams")
    parser.add_argument("--target_states", default=",".join(DEFAULT_TARGET_STATES))
    parser.add_argument("--target_per_class", type=int, default=12)
    parser.add_argument("--min_rows_gate", type=int, default=15000)
    parser.add_argument("--min_candidates_gate", type=int, default=4)
    parser.add_argument("--min_accepted_gate", type=int, default=2)
    parser.add_argument("--output_json", default="artifacts/runs/s2_collection_progress.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.recordings_manifest).resolve()
    streams_root = Path(args.streams_root).resolve()
    output_json = Path(args.output_json).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"recordings_manifest not found: {manifest_path}")

    target_states = _parse_target_states(args.target_states)
    session_id = str(args.session_id).strip()
    target_state_set = {normalize_event_label_input(state) for state in target_states}
    rows = list(csv.DictReader(open(manifest_path, "r", encoding="utf-8-sig", newline="")))
    filtered = [
        row
        for row in rows
        if str(row.get("session_id", "")).strip() == session_id
        and normalize_event_label_input(row.get("target_state", "")) in target_state_set
    ]
    by_state = Counter(public_event_label(row.get("target_state", "")) for row in filtered)
    coverage = {
        state: {
            "target": int(args.target_per_class),
            "current": int(by_state.get(state, 0)),
            "gap": max(0, int(args.target_per_class) - int(by_state.get(state, 0))),
        }
        for state in target_states
    }

    latest_reports = _latest_slice_report_by_state(streams_root)
    gate_by_state: dict[str, dict[str, Any]] = {}
    for state in target_states:
        if normalize_event_label_input(state) == "RELAX":
            continue
        report_path = latest_reports.get(state)
        if report_path is None:
            gate_by_state[state] = {"status": "missing_report"}
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        passed, failures = _evaluate_collection_gate(
            rows=int(payload.get("rows", 0) or 0),
            slice_candidate_count=int(payload.get("slice_candidate_count", 0) or 0),
            accepted_clip_count=int(payload.get("accepted_clip_count", 0) or 0),
            min_rows_gate=int(args.min_rows_gate),
            min_candidates_gate=int(args.min_candidates_gate),
            min_accepted_gate=int(args.min_accepted_gate),
        )
        gate_by_state[state] = {
            "status": "pass" if passed else "fail",
            "failures": failures,
            "report_path": str(report_path),
            "rows": int(payload.get("rows", 0) or 0),
            "slice_candidate_count": int(payload.get("slice_candidate_count", 0) or 0),
            "accepted_clip_count": int(payload.get("accepted_clip_count", 0) or 0),
            "rejected_clip_count": int(payload.get("rejected_clip_count", 0) or 0),
        }

    result = {
        "status": "ok",
        "session_id": session_id,
        "recordings_manifest": str(manifest_path),
        "streams_root": str(streams_root),
        "target_states": target_states,
        "target_per_class": int(args.target_per_class),
        "coverage": coverage,
        "gate_thresholds": {
            "min_rows_gate": int(args.min_rows_gate),
            "min_candidates_gate": int(args.min_candidates_gate),
            "min_accepted_gate": int(args.min_accepted_gate),
        },
        "gate_by_state": gate_by_state,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[S2-PROGRESS] output_json={output_json}")
    print(
        json.dumps(
            {
                "session_id": session_id,
                "coverage_gap": {k: v["gap"] for k, v in coverage.items()},
                "gate_state": {k: v["status"] for k, v in gate_by_state.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
