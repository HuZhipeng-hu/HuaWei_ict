"""Build a low-risk training manifest from collection audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_TARGET_STATES = [
    "RELAX",
    "TENSE_OPEN",
    "V_SIGN",
    "OK_SIGN",
    "THUMB_UP",
    "WRIST_CW",
    "WRIST_CCW",
]


def _normalize_rel(path_value: str) -> str:
    return Path(str(path_value).replace("\\", "/")).as_posix()


def _parse_target_states(raw: str | None) -> list[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_TARGET_STATES)
    parsed = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    return parsed or list(DEFAULT_TARGET_STATES)


def _load_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{k: str(v or "") for k, v in row.items()} for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"manifest has no rows: {path}")
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    return rows, fieldnames


def _load_audit_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = list(payload.get("details", []))
    index: dict[str, dict[str, Any]] = {}
    for item in details:
        rel = _normalize_rel(str(item.get("relative_path", "")))
        if rel:
            index[rel] = dict(item)
    return index


def _classify_row(
    row: dict[str, str],
    detail: dict[str, Any] | None,
    *,
    require_capture_mode: str | None,
    min_selected_windows: int,
    allow_retake_quality: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if require_capture_mode and str(row.get("capture_mode", "")).strip() != str(require_capture_mode).strip():
        reasons.append(f"capture_mode!= {require_capture_mode}")
    if detail is None:
        reasons.append("missing_audit_detail")
        return False, reasons

    category = str(detail.get("category", "")).strip().lower()
    quality_status = str(detail.get("quality_status", "")).strip().lower()
    selected_windows = int(detail.get("selected_windows", 0) or 0)
    dead_channels = list(detail.get("dead_channels", []) or [])

    retake_relaxed = allow_retake_quality and quality_status == "retake_recommended" and not dead_channels

    if category == "retake" and not retake_relaxed:
        reasons.append("retake_category")
    if quality_status == "retake_recommended" and not allow_retake_quality:
        reasons.append("retake_quality_status")
    if selected_windows < int(min_selected_windows):
        reasons.append(f"selected_windows<{int(min_selected_windows)}")
    if dead_channels:
        reasons.append("dead_channels")

    allowed_quality_status = {"pass", "warn"}
    if allow_retake_quality:
        allowed_quality_status.add("retake_recommended")
    if quality_status not in allowed_quality_status:
        reasons.append(f"unsupported_quality_status={quality_status or 'empty'}")

    return (len(reasons) == 0), reasons


def _write_manifest(path: Path, *, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build s2 low-risk train manifest from audit details.")
    parser.add_argument("--recordings_manifest", default="../data/recordings_manifest.csv")
    parser.add_argument("--audit_details_json", default="artifacts/runs/collection_audit_latest/collection_audit_details.json")
    parser.add_argument("--session_id", default="s2")
    parser.add_argument("--target_states", default=",".join(DEFAULT_TARGET_STATES))
    parser.add_argument("--target_per_class", type=int, default=12)
    parser.add_argument("--relax_target_count", type=int, default=24)
    parser.add_argument("--require_capture_mode", default="event_onset")
    parser.add_argument("--action_min_selected_windows", type=int, default=None)
    parser.add_argument("--relax_min_selected_windows", type=int, default=1)
    parser.add_argument("--relax_allow_retake_quality", default="true")
    parser.add_argument("--max_per_class", type=int, default=0)
    parser.add_argument("--min_selected_windows", type=int, default=2)
    parser.add_argument("--output_manifest", default="../data/s2_train_manifest.csv")
    parser.add_argument("--output_drop_report", default="artifacts/runs/s2_train_manifest_dropped_report.json")
    return parser


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def main() -> None:
    args = build_parser().parse_args()
    recordings_manifest = Path(args.recordings_manifest).resolve()
    audit_details = Path(args.audit_details_json).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_drop_report = Path(args.output_drop_report).resolve()

    if not recordings_manifest.exists():
        raise FileNotFoundError(f"recordings_manifest not found: {recordings_manifest}")
    if not audit_details.exists():
        raise FileNotFoundError(f"audit_details_json not found: {audit_details}")

    target_states = _parse_target_states(args.target_states)
    target_set = set(target_states)

    action_min_selected_windows = (
        int(args.action_min_selected_windows)
        if args.action_min_selected_windows is not None
        else int(args.min_selected_windows)
    )
    relax_min_selected_windows = int(args.relax_min_selected_windows)
    relax_allow_retake_quality = _parse_bool(args.relax_allow_retake_quality, default=True)
    max_per_class = int(args.max_per_class)

    rows, fieldnames = _load_manifest(recordings_manifest)
    detail_by_rel = _load_audit_index(audit_details)

    session_id = str(args.session_id).strip()
    considered: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("session_id", "")).strip() != session_id:
            continue
        target_state = str(row.get("target_state", "")).strip().upper()
        if target_state not in target_set:
            continue
        considered.append(dict(row))

    kept_rows: list[dict[str, str]] = []
    dropped_rows: list[dict[str, Any]] = []
    by_target_total = Counter()
    by_target_kept = Counter()
    by_target_dropped = Counter()
    relax_kept_by_rule = Counter()
    relax_dropped_by_rule = Counter()

    for row in considered:
        rel = _normalize_rel(row.get("relative_path", ""))
        row["relative_path"] = rel
        target_state = str(row.get("target_state", "")).strip().upper()
        by_target_total[target_state] += 1
        detail = detail_by_rel.get(rel)
        is_relax = target_state == "RELAX"

        keep, reasons = _classify_row(
            row,
            detail,
            require_capture_mode=str(args.require_capture_mode or "").strip() or None,
            min_selected_windows=(relax_min_selected_windows if is_relax else action_min_selected_windows),
            allow_retake_quality=(relax_allow_retake_quality if is_relax else False),
        )
        if keep:
            kept_rows.append(row)
            by_target_kept[target_state] += 1
            if is_relax:
                quality_status = str((detail or {}).get("quality_status", "")).strip().lower() or "empty"
                category = str((detail or {}).get("category", "")).strip().lower() or "empty"
                relax_kept_by_rule[f"quality_status={quality_status}"] += 1
                relax_kept_by_rule[f"category={category}"] += 1
        else:
            by_target_dropped[target_state] += 1
            if is_relax:
                for reason in reasons:
                    relax_dropped_by_rule[str(reason)] += 1
            dropped_rows.append(
                {
                    "relative_path": rel,
                    "target_state": target_state,
                    "session_id": row.get("session_id", ""),
                    "quality_status": row.get("quality_status", ""),
                    "quality_reasons": row.get("quality_reasons", ""),
                    "drop_reasons": reasons,
                    "audit_category": (detail or {}).get("category") if detail else None,
                    "audit_selected_windows": (detail or {}).get("selected_windows") if detail else None,
                }
            )

    kept_rows_sorted = sorted(
        kept_rows,
        key=lambda item: (
            str(item.get("target_state", "")),
            str(item.get("timestamp", "")),
            str(item.get("relative_path", "")),
        ),
    )

    if max_per_class > 0:
        by_class_cap_counter: Counter[str] = Counter()
        capped_rows: list[dict[str, str]] = []
        for row in kept_rows_sorted:
            target_state = str(row.get("target_state", "")).strip().upper()
            if by_class_cap_counter[target_state] >= max_per_class:
                continue
            capped_rows.append(row)
            by_class_cap_counter[target_state] += 1
        kept_rows_sorted = capped_rows

    kept_by_class_after_cap = Counter(str(row.get("target_state", "")).strip().upper() for row in kept_rows_sorted)
    _write_manifest(output_manifest, rows=kept_rows_sorted, fieldnames=fieldnames)

    coverage = {}
    for state in target_states:
        current = int(kept_by_class_after_cap.get(state, 0))
        target = int(args.relax_target_count) if state == "RELAX" else int(args.target_per_class)
        coverage[state] = {
            "target": target,
            "current": current,
            "gap": max(0, int(target - current)),
        }

    report = {
        "status": "ok",
        "session_id": session_id,
        "source_recordings_manifest": str(recordings_manifest),
        "source_audit_details_json": str(audit_details),
        "output_manifest": str(output_manifest),
        "rules": {
            "require_capture_mode": str(args.require_capture_mode),
            "action_min_selected_windows": int(action_min_selected_windows),
            "relax_min_selected_windows": int(relax_min_selected_windows),
            "relax_allow_retake_quality": bool(relax_allow_retake_quality),
            "relax_target_count": int(args.relax_target_count),
            "max_per_class": int(max_per_class),
            "keep_quality_status": (
                ["pass", "warn", "retake_recommended"]
                if relax_allow_retake_quality
                else ["pass", "warn"]
            ),
            "drop_categories": ["retake"],
            "drop_if_dead_channels": True,
        },
        "target_states": target_states,
        "target_per_class": int(args.target_per_class),
        "counts": {
            "considered": int(len(considered)),
            "kept": int(len(kept_rows_sorted)),
            "dropped": int(len(dropped_rows)),
        },
        "by_target_total": dict(by_target_total),
        "by_target_kept": dict(by_target_kept),
        "by_target_dropped": dict(by_target_dropped),
        "kept_by_class_after_cap": dict(kept_by_class_after_cap),
        "relax_kept_by_rule": dict(relax_kept_by_rule),
        "relax_dropped_by_rule": dict(relax_dropped_by_rule),
        "coverage": coverage,
        "dropped_rows": dropped_rows,
    }
    output_drop_report.parent.mkdir(parents=True, exist_ok=True)
    output_drop_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[S2-MANIFEST] output_manifest={output_manifest}")
    print(f"[S2-MANIFEST] drop_report={output_drop_report}")
    print(
        json.dumps(
            {
                "session_id": session_id,
                "considered": len(considered),
                "kept": len(kept_rows_sorted),
                "dropped": len(dropped_rows),
                "coverage_gap": {k: v["gap"] for k, v in coverage.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


