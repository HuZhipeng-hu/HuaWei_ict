"""Print a short, copy-friendly summary for demo/algo matrix results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be object: {path}")
    return payload


def _pick_metric(row: dict, *keys: str, default: float | None = None):
    for key in keys:
        value = row.get(key, None)
        if value is None:
            continue
        return value
    return default


def _resolve_summary(path_arg: str | None, run_root: Path) -> Path:
    if str(path_arg or "").strip():
        return Path(str(path_arg)).resolve()
    candidates = sorted(
        run_root.glob("*_summary/*summary*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no summary json found under: {run_root}")
    return candidates[0]


def _collect_top_confusions(best_row: dict) -> list[dict]:
    metrics_path = str(best_row.get("metrics_path", "")).strip()
    if not metrics_path:
        return []
    metrics_file = Path(metrics_path)
    if not metrics_file.is_absolute():
        metrics_file = Path.cwd() / metrics_file
    if not metrics_file.exists():
        return []
    payload = _load_json(metrics_file)
    return list(payload.get("top_confusion_pairs") or [])[:5]


def _build_invalid_rows(rows: list[dict]) -> list[dict]:
    invalid = []
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status != "invalid":
            continue
        invalid.append(
            {
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "warning_reasons": row.get("warning_reasons", ""),
                "control_eval_present": row.get("control_eval_present"),
                "control_eval_source": row.get("control_eval_source"),
            }
        )
    return invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Print short summary for matrix/sweep outputs")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary_path = _resolve_summary(args.summary_json, run_root)
    summary = _load_json(summary_path)
    rows = list(summary.get("rows") or [])
    best = dict(summary.get("best_run") or {})

    output = {
        "summary_path": str(summary_path),
        "status": summary.get("status"),
        "best_run_id": summary.get("best_run_id") or best.get("run_id"),
        "best_metrics": {
            "event_action_accuracy": _pick_metric(best, "event_action_accuracy", "test_event_action_accuracy"),
            "event_action_macro_f1": _pick_metric(best, "event_action_macro_f1", "test_event_action_macro_f1"),
            "test_accuracy": _pick_metric(best, "test_accuracy"),
            "test_macro_f1": _pick_metric(best, "test_macro_f1"),
            "command_success_rate": _pick_metric(best, "command_success_rate", "test_command_success_rate"),
            "false_trigger_rate": _pick_metric(best, "false_trigger_rate", "test_false_trigger_rate"),
            "false_release_rate": _pick_metric(best, "false_release_rate", "test_false_release_rate"),
            "safety_ok": best.get("safety_ok"),
        },
        "top_confusion_pairs": _collect_top_confusions(best),
        "invalid_rows": _build_invalid_rows(rows),
        "row_count": len(rows),
        "valid_row_count": summary.get("valid_row_count"),
        "invalid_row_count": summary.get("invalid_row_count"),
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

