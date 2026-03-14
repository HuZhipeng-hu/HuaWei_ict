"""Diagnose DB5 representation pretrain matrix outputs without retraining."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _pick(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", []):
            return value
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isnan(parsed):
            return float(default)
        return parsed
    except Exception:
        return float(default)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    norm = dict(row)
    norm["phase"] = _pick(norm, "phase", "round")
    norm["candidate"] = _pick(norm, "candidate", "name")
    norm["run_id"] = _pick(norm, "run_id", "pretrain_run_id")
    norm["checkpoint_path"] = _pick(norm, "checkpoint_path", "encoder_checkpoint_path")
    norm["pretrain_best_val_macro_f1"] = _as_float(norm.get("pretrain_best_val_macro_f1"), 0.0)
    norm["pretrain_best_val_acc"] = _as_float(norm.get("pretrain_best_val_acc"), 0.0)
    norm["pretrain_test_macro_f1"] = _as_float(norm.get("pretrain_test_macro_f1"), 0.0)
    return norm


def _pretrain_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -_as_float(row.get("pretrain_best_val_macro_f1"), 0.0),
        -_as_float(row.get("pretrain_best_val_acc"), 0.0),
        -_as_float(row.get("pretrain_test_macro_f1"), 0.0),
    )


def _find_latest_summary(run_root: Path) -> Path:
    summaries = sorted(
        run_root.glob("*_summary/db5_repr_method_matrix_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise FileNotFoundError(f"No db5_repr_method_matrix_summary.json under {run_root}")
    return summaries[0]


def _schema_findings(summary: dict[str, Any], rows: list[dict[str, Any]], best_norm: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if not isinstance(summary.get("best_pretrain_run"), dict) or not summary.get("best_pretrain_run"):
        findings.append("best_pretrain_run is missing or empty.")

    summary_best_id = _pick(summary, "best_pretrain_run_id")
    if not summary_best_id:
        findings.append("best_pretrain_run_id missing in summary root.")
    elif str(summary_best_id) != str(best_norm.get("run_id", "")):
        findings.append(
            f"best_pretrain_run_id mismatch: root={summary_best_id} resolved={best_norm.get('run_id')}"
        )

    if not rows:
        findings.append("rows is empty; cannot rank pretrain candidates.")
        return findings

    required_alias_groups = [
        ("phase", "round"),
        ("candidate", "name"),
        ("run_id", "pretrain_run_id"),
        ("checkpoint_path", "encoder_checkpoint_path"),
    ]
    for idx, raw in enumerate(rows):
        for left, right in required_alias_groups:
            if _pick(raw, left, right) in (None, "", []):
                findings.append(f"row[{idx}] missing compatible key pair: {left}|{right}")
    return findings


def _resolve_best(summary: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    best_payload = summary.get("best_pretrain_run")
    if isinstance(best_payload, dict) and best_payload:
        best_norm = _normalize_row(best_payload)
        if best_norm.get("run_id"):
            best_norm["source"] = "summary.best_pretrain_run"
            return best_norm, warnings
        warnings.append("best_pretrain_run exists but run_id/pretrain_run_id is missing.")

    if not rows:
        raise RuntimeError("Cannot resolve best run: no rows in summary.")
    ranked = sorted((_normalize_row(row) for row in rows), key=_pretrain_rank_key)
    best_norm = dict(ranked[0])
    best_norm["source"] = "fallback.rows_ranking"
    warnings.append("Fell back to rows ranking to resolve best run.")
    return best_norm, warnings


def _build_rows_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        norm = _normalize_row(row)
        val_f1 = _as_float(norm.get("pretrain_best_val_macro_f1"), 0.0)
        test_f1 = _as_float(norm.get("pretrain_test_macro_f1"), 0.0)
        output.append(
            {
                "phase": norm.get("phase"),
                "candidate": norm.get("candidate"),
                "run_id": norm.get("run_id"),
                "val_f1": val_f1,
                "val_acc": _as_float(norm.get("pretrain_best_val_acc"), 0.0),
                "test_f1": test_f1,
                "test_val_gap": float(test_f1 - val_f1),
            }
        )
    return output


def _model_findings(best: dict[str, Any], rows_metrics: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    best_val_f1 = _as_float(best.get("pretrain_best_val_macro_f1"), 0.0)
    best_test_f1 = _as_float(best.get("pretrain_test_macro_f1"), 0.0)
    gap = float(best_test_f1 - best_val_f1)
    if best_val_f1 < 0.05 and best_test_f1 < 0.05:
        findings.append(
            f"Representation learning is weak overall (best val_f1={best_val_f1:.4f}, test_f1={best_test_f1:.4f})."
        )
    if gap < -0.03:
        findings.append(f"Generalization gap is high on best run (test-val gap={gap:.4f}).")
    elif best_val_f1 < 0.05 and abs(gap) <= 0.01:
        findings.append(f"Val/Test are both low with small gap (test-val gap={gap:.4f}), likely weak signal not overfit.")

    if rows_metrics:
        best_row = sorted(rows_metrics, key=lambda row: (-row["val_f1"], -row["val_acc"], -row["test_f1"]))[0]
        worst_row = sorted(rows_metrics, key=lambda row: (row["val_f1"], row["val_acc"], row["test_f1"]))[0]
        gain = float(best_row["val_f1"] - worst_row["val_f1"])
        findings.append(f"Matrix spread on val_f1 is {gain:.4f} (best={best_row['val_f1']:.4f}, worst={worst_row['val_f1']:.4f}).")
    return findings


def _artifact_findings(run_root: Path, best_run_id: str) -> tuple[list[str], dict[str, Any]]:
    findings: list[str] = []
    details: dict[str, Any] = {}
    run_dir = run_root / str(best_run_id)
    details["run_dir"] = str(run_dir)

    required = {
        "offline_summary": run_dir / "offline_summary.json",
        "repr_eval_summary": run_dir / "repr_eval" / "repr_eval_summary.json",
        "split_diagnostics": run_dir / "db5_split_diagnostics.json",
        "window_diagnostics": run_dir / "db5_window_diagnostics.json",
    }
    exists = {name: path.exists() for name, path in required.items()}
    details["required_artifacts"] = {name: str(path) for name, path in required.items()}
    details["required_exists"] = exists
    for name, ok in exists.items():
        if not ok:
            findings.append(f"Missing artifact: {name} ({required[name]})")

    if required["offline_summary"].exists():
        offline = _load_json(required["offline_summary"])
        best_epoch = int(_as_float(offline.get("best_val_epoch"), 0))
        details["best_val_epoch"] = best_epoch
        if best_epoch <= 5:
            findings.append(f"Best epoch is very early (best_val_epoch={best_epoch}), training signal may be unstable.")

    if required["split_diagnostics"].exists():
        split_diag = _load_json(required["split_diagnostics"])
        by_split = dict(split_diag.get("by_split", {}) or {})
        details["split_by_split"] = by_split
        for split_name, split_payload in by_split.items():
            class_counts = dict(split_payload.get("class_counts", {}) or {})
            positives = [int(v) for v in class_counts.values() if int(v) > 0]
            min_count = int(split_payload.get("min_class_count", 0) or 0)
            has_empty = bool(split_payload.get("has_empty_classes", False))
            if has_empty or min_count <= 0:
                findings.append(f"Split {split_name} has empty classes (min_class_count={min_count}).")
            if positives:
                ratio = float(max(positives) / max(1, min(positives)))
                if ratio > 20.0:
                    findings.append(f"Split {split_name} is severely imbalanced (max/min ratio={ratio:.2f}).")

    if required["window_diagnostics"].exists():
        window_diag = _load_json(required["window_diagnostics"])
        totals = dict(window_diag.get("totals", {}) or {})
        details["window_totals"] = totals
        raw = int(totals.get("raw_candidates", 0) or 0)
        passed = int(totals.get("passed_quality", 0) or 0)
        selected = int(totals.get("selected", 0) or 0)
        if raw > 0:
            passed_ratio = float(passed / raw)
            selected_ratio = float(selected / raw)
            details["window_passed_ratio"] = passed_ratio
            details["window_selected_ratio"] = selected_ratio
            if passed_ratio < 0.15:
                findings.append(f"Window quality gate may be too strict (passed/raw={passed_ratio:.4f}).")
            if selected_ratio < 0.05:
                findings.append(f"Too few windows survive selection (selected/raw={selected_ratio:.4f}).")
        elif raw == 0:
            findings.append("Window diagnostics reports zero raw candidates.")

    return findings, details


def _new_version_risk_checks(summary: dict[str, Any], rows: list[dict[str, Any]], matrix_script_path: Path) -> list[str]:
    findings: list[str] = []
    text = matrix_script_path.read_text(encoding="utf-8")
    if "def _pretrain_rank_key" in text and "return (-val_f1, -val_acc, -test_f1)" in text:
        findings.append("Ranking rule is explicitly stable: val_f1 -> val_acc -> test_f1.")
    else:
        findings.append("Ranking implementation not found or changed; verify ranking stability manually.")

    if "repr_method_matrix_failure_report.json" in text and "next_command" in text:
        findings.append("Failure report path and next_command are present in matrix script.")
    else:
        findings.append("Failure report contract may be incomplete (missing failure report/next_command markers).")

    has_alias_in_rows = False
    if rows:
        probe = rows[0]
        has_alias_in_rows = (
            _pick(probe, "phase", "round") is not None
            and _pick(probe, "candidate", "name") is not None
            and _pick(probe, "run_id", "pretrain_run_id") is not None
            and _pick(probe, "checkpoint_path", "encoder_checkpoint_path") is not None
        )
    if has_alias_in_rows:
        findings.append("Summary rows already satisfy old/new compatible aliases.")
    else:
        findings.append("Summary rows are missing at least one compatibility alias pair.")

    best = summary.get("best_pretrain_run") if isinstance(summary.get("best_pretrain_run"), dict) else {}
    if best and (
        _pick(best, "run_id", "pretrain_run_id") is not None
        and _pick(best, "checkpoint_path", "encoder_checkpoint_path") is not None
    ):
        findings.append("best_pretrain_run contains compatible run/checkpoint identifiers.")
    else:
        findings.append("best_pretrain_run lacks compatible run/checkpoint identifiers.")
    return findings


def _render_markdown(report: dict[str, Any]) -> str:
    a_findings = report["sections"]["A_parse_schema_findings"]
    b_findings = report["sections"]["B_real_training_findings"]
    c_findings = report["sections"]["C_new_version_risk_audit"]
    best = report.get("best_run", {})
    return "\n".join(
        [
            "# DB5 Repr Matrix Diagnosis",
            "",
            "## Summary",
            f"- summary_path: `{report.get('summary_path', '')}`",
            f"- best_run_id: `{best.get('run_id', '')}`",
            f"- best_val_f1: `{_as_float(best.get('pretrain_best_val_macro_f1'), 0.0):.6f}`",
            f"- best_test_f1: `{_as_float(best.get('pretrain_test_macro_f1'), 0.0):.6f}`",
            "",
            "## A. 解析口径问题清单",
            *([f"- {item}" for item in a_findings] or ["- 无"]),
            "",
            "## B. 真实训练问题清单",
            *([f"- {item}" for item in b_findings] or ["- 无"]),
            "",
            "## C. 新版同类风险排查",
            *([f"- {item}" for item in c_findings] or ["- 无"]),
            "",
            "## Decision",
            f"- retrain_recommended: `{report.get('retrain_recommended', False)}`",
            f"- recommendation: `{report.get('recommendation', '')}`",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose DB5 repr matrix artifacts without retraining.")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--summary_json", default=None, help="Path to db5_repr_method_matrix_summary.json")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    summary_path = Path(args.summary_json) if args.summary_json else _find_latest_summary(run_root)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    summary = _load_json(summary_path)
    rows = list(summary.get("rows", []) or [])
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("summary.rows must be a list of objects")
    rows = [dict(row) for row in rows]

    best_run, best_warnings = _resolve_best(summary, rows)
    rows_metrics = _build_rows_metrics(rows)
    parse_schema_findings = _schema_findings(summary, rows, best_run)
    parse_schema_findings.extend(best_warnings)

    real_training_findings = _model_findings(best_run, rows_metrics)
    artifact_details: dict[str, Any] = {}
    if best_run.get("run_id"):
        artifact_findings, artifact_details = _artifact_findings(run_root, str(best_run["run_id"]))
        real_training_findings.extend(artifact_findings)
    else:
        real_training_findings.append("Best run id is unresolved; artifact-level diagnosis is unavailable.")

    matrix_script_path = Path(__file__).resolve().parent / "pretrain_db5_repr_method_matrix.py"
    new_version_risk_audit = _new_version_risk_checks(summary, rows, matrix_script_path)

    retrain_recommended = any("weak overall" in item for item in real_training_findings)
    recommendation = (
        "Fix data/signal bottlenecks first, then rerun stage-2 matrix."
        if retrain_recommended
        else "Schema/compat looks healthy; retrain only if target metric requires higher F1."
    )

    report = {
        "status": "completed",
        "generated_at_unix": int(time.time()),
        "summary_path": str(summary_path),
        "best_run": best_run,
        "rows_metrics": rows_metrics,
        "artifact_details": artifact_details,
        "sections": {
            "A_parse_schema_findings": parse_schema_findings,
            "B_real_training_findings": real_training_findings,
            "C_new_version_risk_audit": new_version_risk_audit,
        },
        "retrain_recommended": bool(retrain_recommended),
        "recommendation": recommendation,
    }

    default_json = summary_path.parent / "db5_repr_diagnosis.json"
    default_md = summary_path.parent / "db5_repr_diagnosis.md"
    out_json = Path(args.output_json) if args.output_json else default_json
    out_md = Path(args.output_md) if args.output_md else default_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"[DB5-DIAG] summary={summary_path}")
    print(f"[DB5-DIAG] json={out_json}")
    print(f"[DB5-DIAG] md={out_md}")


if __name__ == "__main__":
    main()

