from __future__ import annotations
from pathlib import Path

from scripts.diagnose_db5_repr_results import (
    _new_version_risk_checks,
    _normalize_row,
    _resolve_best,
    _schema_findings,
)


def test_normalize_row_accepts_new_schema() -> None:
    row = _normalize_row(
        {
            "phase": "run_2",
            "candidate": "temp_0p1",
            "run_id": "r2",
            "checkpoint_path": "ckpt.ckpt",
            "pretrain_best_val_macro_f1": 0.1,
            "pretrain_best_val_acc": 0.2,
            "pretrain_test_macro_f1": 0.09,
        }
    )
    assert row["phase"] == "run_2"
    assert row["candidate"] == "temp_0p1"
    assert row["run_id"] == "r2"
    assert row["checkpoint_path"] == "ckpt.ckpt"


def test_normalize_row_accepts_old_schema_aliases() -> None:
    row = _normalize_row(
        {
            "round": "run_3",
            "name": "proj_256",
            "pretrain_run_id": "r3",
            "encoder_checkpoint_path": "encoder.ckpt",
            "pretrain_best_val_macro_f1": 0.2,
            "pretrain_best_val_acc": 0.21,
            "pretrain_test_macro_f1": 0.19,
        }
    )
    assert row["phase"] == "run_3"
    assert row["candidate"] == "proj_256"
    assert row["run_id"] == "r3"
    assert row["checkpoint_path"] == "encoder.ckpt"


def test_resolve_best_falls_back_to_rows_when_best_missing() -> None:
    summary = {"best_pretrain_run": {}, "rows": []}
    rows = [
        {
            "phase": "run_1",
            "candidate": "baseline",
            "run_id": "r1",
            "checkpoint_path": "r1.ckpt",
            "pretrain_best_val_macro_f1": 0.02,
            "pretrain_best_val_acc": 0.02,
            "pretrain_test_macro_f1": 0.02,
        },
        {
            "phase": "run_2",
            "candidate": "temp_0p1",
            "run_id": "r2",
            "checkpoint_path": "r2.ckpt",
            "pretrain_best_val_macro_f1": 0.03,
            "pretrain_best_val_acc": 0.03,
            "pretrain_test_macro_f1": 0.025,
        },
    ]
    best, warnings = _resolve_best(summary, rows)
    assert best["run_id"] == "r2"
    assert best["source"] == "fallback.rows_ranking"
    assert warnings


def test_schema_findings_reports_missing_best_id() -> None:
    summary = {"best_pretrain_run": {"run_id": "r1"}, "rows": []}
    rows: list[dict] = []
    findings = _schema_findings(summary, rows, {"run_id": "r1"})
    assert any("best_pretrain_run_id missing" in item for item in findings)
    assert any("rows is empty" in item for item in findings)


def test_new_version_risk_checks_detects_alias_presence(tmp_path: Path) -> None:
    script_path = tmp_path / "matrix.py"
    script_path.write_text(
        "\n".join(
            [
                "def _pretrain_rank_key(row):",
                "    val_f1 = 0",
                "    val_acc = 0",
                "    test_f1 = 0",
                "    return (-val_f1, -val_acc, -test_f1)",
                "def x():",
                "    pass",
                'path = "repr_method_matrix_failure_report.json"',
                'k = "next_command"',
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "best_pretrain_run": {"run_id": "r1", "checkpoint_path": "r1.ckpt"},
    }
    rows = [
        {
            "phase": "run_1",
            "round": "run_1",
            "candidate": "baseline",
            "name": "baseline",
            "run_id": "r1",
            "pretrain_run_id": "r1",
            "checkpoint_path": "r1.ckpt",
            "encoder_checkpoint_path": "r1.ckpt",
        }
    ]
    findings = _new_version_risk_checks(summary, rows, script_path)
    assert any("Ranking rule is explicitly stable" in item for item in findings)
    assert any("Summary rows already satisfy old/new compatible aliases." in item for item in findings)
