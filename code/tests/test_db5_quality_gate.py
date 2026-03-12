from __future__ import annotations

import json

from scripts.pretrain_db5_quality_gate import (
    _count_log_markers,
    _group_leakage_present,
    _validate_smoke_outputs,
)


def test_count_log_markers_detects_warn_and_error_lines() -> None:
    output = "\n".join(
        [
            "[INFO] ok",
            "[WARN] warning_1",
            "[WARN] warning_2",
            "[ERROR] error_1",
        ]
    )
    warn_count, error_count = _count_log_markers(output)
    assert warn_count == 2
    assert error_count == 1


def test_group_leakage_present_uses_group_keys() -> None:
    clean = {
        "group_keys_train": ["A"],
        "group_keys_val": ["B"],
        "group_keys_test": ["C"],
    }
    leaked = {
        "group_keys_train": ["A", "B"],
        "group_keys_val": ["B"],
        "group_keys_test": ["C"],
    }
    assert _group_leakage_present(clean) is False
    assert _group_leakage_present(leaked) is True


def test_validate_smoke_outputs_requires_complete_non_empty_split(tmp_path) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "group_keys_train": ["U1::S1::R1"],
        "group_keys_val": ["U2::S2::R2"],
        "group_keys_test": ["U3::S3::R3"],
    }
    split_diag = {
        "overall": {"has_any_empty_classes": False},
        "by_split": {
            "train": {"class_counts": {"A": 1, "B": 2}},
            "val": {"class_counts": {"A": 1, "B": 1}},
            "test": {"class_counts": {"A": 2, "B": 1}},
        },
    }
    window_diag = {"totals": {"raw_candidates": 10, "selected": 6}}

    (smoke_dir / "db5_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (smoke_dir / "db5_split_diagnostics.json").write_text(json.dumps(split_diag), encoding="utf-8")
    (smoke_dir / "db5_window_diagnostics.json").write_text(json.dumps(window_diag), encoding="utf-8")

    assert _validate_smoke_outputs(smoke_dir) == []


def test_validate_smoke_outputs_reports_leakage_and_empty_classes(tmp_path) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "group_keys_train": ["U1::S1::R1", "U2::S2::R2"],
        "group_keys_val": ["U2::S2::R2"],
        "group_keys_test": ["U3::S3::R3"],
    }
    split_diag = {
        "overall": {"has_any_empty_classes": True},
        "by_split": {
            "train": {"class_counts": {"A": 1, "B": 0}},
            "val": {"class_counts": {"A": 1, "B": 1}},
            "test": {"class_counts": {"A": 2, "B": 1}},
        },
    }
    window_diag = {"totals": {"raw_candidates": 10, "selected": 6}}

    (smoke_dir / "db5_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (smoke_dir / "db5_split_diagnostics.json").write_text(json.dumps(split_diag), encoding="utf-8")
    (smoke_dir / "db5_window_diagnostics.json").write_text(json.dumps(window_diag), encoding="utf-8")

    issues = _validate_smoke_outputs(smoke_dir)
    joined = " | ".join(issues)
    assert "group leakage detected in manifest" in joined
    assert "split diagnostics reports empty classes" in joined
    assert "train has empty classes" in joined
