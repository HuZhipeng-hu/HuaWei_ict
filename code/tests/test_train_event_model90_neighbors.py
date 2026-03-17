from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from scripts.train_event_model_90_sprint import (
    _build_tune_cmd,
    _build_neighbor_candidates,
    _effective_prepare_session_id,
    _metric_or,
    _model90_split_manifest_path,
    _prepare_split_seeds,
    _reuse_trial_outputs_if_compatible,
)


def test_build_neighbor_candidates_has_reference_and_unique_grid() -> None:
    args = SimpleNamespace(
        neighbor_lr_delta_ratio=0.2,
        neighbor_freeze_delta=2,
    )
    reference = {
        "loss_type": "cross_entropy",
        "base_channels": 16,
        "freeze_emg_epochs": 5,
        "encoder_lr_ratio": 0.2,
        "pretrained_mode": "off",
    }

    rows = _build_neighbor_candidates(args, reference=reference)
    assert [row["variant"] for row in rows] == ["ref", "lr_down", "lr_up", "freeze_down", "freeze_up"]
    assert all(str(row["loss_type"]) == "cross_entropy" for row in rows)
    assert all(int(row["base_channels"]) == 16 for row in rows)

    keys = {
        (
            str(row["loss_type"]),
            int(row["base_channels"]),
            int(row["freeze_emg_epochs"]),
            float(row["encoder_lr_ratio"]),
            str(row["pretrained_mode"]),
        )
        for row in rows
    }
    assert len(keys) == len(rows)


def test_model90_split_manifest_path_uses_explicit_screen_manifest_only_for_screen_seed() -> None:
    args = SimpleNamespace(
        run_prefix="s2_model90",
        screen_split_manifest="artifacts/splits/custom_screen_seed42.json",
        screen_split_seed=42,
    )

    screen_path = _model90_split_manifest_path(args, 42).as_posix()
    longrun_path = _model90_split_manifest_path(args, 52).as_posix()

    assert screen_path.endswith("/artifacts/splits/custom_screen_seed42.json")
    assert longrun_path.endswith("/artifacts/splits/s2_model90_demo3_seed52_v2.json")


def test_prepare_split_seeds_unions_screen_longrun_and_neighbor() -> None:
    args = SimpleNamespace(
        screen_split_seed=42,
        longrun_seeds="52,62,42",
        neighbor_split_seed=52,
    )

    assert _prepare_split_seeds(args) == [42, 52, 62]


def test_effective_prepare_session_id_prefers_explicit_session_alias() -> None:
    args = SimpleNamespace(session_id="s3_demo3_0317", prepare_session_id="s2")
    assert _effective_prepare_session_id(args) == "s3_demo3_0317"

    args = SimpleNamespace(session_id="", prepare_session_id="s2")
    assert _effective_prepare_session_id(args) == "s2"


def test_metric_or_preserves_explicit_zero_values() -> None:
    assert _metric_or({"false_trigger_rate": 0.0}, "false_trigger_rate", default=1.0) == 0.0
    assert _metric_or({"false_release_rate": "0.0"}, "false_release_rate", default=1.0) == 0.0
    assert _metric_or({}, "false_release_rate", default=1.0) == 1.0


def test_build_tune_cmd_uses_prepared_manifest_when_available(tmp_path: Path) -> None:
    args = SimpleNamespace(
        run_root="artifacts/runs",
        training_config="configs/training_event_onset_demo3_two_stage.yaml",
        runtime_config="configs/runtime_event_onset_demo3_latch.yaml",
        data_dir="../data",
        recordings_manifest="../data/recordings_manifest.csv",
        target_db5_keys="TENSE_OPEN,THUMB_UP,WRIST_CW",
        control_backend="ckpt",
        device_target="Ascend",
        _prepared_recordings_manifest=str(tmp_path / "prepared_manifest.csv"),
    )

    cmd = _build_tune_cmd(
        args,
        best_run_id="demo_best",
        output_json=tmp_path / "summary.json",
        output_csv=tmp_path / "summary.csv",
        output_runtime_config=tmp_path / "runtime.yaml",
    )

    idx = cmd.index("--recordings_manifest")
    assert cmd[idx + 1] == str(tmp_path / "prepared_manifest.csv")


def test_reuse_trial_outputs_requires_matching_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    eval_dir = run_dir / "evaluation"
    snapshots_dir = run_dir / "config_snapshots"
    eval_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)

    actual_manifest = tmp_path / "prepared_manifest.csv"
    expected_manifest = tmp_path / "other_manifest.csv"
    split_manifest = tmp_path / "split.json"
    checkpoint = tmp_path / "best.ckpt"
    quality_report = tmp_path / "quality_report.json"
    history_csv = tmp_path / "history.csv"
    report_json = eval_dir / "test_report.json"
    report_csv = eval_dir / "test_metrics.csv"
    control_json = eval_dir / "control_eval_summary.json"

    for path in [
        actual_manifest,
        expected_manifest,
        split_manifest,
        checkpoint,
        quality_report,
        history_csv,
        report_json,
        report_csv,
        control_json,
    ]:
        path.write_text("{}", encoding="utf-8")

    (run_dir / "offline_summary.json").write_text(
        json.dumps({"manifest_path": str(split_manifest), "checkpoint_path": str(checkpoint)}),
        encoding="utf-8",
    )
    (eval_dir / "test_metrics.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
    (snapshots_dir / "effective_overrides.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  base_channels: 16",
                "  pretrained_emg_checkpoint: ''",
                "training:",
                "  loss_type: cross_entropy",
                "  freeze_emg_epochs: 5",
                "  encoder_lr_ratio: 0.3",
                "device:",
                "  device_target: CPU",
                "  device_id: 0",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "recordings_manifest_path": str(actual_manifest),
                "training_device": {"target": "CPU", "id": 0},
                "quality_report": str(quality_report),
                "training_history": str(history_csv),
                "evaluation_outputs": {
                    "json": str(report_json),
                    "csv": str(report_csv),
                },
                "class_names": ["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW", "WRIST_CCW"],
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        skip_control_eval=False,
        data_dir=str(tmp_path),
        device_target="CPU",
        device_id=0,
        pretrained_emg_checkpoint="",
    )

    assert (
        _reuse_trial_outputs_if_compatible(
            args,
            run_dir=run_dir,
            recordings_manifest_path=str(expected_manifest),
            split_manifest_path=str(split_manifest),
            loss_type="cross_entropy",
            base_channels=16,
            freeze_emg_epochs=5,
            encoder_lr_ratio=0.3,
            pretrained_mode="off",
        )
        is False
    )
