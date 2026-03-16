from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.evaluate_event_dualtrack as dualtrack


def _prepare_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    run_root = tmp_path / "runs"
    run_dir = run_root / "demo_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")

    model_cfg = SimpleNamespace(num_classes=0)
    data_cfg = SimpleNamespace(
        target_db5_keys=[],
        recordings_manifest_path="s2_train_manifest_relax12.csv",
        split_manifest_path="artifacts/splits/s2_relax12_4class_seed42_v2.json",
        label_mode="event_onset",
    )
    runtime_cfg = SimpleNamespace(
        data=SimpleNamespace(target_db5_keys=[], recordings_manifest_path="s2_train_manifest_relax12.csv"),
        inference=SimpleNamespace(),
        runtime=SimpleNamespace(),
        actuation_mapping_path="configs/event_actuation_mapping_demo_latch.yaml",
        model_path="models/event_onset.mindir",
        model_metadata_path="models/event_onset.model_metadata.json",
    )
    label_spec = SimpleNamespace(class_names=["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW", "WRIST_CCW"])

    class _Predictor:
        def __init__(self, **_kwargs):
            self.metadata = SimpleNamespace(class_names=list(label_spec.class_names))

        def predict_proba(self, *_args, **_kwargs):
            return np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    class _Loader:
        def __init__(self, *_args, **_kwargs):
            pass

        def load_all_with_sources(self):
            emg = np.zeros((2, 8, 24, 3), dtype=np.float32)
            imu = np.zeros((2, 6, 16), dtype=np.float32)
            labels = np.asarray([0, 1], dtype=np.int32)
            sources = np.asarray(["clip_a", "clip_b"], dtype=object)
            return emg, imu, labels, sources

        def iter_clips(self):
            rows = np.zeros((1200, 14), dtype=np.float32)
            yield "RELAX", "RELAX", rows, {"relative_path": "clip_a"}
            yield "RELAX", "TENSE_OPEN", rows, {"relative_path": "clip_b"}

    monkeypatch.setattr(dualtrack, "load_event_training_config", lambda *_: (model_cfg, data_cfg, None, None))
    monkeypatch.setattr(dualtrack, "load_event_runtime_config", lambda *_: runtime_cfg)
    monkeypatch.setattr(dualtrack, "get_label_mode_spec", lambda *_: label_spec)
    monkeypatch.setattr(
        dualtrack,
        "load_and_validate_actuation_map",
        lambda *_args, **_kwargs: ({0: "RELAX", 1: "RELAX", 2: "THUMB_UP", 3: "WRIST_CW", 4: "WRIST_CCW"}, {
            "RELAX": "RELAX",
            "TENSE_OPEN": "RELAX",
            "THUMB_UP": "THUMB_UP",
            "WRIST_CW": "WRIST_CW",
            "WRIST_CCW": "WRIST_CCW",
        }),
    )
    monkeypatch.setattr(dualtrack, "EventPredictor", lambda **kwargs: _Predictor(**kwargs))
    monkeypatch.setattr(dualtrack, "EventClipDatasetLoader", _Loader)
    monkeypatch.setattr(dualtrack, "load_manifest", lambda *_: SimpleNamespace(test_sources={"clip_a", "clip_b"}))
    monkeypatch.setattr(dualtrack, "_resolve_split_manifest", lambda **_kwargs: split_path)
    monkeypatch.setattr(
        dualtrack,
        "_evaluate_window_metrics",
        lambda **_kwargs: {
            "accuracy": 0.94,
            "macro_f1": 0.93,
            "event_action_accuracy": 0.91,
            "event_action_macro_f1": 0.90,
            "gate_accept_rate": 0.0,
            "gate_action_recall": 0.0,
            "stage2_action_acc": 0.0,
            "rule_hit_rate": 0.0,
            "top_confusion_pairs": [],
        },
    )
    monkeypatch.setattr(
        dualtrack,
        "_evaluate_control_metrics",
        lambda **_kwargs: {
            "command_success_rate": 0.92,
            "false_release_rate": 0.03,
            "false_trigger_rate": 0.02,
        },
    )
    return run_root


def test_model_only_mode_outputs_model_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_root = _prepare_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(dualtrack, "EventAlgoPredictor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("algo should not run")))

    output_json = tmp_path / "model_only_summary.json"
    argv = [
        "evaluate_event_dualtrack.py",
        "--run_root",
        str(run_root),
        "--model_run_id",
        "demo_run",
        "--eval_mode",
        "model_only",
        "--output_json",
        str(output_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    dualtrack.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["eval_mode"] == "model_only"
    assert "model" in payload["tracks"]
    assert "algo" not in payload["tracks"]
    assert payload["recommended_backend"] == "model"
    assert payload["tracks"]["model"]["pass_strict_online_gate"] is True


def test_dualtrack_requires_algo_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_root = _prepare_stubs(monkeypatch, tmp_path)
    argv = [
        "evaluate_event_dualtrack.py",
        "--run_root",
        str(run_root),
        "--model_run_id",
        "demo_run",
        "--eval_mode",
        "dualtrack",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="requires --algo_model_path"):
        dualtrack.main()
