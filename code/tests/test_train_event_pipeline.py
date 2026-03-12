from __future__ import annotations

import json
from pathlib import Path

from scripts import train_event_pipeline as pipeline


def test_select_best_finetune_prefers_scratch_when_pretrained_not_better():
    scratch = {"checkpoint_path": "scratch.ckpt", "test_macro_f1": 0.42, "test_accuracy": 0.70}
    pretrained = {"checkpoint_path": "pre.ckpt", "test_macro_f1": 0.42, "test_accuracy": 0.70}

    selected = pipeline.select_best_finetune(scratch, pretrained)

    assert selected.variant == "scratch"
    assert selected.checkpoint_path == "scratch.ckpt"


def test_select_best_finetune_prefers_pretrained_when_strictly_better():
    scratch = {"checkpoint_path": "scratch.ckpt", "test_macro_f1": 0.35, "test_accuracy": 0.60}
    pretrained = {"checkpoint_path": "pre.ckpt", "test_macro_f1": 0.36, "test_accuracy": 0.58}

    selected = pipeline.select_best_finetune(scratch, pretrained)

    assert selected.variant == "pretrained"
    assert selected.checkpoint_path == "pre.ckpt"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_one_click_pipeline_auto_builds_foundation_when_missing(monkeypatch, tmp_path: Path):
    code_root = tmp_path / "code"
    code_root.mkdir(parents=True, exist_ok=True)
    run_root = code_root / "artifacts" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    foundation_dir = code_root / "artifacts" / "foundation" / "db5_full53"

    monkeypatch.setattr(pipeline, "CODE_ROOT", code_root)
    steps: list[str] = []

    def _fake_run_step(step_name: str, command: list[str], *, cwd: Path) -> None:
        assert cwd == code_root
        steps.append(step_name)
        run_id = "pipeline_test"

        if step_name == "preflight":
            return
        if step_name == "build_foundation":
            ckpt = foundation_dir / "checkpoints" / "db5_full53_foundation.ckpt"
            _write_file(ckpt, "foundation")
            _write_json(
                foundation_dir / "foundation_manifest.json",
                {
                    "checkpoint_path": str(ckpt),
                    "foundation_version": "db5_full53_v1",
                },
            )
            return
        if step_name == "finetune_scratch":
            _write_json(
                run_root / f"{run_id}_finetune_scratch" / "offline_summary.json",
                {
                    "checkpoint_path": str(run_root / "scratch_best.ckpt"),
                    "test_macro_f1": 0.45,
                    "test_accuracy": 0.72,
                },
            )
            return
        if step_name == "finetune_pretrained":
            _write_json(
                run_root / f"{run_id}_finetune_pretrained" / "offline_summary.json",
                {
                    "checkpoint_path": str(run_root / "pretrained_best.ckpt"),
                    "test_macro_f1": 0.44,
                    "test_accuracy": 0.71,
                },
            )
            return
        if step_name == "convert":
            _write_json(
                run_root / f"{run_id}_convert" / "conversion" / "event_conversion_summary.json",
                {
                    "output_path": str(run_root / "event_onset_selected.mindir"),
                    "metadata_path": str(run_root / "event_onset_selected.model_metadata.json"),
                },
            )
            return
        if step_name == "benchmark":
            output_index = command.index("--output") + 1
            output_path = Path(command[output_index])
            _write_json(
                output_path,
                {
                    "merge_gate": {
                        "passed": True,
                        "checks": {
                            "transition_hit_rate": True,
                            "false_trigger_rate": True,
                            "state_hold_accuracy": True,
                            "release_accuracy": True,
                            "latency_p95_ms": True,
                        },
                    }
                },
            )
            return
        raise AssertionError(f"Unexpected step_name={step_name}")

    monkeypatch.setattr(pipeline, "_run_step", _fake_run_step)
    monkeypatch.setattr(
        pipeline.sys,
        "argv",
        [
            "train_event_pipeline.py",
            "--run_id",
            "pipeline_test",
            "--run_root",
            "artifacts/runs",
            "--db5_data_dir",
            "../data_ninaproDB5",
            "--wearer_data_dir",
            "../data_event_onset",
            "--target_db5_keys",
            "E1_G01,E1_G02",
        ],
    )

    pipeline.main()

    final_selection = run_root / "pipeline_test" / "final_selection.json"
    final_artifacts = run_root / "pipeline_test" / "final_artifacts.json"
    assert final_selection.exists()
    assert final_artifacts.exists()
    assert "build_foundation" in steps

    selection_payload = json.loads(final_selection.read_text(encoding="utf-8"))
    artifacts_payload = json.loads(final_artifacts.read_text(encoding="utf-8"))
    assert selection_payload["selected_finetune_variant"] == "scratch"
    assert artifacts_payload["merge_gate_passed"] is True


def test_one_click_pipeline_reuses_existing_foundation_without_build(monkeypatch, tmp_path: Path):
    code_root = tmp_path / "code"
    code_root.mkdir(parents=True, exist_ok=True)
    run_root = code_root / "artifacts" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    foundation_dir = code_root / "artifacts" / "foundation" / "db5_full53"

    ckpt = foundation_dir / "checkpoints" / "db5_full53_foundation.ckpt"
    _write_file(ckpt, "foundation")
    _write_json(
        foundation_dir / "foundation_manifest.json",
        {
            "checkpoint_path": str(ckpt),
            "foundation_version": "db5_full53_v1",
        },
    )

    monkeypatch.setattr(pipeline, "CODE_ROOT", code_root)
    steps: list[str] = []

    def _fake_run_step(step_name: str, command: list[str], *, cwd: Path) -> None:
        assert cwd == code_root
        steps.append(step_name)
        run_id = "pipeline_reuse"
        if step_name == "preflight":
            return
        if step_name == "finetune_scratch":
            _write_json(
                run_root / f"{run_id}_finetune_scratch" / "offline_summary.json",
                {
                    "checkpoint_path": str(run_root / "scratch_best.ckpt"),
                    "test_macro_f1": 0.45,
                    "test_accuracy": 0.72,
                },
            )
            return
        if step_name == "finetune_pretrained":
            _write_json(
                run_root / f"{run_id}_finetune_pretrained" / "offline_summary.json",
                {
                    "checkpoint_path": str(run_root / "pretrained_best.ckpt"),
                    "test_macro_f1": 0.46,
                    "test_accuracy": 0.73,
                },
            )
            return
        if step_name == "convert":
            _write_json(
                run_root / f"{run_id}_convert" / "conversion" / "event_conversion_summary.json",
                {
                    "output_path": str(run_root / "event_onset_selected.mindir"),
                    "metadata_path": str(run_root / "event_onset_selected.model_metadata.json"),
                },
            )
            return
        if step_name == "benchmark":
            output_index = command.index("--output") + 1
            output_path = Path(command[output_index])
            _write_json(
                output_path,
                {
                    "merge_gate": {
                        "passed": True,
                        "checks": {
                            "transition_hit_rate": True,
                            "false_trigger_rate": True,
                            "state_hold_accuracy": True,
                            "release_accuracy": True,
                            "latency_p95_ms": True,
                        },
                    }
                },
            )
            return
        raise AssertionError(f"Unexpected step_name={step_name}")

    monkeypatch.setattr(pipeline, "_run_step", _fake_run_step)
    monkeypatch.setattr(
        pipeline.sys,
        "argv",
        [
            "train_event_pipeline.py",
            "--run_id",
            "pipeline_reuse",
            "--run_root",
            "artifacts/runs",
            "--db5_data_dir",
            "../data_ninaproDB5",
            "--wearer_data_dir",
            "../data_event_onset",
            "--target_db5_keys",
            "E1_G01,E1_G02",
        ],
    )

    pipeline.main()

    assert "build_foundation" not in steps
    final_selection = json.loads((run_root / "pipeline_reuse" / "final_selection.json").read_text(encoding="utf-8"))
    assert final_selection["selected_finetune_variant"] == "pretrained"
