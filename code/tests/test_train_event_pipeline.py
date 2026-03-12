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


def test_one_click_pipeline_writes_final_outputs(monkeypatch, tmp_path: Path):
    code_root = tmp_path / "code"
    code_root.mkdir(parents=True, exist_ok=True)
    run_root = code_root / "artifacts" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "CODE_ROOT", code_root)

    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fake_run_step(step_name: str, command: list[str], *, cwd: Path) -> None:
        assert cwd == code_root
        run_id = "pipeline_test"
        if step_name == "pretrain_aligned3":
            _write_json(
                run_root / f"{run_id}_db5_pretrain" / "offline_summary.json",
                {
                    "checkpoint_path": str(run_root / "db5_pretrain.ckpt"),
                    "test_macro_f1": 0.10,
                    "test_accuracy": 0.20,
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
        if step_name == "preflight":
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
            "../data",
        ],
    )

    pipeline.main()

    final_selection = run_root / "pipeline_test" / "final_selection.json"
    final_artifacts = run_root / "pipeline_test" / "final_artifacts.json"
    assert final_selection.exists()
    assert final_artifacts.exists()

    selection_payload = json.loads(final_selection.read_text(encoding="utf-8"))
    artifacts_payload = json.loads(final_artifacts.read_text(encoding="utf-8"))
    assert selection_payload["selected_finetune_variant"] == "scratch"
    assert artifacts_payload["merge_gate_passed"] is True
