from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_event_runtime import _validate_release_contract, _validate_startup_artifacts


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")
    return path


def test_validate_startup_artifacts_model_ckpt_requires_checkpoint(tmp_path: Path):
    ckpt = _touch(tmp_path / "event.ckpt")
    resolved = _validate_startup_artifacts(
        recognizer_backend="model",
        model_backend="ckpt",
        checkpoint_path=ckpt,
        model_path=tmp_path / "unused.mindir",
        model_metadata_path=tmp_path / "unused.json",
        algo_model_path=None,
    )
    assert resolved["checkpoint_path"] == str(ckpt)


def test_validate_startup_artifacts_model_lite_requires_model_and_metadata(tmp_path: Path):
    mindir = _touch(tmp_path / "event.mindir")
    metadata = _touch(tmp_path / "event.model_metadata.json")
    resolved = _validate_startup_artifacts(
        recognizer_backend="model",
        model_backend="lite",
        checkpoint_path=tmp_path / "unused.ckpt",
        model_path=mindir,
        model_metadata_path=metadata,
        algo_model_path=None,
    )
    assert resolved["model_path"] == str(mindir)
    assert resolved["model_metadata_path"] == str(metadata)


def test_validate_startup_artifacts_algo_requires_algo_model(tmp_path: Path):
    with pytest.raises(ValueError, match="--algo_model_path is required"):
        _validate_startup_artifacts(
            recognizer_backend="algo",
            model_backend="lite",
            checkpoint_path=tmp_path / "unused.ckpt",
            model_path=tmp_path / "unused.mindir",
            model_metadata_path=tmp_path / "unused.json",
            algo_model_path=None,
        )

    algo_path = _touch(tmp_path / "algo_model.json")
    resolved = _validate_startup_artifacts(
        recognizer_backend="algo",
        model_backend="lite",
        checkpoint_path=tmp_path / "unused.ckpt",
        model_path=tmp_path / "unused.mindir",
        model_metadata_path=tmp_path / "unused.json",
        algo_model_path=algo_path,
    )
    assert resolved["algo_model_path"] == str(algo_path)


def test_validate_startup_artifacts_fails_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="MindIR model not found"):
        _validate_startup_artifacts(
            recognizer_backend="model",
            model_backend="lite",
            checkpoint_path=tmp_path / "unused.ckpt",
            model_path=tmp_path / "missing.mindir",
            model_metadata_path=_touch(tmp_path / "event.model_metadata.json"),
            algo_model_path=None,
        )


def test_validate_release_contract_requires_tense_open_for_command_only():
    with pytest.raises(ValueError, match="requires class TENSE_OPEN"):
        _validate_release_contract(
            release_mode="command_only",
            class_names=["RELAX", "THUMB_UP"],
            mapping_by_name={"RELAX": "RELAX", "THUMB_UP": "THUMB_UP"},
        )

    with pytest.raises(ValueError, match="requires mapping TENSE_OPEN -> RELAX"):
        _validate_release_contract(
            release_mode="command_only",
            class_names=["RELAX", "TENSE_OPEN", "THUMB_UP"],
            mapping_by_name={"RELAX": "RELAX", "TENSE_OPEN": "THUMB_UP", "THUMB_UP": "THUMB_UP"},
        )

    _validate_release_contract(
        release_mode="command_only",
        class_names=["RELAX", "TENSE_OPEN", "THUMB_UP"],
        mapping_by_name={"RELAX": "RELAX", "TENSE_OPEN": "RELAX", "THUMB_UP": "THUMB_UP"},
    )
