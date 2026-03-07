"""Unit tests for training split-manifest fallback behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data.split_strategy import build_manifest, save_manifest
from training.train import _prepare_split_manifest


def _mock_labels_and_sources():
    labels = []
    source_ids = []
    for class_id in range(6):
        for source_idx in range(5):
            source = f"class_{class_id}/file_{source_idx}.csv"
            for _ in range(2):
                labels.append(class_id)
                source_ids.append(source)
    return (
        np.asarray(labels, dtype=np.int32),
        np.asarray(source_ids, dtype=object),
    )


def test_config_manifest_missing_auto_generates_and_saves(tmp_path: Path):
    labels, source_ids = _mock_labels_and_sources()
    manifest_path = tmp_path / "artifacts" / "splits" / "default_split_manifest.json"

    manifest = _prepare_split_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        split_seed=42,
        data_dir=str(tmp_path),
        cli_manifest_in=None,
        config_manifest_path=str(manifest_path),
        manifest_out=None,
    )

    assert manifest_path.exists()
    assert manifest.num_samples == len(labels)
    assert manifest.split_mode == "grouped_file"


def test_explicit_manifest_missing_raises_file_not_found(tmp_path: Path):
    labels, source_ids = _mock_labels_and_sources()
    missing_manifest = tmp_path / "missing_manifest.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        _prepare_split_manifest(
            labels=labels,
            source_ids=source_ids,
            split_mode="grouped_file",
            val_ratio=0.2,
            test_ratio=0.2,
            split_seed=42,
            data_dir=str(tmp_path),
            cli_manifest_in=str(missing_manifest),
            config_manifest_path=None,
            manifest_out=None,
        )

    message = str(exc_info.value)
    assert "--split_manifest_out <path>" in message
    assert "--split_manifest_in <path>" in message


def test_existing_manifest_loads_without_rebuild(tmp_path: Path):
    labels, source_ids = _mock_labels_and_sources()
    manifest_path = tmp_path / "existing_manifest.json"

    existing_manifest = build_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
        data_dir=str(tmp_path),
    )
    save_manifest(existing_manifest, manifest_path)

    loaded_manifest = _prepare_split_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        split_seed=999,
        data_dir=str(tmp_path),
        cli_manifest_in=None,
        config_manifest_path=str(manifest_path),
        manifest_out=None,
    )

    assert loaded_manifest.seed == 7
    assert loaded_manifest.train_indices == existing_manifest.train_indices
    assert loaded_manifest.val_indices == existing_manifest.val_indices
    assert loaded_manifest.test_indices == existing_manifest.test_indices
