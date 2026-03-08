from pathlib import Path

import numpy as np
import pytest

from training.data.split_strategy import build_manifest, save_manifest
from training.train import _prepare_manifest


def _fake_data():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    source_ids = np.array(
        [
            "RELAX/rec_a.csv",
            "RELAX/rec_a.csv",
            "FIST/rec_b.csv",
            "FIST/rec_b.csv",
            "PINCH/rec_c.csv",
            "PINCH/rec_c.csv",
        ],
        dtype=object,
    )
    metadata = [
        {"recording_id": "rec_a", "session_id": "s1"},
        {"recording_id": "rec_a", "session_id": "s1"},
        {"recording_id": "rec_b", "session_id": "s1"},
        {"recording_id": "rec_b", "session_id": "s1"},
        {"recording_id": "rec_c", "session_id": "s2"},
        {"recording_id": "rec_c", "session_id": "s2"},
    ]
    return labels, source_ids, metadata


def test_config_manifest_missing_auto_generates(tmp_path: Path):
    labels, source_ids, metadata = _fake_data()
    manifest_path = tmp_path / "artifacts/splits/default_split_manifest.json"
    class_names = ["RELAX", "FIST", "PINCH"]

    manifest, path = _prepare_manifest(
        labels=labels,
        source_ids=source_ids,
        source_metadata=metadata,
        seed=42,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        class_names=class_names,
        manifest_in_cli=None,
        manifest_in_config=str(manifest_path),
        manifest_out_cli=None,
        manifest_strategy="v2",
    )
    assert manifest.manifest_strategy == "v2"
    assert path == str(manifest_path)
    assert manifest_path.exists()


def test_explicit_manifest_missing_raises(tmp_path: Path):
    labels, source_ids, metadata = _fake_data()
    class_names = ["RELAX", "FIST", "PINCH"]
    missing_path = tmp_path / "no_manifest.json"

    with pytest.raises(FileNotFoundError, match="--split_manifest_in"):
        _prepare_manifest(
            labels=labels,
            source_ids=source_ids,
            source_metadata=metadata,
            seed=42,
            split_mode="grouped_file",
            val_ratio=0.2,
            test_ratio=0.2,
            class_names=class_names,
            manifest_in_cli=str(missing_path),
            manifest_in_config=None,
            manifest_out_cli=None,
            manifest_strategy="v2",
        )


def test_existing_manifest_loads_without_rebuild(tmp_path: Path):
    labels, source_ids, metadata = _fake_data()
    class_names = ["RELAX", "FIST", "PINCH"]
    manifest = build_manifest(
        labels,
        source_ids,
        seed=42,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        num_classes=len(class_names),
        class_names=class_names,
        manifest_strategy="v2",
        source_metadata=metadata,
    )
    path = save_manifest(manifest, str(tmp_path / "manifest.json"))

    loaded, chosen = _prepare_manifest(
        labels=labels,
        source_ids=source_ids,
        source_metadata=metadata,
        seed=1,
        split_mode="grouped_file",
        val_ratio=0.3,
        test_ratio=0.3,
        class_names=class_names,
        manifest_in_cli=None,
        manifest_in_config=str(path),
        manifest_out_cli=None,
        manifest_strategy="v2",
    )
    assert chosen == str(path)
    assert loaded.train_indices == manifest.train_indices
    assert loaded.val_indices == manifest.val_indices
    assert loaded.test_indices == manifest.test_indices

