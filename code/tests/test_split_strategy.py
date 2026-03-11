"""Unit tests for split strategy and augmentation boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data.split_strategy import (
    SplitManifest,
    build_manifest,
    load_manifest,
    split_and_optionally_augment,
    split_arrays_from_manifest,
)


def _make_grouped_mock_dataset(samples_per_source: int = 3):
    samples = []
    labels = []
    source_ids = []

    for class_id in range(6):
        for source_idx in range(5):
            source_id = f"class_{class_id}/file_{source_idx}.csv"
            for local_idx in range(samples_per_source):
                # Sample value encodes class/source/sample index for easy assertions.
                value = class_id * 1000 + source_idx * 100 + local_idx
                samples.append(np.full((1, 1, 1), value, dtype=np.float32))
                labels.append(class_id)
                source_ids.append(source_id)

    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(source_ids, dtype=object),
    )


def test_grouped_file_split_has_no_source_leakage():
    samples, labels, source_ids = _make_grouped_mock_dataset()
    manifest = build_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        data_dir="mock_data",
    )

    train_src = set(manifest.train_sources)
    val_src = set(manifest.val_sources)
    test_src = set(manifest.test_sources)

    assert train_src.isdisjoint(val_src)
    assert train_src.isdisjoint(test_src)
    assert val_src.isdisjoint(test_src)

    # Source IDs reconstructed from split indices should match manifest source sets.
    assert set(source_ids[manifest.train_indices].tolist()) == train_src
    assert set(source_ids[manifest.val_indices].tolist()) == val_src
    assert set(source_ids[manifest.test_indices].tolist()) == test_src

    # Index splits must form a partition.
    all_indices = set(manifest.train_indices + manifest.val_indices + manifest.test_indices)
    assert len(all_indices) == len(samples)


class _SpyAugmentor:
    def __init__(self):
        self.called = 0
        self.last_input_len = 0

    def augment_batch(self, samples, labels, factor=1, use_mixup=False):
        self.called += 1
        self.last_input_len = len(samples)
        # Return deterministic expanded train split to validate path.
        return np.concatenate([samples, samples], axis=0), np.concatenate([labels, labels], axis=0)


def test_augmentation_only_applies_to_train_split():
    samples, labels, source_ids = _make_grouped_mock_dataset(samples_per_source=2)
    manifest = build_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode="grouped_file",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )

    (train_raw, train_labels_raw), (val_raw, val_labels_raw), (test_raw, test_labels_raw) = split_arrays_from_manifest(
        samples,
        labels,
        manifest,
    )

    spy = _SpyAugmentor()
    (train_aug, train_labels_aug), (val_after, val_labels_after), (test_after, test_labels_after) = split_and_optionally_augment(
        samples=samples,
        labels=labels,
        manifest=manifest,
        augmentor=spy,
        augment_factor=2,
        use_mixup=False,
    )

    assert spy.called == 1
    assert spy.last_input_len == len(train_raw)

    # Train split changed by augmentation.
    assert len(train_aug) == len(train_raw) * 2
    assert len(train_labels_aug) == len(train_labels_raw) * 2

    # Val/Test remain untouched.
    np.testing.assert_array_equal(val_after, val_raw)
    np.testing.assert_array_equal(val_labels_after, val_labels_raw)
    np.testing.assert_array_equal(test_after, test_raw)
    np.testing.assert_array_equal(test_labels_after, test_labels_raw)


def test_manifest_v2_group_key_separates_same_session_recording_across_users():
    labels = np.array([0, 0], dtype=np.int32)
    source_ids = np.array(["RELAX/shared.csv", "RELAX/shared.csv"], dtype=object)
    metadata = [
        {"recording_id": "shared", "session_id": "s1", "user_id": "u1"},
        {"recording_id": "shared", "session_id": "s1", "user_id": "u2"},
    ]

    manifest = build_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode="grouped_file",
        val_ratio=0.0,
        test_ratio=0.5,
        seed=5,
        manifest_strategy="v2",
        source_metadata=metadata,
        num_classes=1,
        class_names=["RELAX"],
    )

    all_group_keys = set(manifest.group_keys_train + manifest.group_keys_val + manifest.group_keys_test)
    assert "u1::s1::shared" in all_group_keys
    assert "u2::s1::shared" in all_group_keys


def test_split_manifest_from_dict_ignores_legacy_version_field():
    manifest = SplitManifest.from_dict(
        {
            "train_indices": [],
            "val_indices": [],
            "test_indices": [],
            "train_sources": [],
            "val_sources": [],
            "test_sources": [],
            "seed": 42,
            "split_mode": "grouped_file",
            "version": "legacy-v1",
        }
    )

    assert manifest.split_mode == "grouped_file"
    assert manifest.manifest_strategy == "v1"


def test_load_manifest_accepts_legacy_version_field(tmp_path: Path):
    manifest_path = tmp_path / "legacy_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_indices": [],
                "val_indices": [],
                "test_indices": [],
                "train_sources": [],
                "val_sources": [],
                "test_sources": [],
                "seed": 42,
                "split_mode": "grouped_file",
                "version": "legacy-v1",
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(str(manifest_path))

    assert manifest.seed == 42
    assert manifest.split_mode == "grouped_file"
