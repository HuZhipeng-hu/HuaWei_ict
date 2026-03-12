from __future__ import annotations

import numpy as np
import pytest

from event_onset.train_pipeline import _apply_train_budget_to_manifest
from training.data.split_strategy import SplitManifest


def _build_manifest(num_samples: int) -> SplitManifest:
    indices = list(range(num_samples))
    return SplitManifest(
        train_indices=indices,
        val_indices=[],
        test_indices=[],
        train_sources=[f"src_{idx}" for idx in indices],
        val_sources=[],
        test_sources=[],
        seed=42,
        split_mode="grouped_file",
        manifest_strategy="v2",
        num_samples=num_samples,
    )


def test_budget_sampling_is_reproducible_per_seed():
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int32)
    manifest = _build_manifest(len(labels))
    class_names = ["RELAX", "FIST", "PINCH"]

    m1, report1 = _apply_train_budget_to_manifest(
        manifest,
        labels,
        class_names,
        budget_per_class=6,
        budget_seed=7,
    )
    m2, report2 = _apply_train_budget_to_manifest(
        manifest,
        labels,
        class_names,
        budget_per_class=6,
        budget_seed=7,
    )
    m3, _ = _apply_train_budget_to_manifest(
        manifest,
        labels,
        class_names,
        budget_per_class=6,
        budget_seed=13,
    )

    assert m1.train_indices == m2.train_indices
    assert report1 == report2
    assert len(m1.train_indices) == 18
    assert m1.train_indices != m3.train_indices
    assert all(v["selected"] == 6 for v in report1.values())


def test_budget_sampling_fails_fast_when_train_split_missing_class():
    labels = np.array([0] * 8 + [1] * 8, dtype=np.int32)
    manifest = _build_manifest(len(labels))
    class_names = ["RELAX", "FIST", "PINCH"]

    with pytest.raises(RuntimeError, match="zero samples for classes"):
        _apply_train_budget_to_manifest(
            manifest,
            labels,
            class_names,
            budget_per_class=4,
            budget_seed=42,
        )
