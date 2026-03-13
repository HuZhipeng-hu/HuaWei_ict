from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrain_ninapro_db5_repr import (
    _build_class_source_balanced_indices,
    _resolve_existing_recordings_manifest,
    _resolve_augmentation_params,
    _resolve_repr_objective,
    _resolve_sampler_mode,
)


def test_resolve_repr_objective_supports_supcon_and_joint_ce() -> None:
    assert _resolve_repr_objective("supcon") == "supcon"
    assert _resolve_repr_objective("supcon_ce") == "supcon_ce"
    assert _resolve_repr_objective("  SUPCON+CE  ") == "supcon_ce"
    with pytest.raises(ValueError, match="Unsupported repr objective"):
        _resolve_repr_objective("triplet")


def test_resolve_sampler_mode_supports_source_balanced() -> None:
    assert _resolve_sampler_mode("class_balanced") == "class_balanced"
    assert _resolve_sampler_mode("balanced") == "class_balanced"
    assert _resolve_sampler_mode("class_source_balanced") == "class_source_balanced"
    with pytest.raises(ValueError, match="Unsupported sampler_mode"):
        _resolve_sampler_mode("random")


def test_resolve_augmentation_params_allows_profile_with_overrides() -> None:
    params = _resolve_augmentation_params(
        profile="strong",
        noise_std=0.012,
        channel_drop_ratio=0.11,
    )
    assert params["profile"] == "strong"
    assert params["noise_std"] == pytest.approx(0.012)
    assert params["channel_drop_ratio"] == pytest.approx(0.11)
    assert params["time_mask_ratio"] > 0.0
    assert params["freq_mask_ratio"] > 0.0


def test_source_balanced_indices_cover_all_classes_per_batch_when_possible() -> None:
    labels = np.asarray([0] * 12 + [1] * 12 + [2] * 12, dtype=np.int32)
    source_ids = np.asarray(
        [100] * 6 + [101] * 6 + [200] * 6 + [201] * 6 + [300] * 6 + [301] * 6,
        dtype=np.int32,
    )
    batch_size = 6
    steps = 4
    indices = _build_class_source_balanced_indices(
        labels=labels,
        source_ids=source_ids,
        batch_size=batch_size,
        steps=steps,
        seed=7,
    )
    assert indices.shape[0] == batch_size * steps
    assert np.all(indices >= 0)
    assert np.all(indices < labels.shape[0])

    for step in range(steps):
        batch = indices[step * batch_size : (step + 1) * batch_size]
        batch_classes = set(labels[batch].tolist())
        assert {0, 1, 2}.issubset(batch_classes)


def test_resolve_existing_recordings_manifest_prefers_explicit_path(tmp_path) -> None:
    manifest = tmp_path / "explicit_manifest.csv"
    manifest.write_text("relative_path\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    resolved = _resolve_existing_recordings_manifest(
        data_dir=str(tmp_path),
        config_path=str(config),
        manifest_arg=str(manifest),
    )
    assert resolved.endswith("explicit_manifest.csv")


def test_resolve_existing_recordings_manifest_missing_raises_clear_error(tmp_path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("data:\n  recordings_manifest_path: recordings_manifest.csv\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="run_downstream_fewshot=true requires recordings manifest"):
        _resolve_existing_recordings_manifest(
            data_dir=str(tmp_path),
            config_path=str(config),
            manifest_arg=None,
        )
