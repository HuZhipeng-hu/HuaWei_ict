from pathlib import Path

import numpy as np
import pytest

from shared.config import (
    get_protocol_input_shape,
    get_protocol_model_in_channels,
    load_config,
    load_training_config,
)
from shared.preprocessing import PreprocessPipeline
from training.trainer import build_balanced_sample_indices, compute_class_balanced_weights


def test_training_config_dual_branch_matches_model_channels():
    cfg_path = Path("configs/training.yaml")
    model_cfg, preprocess_cfg, train_cfg, _ = load_training_config(cfg_path)

    if preprocess_cfg.dual_branch.enabled:
        assert model_cfg.in_channels == 16
        assert model_cfg.in_channels == get_protocol_model_in_channels(preprocess_cfg)


def test_balanced_sampler_distribution():
    labels = np.array([0] * 10 + [1] * 5 + [2] * 2, dtype=np.int32)

    class _SamplerCfg:
        type = "balanced"
        hard_mining_ratio = 0.3
        confusion_pairs = [[0, 1]]

    idx = build_balanced_sample_indices(labels, batch_size=6, sampler_cfg=_SamplerCfg(), steps=10, seed=42)
    sampled = labels[idx]
    counts = np.bincount(sampled, minlength=3)

    assert idx.shape[0] == 60
    assert counts.min() > 0


def test_class_balanced_weights_are_finite():
    labels = np.array([0] * 100 + [1] * 20 + [2] * 5, dtype=np.int32)
    weights = compute_class_balanced_weights(labels, num_classes=3, beta=0.999)
    assert weights.shape == (3,)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()


def test_pipeline_shape_expected():
    model_cfg, preprocess_cfg, _, _ = load_training_config("configs/training.yaml")
    pipeline = PreprocessPipeline(preprocess_cfg)
    raw = np.random.randn(pipeline.get_required_window_size(), 8).astype(np.float32)
    feat = pipeline.process_window(raw)
    assert feat.shape[0] == model_cfg.in_channels
    assert (1,) + tuple(feat.shape) == get_protocol_input_shape(preprocess_cfg)


def test_protocol_helpers_reject_legacy_6_channel_config():
    _, preprocess_cfg, _, _ = load_training_config("configs/training.yaml")
    preprocess_cfg.num_channels = 6

    with pytest.raises(ValueError, match="8-channel dual-branch 16x24x6 protocol"):
        get_protocol_input_shape(preprocess_cfg)


def test_protocol_helpers_reject_single_branch_config():
    _, preprocess_cfg, _, _ = load_training_config("configs/training.yaml")
    preprocess_cfg.dual_branch.enabled = False

    with pytest.raises(ValueError, match="8-channel dual-branch 16x24x6 protocol"):
        get_protocol_model_in_channels(preprocess_cfg)
