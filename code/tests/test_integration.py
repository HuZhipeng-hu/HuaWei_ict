from pathlib import Path

import numpy as np

from event_onset.config import load_event_training_config
from event_onset.runtime import EventFeatureExtractor
from training.trainer import build_balanced_sample_indices, compute_class_balanced_weights


def test_event_training_config_expected_defaults():
    cfg_path = Path("configs/training_event_onset.yaml")
    model_cfg, data_cfg, train_cfg, _ = load_event_training_config(cfg_path)

    assert model_cfg.model_type == "event_onset"
    assert model_cfg.num_classes == 3
    assert model_cfg.emg_in_channels == 8
    assert model_cfg.imu_input_dim == 6
    assert data_cfg.label_mode == "event_onset"
    assert train_cfg.val_ratio > 0.0
    assert train_cfg.test_ratio > 0.0


def test_event_feature_extractor_shapes_match_model_config():
    model_cfg, data_cfg, _, _ = load_event_training_config("configs/training_event_onset.yaml")
    extractor = EventFeatureExtractor(data_cfg)
    matrix = np.random.randn(data_cfg.context_samples, 14).astype(np.float32)

    emg_feature, imu_feature, energy = extractor.build_inputs(matrix)

    assert emg_feature.shape == (
        model_cfg.emg_in_channels,
        model_cfg.emg_freq_bins,
        model_cfg.emg_time_frames,
    )
    assert imu_feature.shape == (model_cfg.imu_input_dim, model_cfg.imu_num_steps)
    assert np.isfinite(energy)


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
