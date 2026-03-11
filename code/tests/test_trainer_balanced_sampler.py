from __future__ import annotations

import numpy as np

from shared.config import SamplerConfig
from training.trainer import build_balanced_sample_indices


def test_balanced_sampler_handles_more_classes_than_batch_size():
    labels = np.repeat(np.arange(3, dtype=np.int32), 4)
    sampler_cfg = SamplerConfig(type="balanced", hard_mining_ratio=0.0, confusion_pairs=[])

    indices = build_balanced_sample_indices(
        labels,
        batch_size=2,
        sampler_cfg=sampler_cfg,
        steps=50,
        seed=123,
    )

    sampled_labels = labels[indices]
    assert indices.shape[0] == 100
    assert set(np.unique(sampled_labels).tolist()) == {0, 1, 2}
