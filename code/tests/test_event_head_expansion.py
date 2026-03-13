from __future__ import annotations

import numpy as np

from event_onset.head_expansion import (
    build_event_class_names,
    build_row_mapping,
    expand_classifier_rows,
    normalize_action_keys,
)


def test_normalize_action_keys_deduplicates_and_ignores_relax() -> None:
    keys = normalize_action_keys("e1_g01, RELAX, e1_g02, E1_G01")
    assert keys == ["E1_G01", "E1_G02"]


def test_build_row_mapping_uses_class_names() -> None:
    mapping = build_row_mapping(
        old_class_names=["RELAX", "E1_G01", "E1_G02"],
        new_class_names=["RELAX", "E1_G01", "E1_G03", "E1_G02"],
    )
    assert mapping == {0: 0, 1: 1, 3: 2}


def test_expand_classifier_rows_reuses_old_rows_and_initializes_new_rows() -> None:
    old_classes = build_event_class_names(["E1_G01", "E1_G02"])
    new_classes = build_event_class_names(["E1_G01", "E1_G02", "E1_G03"])
    old_weight = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
        ],
        dtype=np.float32,
    )
    old_bias = np.asarray([0.01, 0.11, 0.21], dtype=np.float32)
    target_weight = np.zeros((4, 3), dtype=np.float32)
    target_bias = np.ones((4,), dtype=np.float32)

    expanded_w, expanded_b, stats = expand_classifier_rows(
        old_weight=old_weight,
        old_bias=old_bias,
        target_weight=target_weight,
        target_bias=target_bias,
        old_class_names=old_classes,
        new_class_names=new_classes,
        init_seed=7,
    )

    assert np.allclose(expanded_w[0], old_weight[0])
    assert np.allclose(expanded_w[1], old_weight[1])
    assert np.allclose(expanded_w[2], old_weight[2])
    assert np.allclose(expanded_b[:3], old_bias)
    assert np.allclose(expanded_b[3], 0.0)
    assert not np.allclose(expanded_w[3], 0.0)
    assert stats.reused_class_count == 3
    assert stats.new_class_count == 1
    assert stats.new_classes == ["E1_G03"]
