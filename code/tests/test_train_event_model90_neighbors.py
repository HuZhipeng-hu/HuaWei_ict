from __future__ import annotations

from types import SimpleNamespace

from scripts.train_event_model_90_sprint import _build_neighbor_candidates


def test_build_neighbor_candidates_has_reference_and_unique_grid() -> None:
    args = SimpleNamespace(
        neighbor_lr_delta_ratio=0.2,
        neighbor_freeze_delta=2,
        screen_loss_types="cross_entropy,cb_focal",
        screen_base_channels="16,24",
    )
    reference = {
        "loss_type": "cross_entropy",
        "base_channels": 16,
        "freeze_emg_epochs": 5,
        "encoder_lr_ratio": 0.2,
        "pretrained_mode": "off",
    }

    rows = _build_neighbor_candidates(args, reference=reference)
    assert any(row["variant"] == "ref" for row in rows)
    assert any(str(row["loss_type"]) == "cb_focal" for row in rows)
    assert any(int(row["base_channels"]) == 24 for row in rows)

    keys = {
        (
            str(row["loss_type"]),
            int(row["base_channels"]),
            int(row["freeze_emg_epochs"]),
            float(row["encoder_lr_ratio"]),
            str(row["pretrained_mode"]),
        )
        for row in rows
    }
    assert len(keys) == len(rows)

