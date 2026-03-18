from __future__ import annotations

import numpy as np

import experimental.ninapro_db5.model as db5_model


class _Param:
    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape
        self.data = None

    def set_data(self, tensor):
        self.data = tensor


class _EventModel:
    def __init__(self):
        self.params = [
            _Param("emg_block1.conv", (2, 2)),
            _Param("emg_block2.conv", (3, 3)),
        ]

    def get_parameters(self):
        return self.params


def test_db5_transfer_loads_only_shape_matching_emg_blocks(monkeypatch):
    event_model = _EventModel()
    source = {
        "block1.conv": np.zeros((2, 2), dtype=np.float32),
        "block2.conv": np.zeros((3, 3), dtype=np.float32),
        "block2.mismatch": np.zeros((9, 9), dtype=np.float32),
        "classifier.weight": np.zeros((1, 1), dtype=np.float32),
    }

    monkeypatch.setattr(db5_model, "_check_mindspore", lambda: None)
    monkeypatch.setattr(db5_model, "load_checkpoint", lambda _path: source)

    result = db5_model.load_emg_encoder_from_db5_checkpoint(event_model, "dummy.ckpt")

    assert result == {"loaded": 2, "skipped": 2}
    assert event_model.params[0].data is source["block1.conv"]
    assert event_model.params[1].data is source["block2.conv"]
