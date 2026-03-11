from __future__ import annotations

import numpy as np

from event_onset.config import EventModelConfig
from event_onset.inference import EventPredictor


class _FakeInput:
    def __init__(self, shape):
        self.shape = shape
        self.data = None

    def set_data_from_numpy(self, value):
        self.data = np.asarray(value, dtype=np.float32)


class _FakeOutput:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def get_data_to_numpy(self):
        return self._value


class _FakeLiteModel:
    def __init__(self):
        self._inputs = [_FakeInput((1, 8, 24, 3)), _FakeInput((1, 6, 16))]

    def get_inputs(self):
        return self._inputs

    def predict(self, _inputs):
        return [_FakeOutput([[0.2, 1.0, -0.4]])]


def test_event_predictor_lite_backend_uses_two_inputs(monkeypatch):
    def _fake_load_lite(self, _model_path):
        self._lite_model = _FakeLiteModel()
        self._emg_input_index = 0
        self._imu_input_index = 1
        self._expected_emg_shape = (1, 8, 24, 3)
        self._expected_imu_shape = (1, 6, 16)

    monkeypatch.setattr(EventPredictor, "_load_lite", _fake_load_lite)

    predictor = EventPredictor(
        backend="lite",
        model_config=EventModelConfig(),
        model_path="models/event_onset.mindir",
    )
    probs = predictor.predict_proba(
        np.zeros((8, 24, 3), dtype=np.float32),
        np.zeros((6, 16), dtype=np.float32),
    )

    assert probs.shape == (3,)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert np.argmax(probs) == 1
