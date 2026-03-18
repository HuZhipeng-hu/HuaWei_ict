from __future__ import annotations

import numpy as np

from event_onset.config import EventModelConfig
import event_onset.inference as inference
from event_onset.inference import EventModelMetadata, EventPredictor


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


class _FakeTwoStageLiteModel:
    def __init__(self):
        self._inputs = [_FakeInput((1, 8, 24, 3)), _FakeInput((1, 6, 16))]

    def get_inputs(self):
        return self._inputs

    def predict(self, _inputs):
        return [
            _FakeOutput([[0.1, 1.2]]),
            _FakeOutput([[2.2, 0.4, -0.3]]),
        ]


class _FakeBuildLiteModel(_FakeLiteModel):
    def __init__(self):
        super().__init__()
        self.build_calls = []

    def build_from_file(self, path, *, model_type, context):
        self.build_calls.append(
            {
                "path": path,
                "model_type": model_type,
                "context": context,
            }
        )


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


def test_event_predictor_ckpt_does_not_require_model_metadata(monkeypatch):
    def _fake_load_ckpt(self, _checkpoint_path):
        self._ckpt_model = object()
        self._expected_emg_shape = (1, 8, 24, 3)
        self._expected_imu_shape = (1, 6, 16)

    def _fail_if_called(_path):
        raise AssertionError("EventModelMetadata.load should not be called for ckpt backend")

    monkeypatch.setattr(EventPredictor, "_load_ckpt", _fake_load_ckpt)
    monkeypatch.setattr(EventModelMetadata, "load", staticmethod(_fail_if_called))

    predictor = EventPredictor(
        backend="ckpt",
        model_config=EventModelConfig(),
        checkpoint_path="artifacts/runs/any/checkpoints/event_onset_best.ckpt",
        model_metadata_path="models/event_onset.model_metadata.json",
    )
    assert predictor.metadata is None


def test_event_predictor_two_stage_lite_backend_returns_detail(monkeypatch):
    def _fake_load_lite(self, _model_path):
        self._lite_model = _FakeTwoStageLiteModel()
        self._emg_input_index = 0
        self._imu_input_index = 1
        self._expected_emg_shape = (1, 8, 24, 3)
        self._expected_imu_shape = (1, 6, 16)

    monkeypatch.setattr(EventPredictor, "_load_lite", _fake_load_lite)
    monkeypatch.setattr(
        EventModelMetadata,
        "load",
        staticmethod(
            lambda _path: EventModelMetadata(
                inputs={"emg": (1, 8, 24, 3), "imu": (1, 6, 16)},
                class_names=("CONTINUE", "TENSE_OPEN", "THUMB_UP", "WRIST_CW"),
                public_class_names=("CONTINUE", "TENSE_OPEN", "THUMB_UP", "WRIST_CW"),
                gate_classes=("CONTINUE", "COMMAND"),
                command_classes=("TENSE_OPEN", "THUMB_UP", "WRIST_CW"),
                model_variant="event_onset_two_stage_demo3",
                output_names=("gate_logits", "command_logits"),
            )
        ),
    )

    predictor = EventPredictor(
        backend="lite",
        model_config=EventModelConfig(model_type="event_onset_two_stage_demo3", num_classes=4),
        model_path="models/event_onset_demo3.mindir",
        model_metadata_path="models/event_onset_demo3.model_metadata.json",
    )
    detail = predictor.predict_detail(
        np.zeros((8, 24, 3), dtype=np.float32),
        np.zeros((6, 16), dtype=np.float32),
    )

    assert detail.public_probs.shape == (4,)
    assert detail.gate_probs is not None
    assert detail.command_probs is not None
    assert np.argmax(detail.public_probs) == 1
    assert np.isclose(np.sum(detail.gate_probs), 1.0, atol=1e-5)
    assert np.isclose(np.sum(detail.command_probs), 1.0, atol=1e-5)


def test_event_predictor_load_lite_uses_modeltype_enum_when_available(tmp_path, monkeypatch):
    model_path = tmp_path / "event_onset.mindir"
    model_path.write_bytes(b"mindir")

    fake_model = _FakeBuildLiteModel()

    class _FakeContext:
        def __init__(self):
            self.target = []

    class _FakeModelType:
        MINDIR = "mindir_enum"

    monkeypatch.setattr(inference, "LiteContext", _FakeContext)
    monkeypatch.setattr(inference, "LiteModel", lambda: fake_model)
    monkeypatch.setattr(inference, "LiteModelType", _FakeModelType)

    predictor = EventPredictor(
        backend="lite",
        model_config=EventModelConfig(),
        model_path=model_path,
    )

    assert predictor._lite_model is fake_model
    assert fake_model.build_calls
    assert fake_model.build_calls[0]["model_type"] == "mindir_enum"


def test_event_predictor_validate_inputs_returns_contiguous_arrays(monkeypatch):
    def _fake_load_ckpt(self, _checkpoint_path):
        self._ckpt_model = object()
        self._expected_emg_shape = (1, 8, 24, 3)
        self._expected_imu_shape = (1, 6, 16)

    monkeypatch.setattr(EventPredictor, "_load_ckpt", _fake_load_ckpt)
    predictor = EventPredictor(
        backend="ckpt",
        model_config=EventModelConfig(),
        checkpoint_path="artifacts/runs/any/checkpoints/event_onset_best.ckpt",
    )

    emg, imu = predictor._validate_inputs(
        np.zeros((8, 24, 3), dtype=np.float32),
        np.zeros((6, 16), dtype=np.float32),
    )
    assert emg.flags["C_CONTIGUOUS"]
    assert imu.flags["C_CONTIGUOUS"]
