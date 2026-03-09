from pathlib import Path

import numpy as np
import pytest

import training.trainer as trainer_module
from shared.config import EMAConfig, LossConfig, TrainingConfig


class _DummyTensor:
    def __init__(self, value):
        self._value = value

    def asnumpy(self):
        return np.asarray(self._value)


class _DummyDataset:
    def __init__(self, batches):
        self._batches = list(batches)

    def create_tuple_iterator(self):
        return iter(self._batches)


class _DummyModel:
    def trainable_params(self):
        return ["param"]

    def get_parameters(self):
        return []


class _FakeSoftmaxCrossEntropyWithLogits:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeWithLossCell:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn


class _FakeTrainOneStepCell:
    def __init__(self, loss_cell, optimizer):
        self.loss_cell = loss_cell
        self.optimizer = optimizer
        self.train_enabled = False

    def set_train(self, enabled):
        self.train_enabled = enabled

    def __call__(self, sample, label):
        return _DummyTensor(0.25)


class _FakeAdamWeightDecay:
    created = []

    def __init__(self, params, learning_rate, weight_decay):
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        type(self).created.append(self)

    def __setattr__(self, name, value):
        if name == "learning_rate" and hasattr(self, "learning_rate"):
            raise AssertionError("optimizer.learning_rate must not be reassigned after construction")
        object.__setattr__(self, name, value)


class _FakeNN:
    AdamWeightDecay = _FakeAdamWeightDecay
    SoftmaxCrossEntropyWithLogits = _FakeSoftmaxCrossEntropyWithLogits
    WithLossCell = _FakeWithLossCell
    TrainOneStepCell = _FakeTrainOneStepCell


def _make_config(**overrides):
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=0.1,
        warmup_epochs=1,
        loss=LossConfig(type="ce"),
        label_smoothing=0.0,
        ema=EMAConfig(enabled=False, decay=0.999),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_build_lr_returns_python_float_schedule(monkeypatch):
    monkeypatch.setattr(trainer_module, "ms", object())
    monkeypatch.setattr(trainer_module, "nn", _FakeNN)
    trainer = trainer_module.Trainer(_DummyModel(), _make_config(), ["RELAX", "FIST"])

    schedule = trainer._build_lr(steps_per_epoch=3)

    assert len(schedule) == 6
    assert all(isinstance(item, float) for item in schedule)
    assert schedule[0] < schedule[1] < schedule[2]
    assert schedule[3] > schedule[4] > schedule[5]


def test_build_optimizer_rejects_zero_steps(monkeypatch):
    monkeypatch.setattr(trainer_module, "ms", object())
    monkeypatch.setattr(trainer_module, "nn", _FakeNN)
    trainer = trainer_module.Trainer(_DummyModel(), _make_config(), ["RELAX", "FIST"])

    with pytest.raises(ValueError, match="zero steps per epoch"):
        trainer._build_optimizer(0)


def test_train_builds_optimizer_with_schedule_and_never_reassigns_lr(monkeypatch):
    _FakeAdamWeightDecay.created = []
    monkeypatch.setattr(trainer_module, "ms", object())
    monkeypatch.setattr(trainer_module, "nn", _FakeNN)
    monkeypatch.setattr(trainer_module, "save_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(
        trainer_module,
        "create_mindspore_dataset",
        lambda *_args, **_kwargs: _DummyDataset([(_DummyTensor([1.0]), _DummyTensor([0]))]),
    )
    monkeypatch.setattr(
        trainer_module.Trainer,
        "_build_train_dataset_for_epoch",
        lambda self, *_args, **_kwargs: _DummyDataset([(_DummyTensor([1.0]), _DummyTensor([0]))]),
    )
    monkeypatch.setattr(
        trainer_module.Trainer,
        "_evaluate",
        lambda self, _dataset: {"loss": 0.2, "acc": 0.8, "macro_f1": 0.7},
    )

    trainer = trainer_module.Trainer(
        _DummyModel(),
        _make_config(epochs=1),
        ["RELAX", "FIST"],
        output_dir="tests/.tmp_trainer_lr",
    )
    history = trainer.train(
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([[3.0], [4.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int32),
    )

    assert len(_FakeAdamWeightDecay.created) == 1
    optimizer = _FakeAdamWeightDecay.created[0]
    assert optimizer.learning_rate == [0.1]
    assert trainer.optimizer is optimizer
    assert trainer.train_step.optimizer is optimizer
    assert history["lr"] == [0.1]