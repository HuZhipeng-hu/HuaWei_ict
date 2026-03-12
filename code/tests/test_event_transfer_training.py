from __future__ import annotations

import numpy as np

import event_onset.trainer as event_trainer
from event_onset.config import EventModelConfig
from shared.config import EMAConfig, LossConfig, TrainingConfig


class _FakeParam:
    def __init__(self, name: str):
        self.name = name
        self.requires_grad = True


class _FakeModel:
    def __init__(self):
        self._params = [
            _FakeParam("emg_block1.conv.weight"),
            _FakeParam("emg_block2.conv.weight"),
            _FakeParam("imu_branch.0.weight"),
            _FakeParam("fusion.0.weight"),
            _FakeParam("fusion.3.weight"),
        ]

    def get_parameters(self):
        return self._params


class _FakeSoftmaxCrossEntropyWithLogits:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeAdamWeightDecay:
    def __init__(self, params, learning_rate, weight_decay):
        self.params = params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


class _FakeNN:
    AdamWeightDecay = _FakeAdamWeightDecay
    SoftmaxCrossEntropyWithLogits = _FakeSoftmaxCrossEntropyWithLogits


def test_transfer_phase_schedule():
    phases = event_trainer.build_transfer_phase_schedule(total_epochs=10, freeze_emg_epochs=3)
    assert [(item.name, item.epochs) for item in phases] == [("head_only", 3), ("unfreeze", 7)]

    phases_no_freeze = event_trainer.build_transfer_phase_schedule(total_epochs=8, freeze_emg_epochs=0)
    assert [(item.name, item.epochs) for item in phases_no_freeze] == [("full", 8)]


def test_optimizer_groups_apply_head_vs_encoder_lr_ratios(monkeypatch):
    monkeypatch.setattr(event_trainer, "ms", object())
    monkeypatch.setattr(event_trainer, "nn", _FakeNN)

    cfg = TrainingConfig(
        epochs=6,
        batch_size=4,
        learning_rate=0.01,
        warmup_epochs=1,
        freeze_emg_epochs=2,
        unfreeze_last_blocks=True,
        encoder_lr_ratio=0.25,
        head_lr_ratio=1.0,
        loss=LossConfig(type="ce"),
        ema=EMAConfig(enabled=False, decay=0.999),
    )
    trainer = event_trainer.EventTrainer(
        _FakeModel(),
        EventModelConfig(),
        cfg,
        class_names=["RELAX", "FIST", "PINCH"],
        output_dir="tests/.tmp_event_transfer_training",
    )

    _, _, _ = trainer._build_optimizer_for_phase(phase_name="head_only", phase_epochs=2, steps_per_epoch=3)
    params = trainer.model.get_parameters()
    assert params[0].requires_grad is False
    assert params[1].requires_grad is False
    assert params[2].requires_grad is False
    assert params[3].requires_grad is True

    optimizer, _, _ = trainer._build_optimizer_for_phase(phase_name="unfreeze", phase_epochs=4, steps_per_epoch=3)
    assert params[0].requires_grad is False
    assert params[1].requires_grad is True
    assert params[2].requires_grad is True
    assert params[3].requires_grad is True

    assert isinstance(optimizer.params, list)
    assert isinstance(optimizer.params[0], dict)
    encoder_lr = np.asarray(optimizer.params[0]["lr"], dtype=np.float64)
    head_lr = np.asarray(optimizer.params[1]["lr"], dtype=np.float64)
    assert encoder_lr.shape == head_lr.shape
    assert np.all(head_lr > encoder_lr)
    assert np.allclose(encoder_lr, head_lr * 0.25, atol=1e-8)
