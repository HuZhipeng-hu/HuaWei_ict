"""MindSpore models for the event-onset experiment."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import mindspore.nn as nn
    import mindspore.ops as ops

    MINDSPORE_AVAILABLE = True
except Exception:
    nn = None  # type: ignore
    ops = None  # type: ignore
    MINDSPORE_AVAILABLE = False

from event_onset.config import EventModelConfig
from shared.models.blocks import GlobalAvgPool2DCompat, ParallelConvBlock, _check_mindspore

TWO_STAGE_DEMO3_MODEL_TYPE = "event_onset_two_stage_demo3"
TWO_STAGE_GATE_CLASSES = ("CONTINUE", "COMMAND")
TWO_STAGE_COMMAND_CLASSES = ("TENSE_OPEN", "THUMB_UP", "WRIST_CW")


def is_two_stage_demo3_model(model_type: str | None) -> bool:
    return str(model_type or "").strip().lower() == TWO_STAGE_DEMO3_MODEL_TYPE


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    denom = np.sum(exp_logits, axis=-1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (exp_logits / denom).astype(np.float32)


def combine_two_stage_public_probabilities(
    gate_probs: np.ndarray,
    command_probs: np.ndarray,
) -> np.ndarray:
    gate = np.asarray(gate_probs, dtype=np.float32)
    command = np.asarray(command_probs, dtype=np.float32)
    squeeze = gate.ndim == 1
    if squeeze:
        gate = gate[np.newaxis, ...]
    if command.ndim == 1:
        command = command[np.newaxis, ...]
    if gate.shape[0] != command.shape[0]:
        raise ValueError(
            f"gate and command batch sizes must match, got {gate.shape} vs {command.shape}"
        )
    if gate.shape[-1] != 2:
        raise ValueError(f"gate_probs must have 2 classes, got shape={gate.shape}")
    if command.shape[-1] != len(TWO_STAGE_COMMAND_CLASSES):
        raise ValueError(
            f"command_probs must have {len(TWO_STAGE_COMMAND_CLASSES)} classes, got shape={command.shape}"
        )
    public = np.zeros((gate.shape[0], 1 + command.shape[-1]), dtype=np.float32)
    public[:, 0] = gate[:, 0]
    public[:, 1:] = gate[:, 1:2] * command
    return public[0] if squeeze else public


def combine_two_stage_public_probabilities_from_logits(
    gate_logits: np.ndarray,
    command_logits: np.ndarray,
) -> np.ndarray:
    return combine_two_stage_public_probabilities(
        _softmax_np(gate_logits),
        _softmax_np(command_logits),
    )


if MINDSPORE_AVAILABLE:

    class GlobalAvgPool1DCompat(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reduce_mean = ops.ReduceMean(keep_dims=False)

        def construct(self, x):
            return self.reduce_mean(x, 2)


    class _EventOnsetFeatureMixin:
        def _init_feature_layers(self, config: EventModelConfig) -> None:
            self.emg_block1 = ParallelConvBlock(
                config.emg_in_channels,
                config.base_channels,
                use_se=config.use_se,
            )
            self.emg_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.emg_block2 = ParallelConvBlock(
                config.base_channels * 3,
                config.base_channels * 2,
                use_se=config.use_se,
            )
            self.emg_global_pool = GlobalAvgPool2DCompat()
            self.emg_flatten = nn.Flatten()

            self.imu_branch = nn.SequentialCell(
                [
                    nn.Conv1d(
                        config.imu_input_dim,
                        config.imu_base_channels,
                        kernel_size=3,
                        pad_mode="same",
                        has_bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        config.imu_base_channels,
                        config.imu_base_channels * 2,
                        kernel_size=3,
                        pad_mode="same",
                        has_bias=False,
                    ),
                    nn.ReLU(),
                ]
            )
            self.imu_pool = GlobalAvgPool1DCompat()
            self.concat = ops.Concat(axis=1)
            self.embedding_dim = config.base_channels * 6 + config.imu_base_channels * 2

        def _encode_fused(self, emg, imu):
            emg = self.emg_block1(emg)
            emg = self.emg_pool(emg)
            emg = self.emg_block2(emg)
            emg = self.emg_global_pool(emg)
            emg = self.emg_flatten(emg)

            imu = self.imu_branch(imu)
            imu = self.imu_pool(imu)
            return self.concat((emg, imu))


    class EventOnsetNet(nn.Cell, _EventOnsetFeatureMixin):
        """Late-fusion EMG+IMU event classifier."""

        def __init__(self, config: EventModelConfig):
            super().__init__()
            self.config = config
            self._init_feature_layers(config)
            self.fusion = nn.SequentialCell(
                [
                    nn.Dense(self.embedding_dim, config.fusion_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=config.dropout_rate),
                    nn.Dense(config.fusion_hidden_dim, config.num_classes),
                ]
            )

        def construct(self, emg, imu):
            fused = self._encode_fused(emg, imu)
            return self.fusion(fused)


    class EventOnsetTwoStageDemo3Net(nn.Cell, _EventOnsetFeatureMixin):
        """Two-stage event-onset model for demo3 latch control."""

        def __init__(self, config: EventModelConfig):
            super().__init__()
            self.config = config
            self._init_feature_layers(config)
            self.fusion = nn.SequentialCell(
                [
                    nn.Dense(self.embedding_dim, config.fusion_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=config.dropout_rate),
                ]
            )
            self.gate_head = nn.Dense(config.fusion_hidden_dim, len(TWO_STAGE_GATE_CLASSES))
            self.command_head = nn.Dense(config.fusion_hidden_dim, len(TWO_STAGE_COMMAND_CLASSES))

        def construct(self, emg, imu):
            fused = self._encode_fused(emg, imu)
            hidden = self.fusion(fused)
            return self.gate_head(hidden), self.command_head(hidden)


else:

    class EventOnsetNet:
        def __init__(self, *_args, **_kwargs):
            _check_mindspore()


    class EventOnsetTwoStageDemo3Net:
        def __init__(self, *_args, **_kwargs):
            _check_mindspore()


def build_event_model(config: EventModelConfig) -> Any:
    _check_mindspore()
    if is_two_stage_demo3_model(config.model_type):
        return EventOnsetTwoStageDemo3Net(config)
    return EventOnsetNet(config)
