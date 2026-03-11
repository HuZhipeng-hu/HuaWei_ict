"""MindSpore models for the event-onset experiment."""

from __future__ import annotations

from typing import Any

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


if MINDSPORE_AVAILABLE:

    class GlobalAvgPool1DCompat(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reduce_mean = ops.ReduceMean(keep_dims=False)

        def construct(self, x):
            return self.reduce_mean(x, 2)


    class EventOnsetNet(nn.Cell):
        """Late-fusion EMG+IMU event classifier."""

        def __init__(self, config: EventModelConfig):
            super().__init__()
            self.config = config

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
                    nn.Conv1d(config.imu_input_dim, config.imu_base_channels, kernel_size=3, pad_mode="same", has_bias=False),
                    nn.ReLU(),
                    nn.Conv1d(config.imu_base_channels, config.imu_base_channels * 2, kernel_size=3, pad_mode="same", has_bias=False),
                    nn.ReLU(),
                ]
            )
            self.imu_pool = GlobalAvgPool1DCompat()

            emg_embedding_dim = config.base_channels * 6
            imu_embedding_dim = config.imu_base_channels * 2
            self.fusion = nn.SequentialCell(
                [
                    nn.Dense(emg_embedding_dim + imu_embedding_dim, config.fusion_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=config.dropout_rate),
                    nn.Dense(config.fusion_hidden_dim, config.num_classes),
                ]
            )
            self.concat = ops.Concat(axis=1)

        def construct(self, emg, imu):
            emg = self.emg_block1(emg)
            emg = self.emg_pool(emg)
            emg = self.emg_block2(emg)
            emg = self.emg_global_pool(emg)
            emg = self.emg_flatten(emg)

            imu = self.imu_branch(imu)
            imu = self.imu_pool(imu)

            fused = self.concat((emg, imu))
            return self.fusion(fused)


else:

    class EventOnsetNet:
        def __init__(self, *_args, **_kwargs):
            _check_mindspore()


def build_event_model(config: EventModelConfig) -> Any:
    _check_mindspore()
    return EventOnsetNet(config)
