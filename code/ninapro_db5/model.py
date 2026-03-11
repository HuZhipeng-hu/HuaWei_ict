"""Models and checkpoint transfer for NinaPro DB5 pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import load_checkpoint
    MINDSPORE_AVAILABLE = True
except Exception:
    nn = None  # type: ignore
    ops = None  # type: ignore
    load_checkpoint = None  # type: ignore
    MINDSPORE_AVAILABLE = False

from event_onset.model import GlobalAvgPool2DCompat
from ninapro_db5.config import DB5PretrainConfig
from shared.models.blocks import ParallelConvBlock, _check_mindspore


if MINDSPORE_AVAILABLE:

    class DB5PretrainNet(nn.Cell):
        def __init__(self, config: DB5PretrainConfig, num_classes: int):
            super().__init__()
            self.block1 = ParallelConvBlock(
                config.feature.first_myo_channel_count,
                config.base_channels,
                use_se=config.use_se,
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.block2 = ParallelConvBlock(
                config.base_channels * 3,
                config.base_channels * 2,
                use_se=config.use_se,
            )
            self.global_pool = GlobalAvgPool2DCompat()
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=config.dropout_rate)
            self.classifier = nn.SequentialCell(
                [
                    nn.Dense(config.base_channels * 6, config.classifier_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=config.dropout_rate),
                    nn.Dense(config.classifier_hidden_dim, num_classes),
                ]
            )

        def construct(self, x):
            x = self.block1(x)
            x = self.pool(x)
            x = self.block2(x)
            x = self.global_pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            return self.classifier(x)


else:

    class DB5PretrainNet:
        def __init__(self, *_args, **_kwargs):
            _check_mindspore()


def build_db5_pretrain_model(config: DB5PretrainConfig, num_classes: int) -> Any:
    _check_mindspore()
    return DB5PretrainNet(config, num_classes)


def load_emg_encoder_from_db5_checkpoint(event_model, checkpoint_path: str | Path) -> dict[str, int]:
    """Copy matching EMG branch weights from a DB5 pretrain checkpoint."""
    _check_mindspore()
    param_dict = load_checkpoint(str(checkpoint_path))
    current = {param.name: param for param in event_model.get_parameters()}
    loaded = 0
    skipped = 0
    mapping_prefixes = {
        "block1.": "emg_block1.",
        "block2.": "emg_block2.",
    }
    for source_name, tensor in param_dict.items():
        target_name = None
        for src_prefix, dst_prefix in mapping_prefixes.items():
            if source_name.startswith(src_prefix):
                target_name = dst_prefix + source_name[len(src_prefix) :]
                break
        if target_name is None or target_name not in current:
            skipped += 1
            continue
        target_param = current[target_name]
        if tuple(int(x) for x in target_param.shape) != tuple(int(x) for x in tensor.shape):
            skipped += 1
            continue
        target_param.set_data(tensor)
        loaded += 1
    return {"loaded": loaded, "skipped": skipped}
