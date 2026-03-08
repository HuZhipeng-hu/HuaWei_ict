"""Compatibility wrappers for training-side model imports."""

from __future__ import annotations

from typing import Any, Dict

from shared.config import ModelConfig
from shared.models import NeuroGripNet as _SharedNeuroGripNet
from shared.models import NeuroGripNetLite as _SharedNeuroGripNetLite
from shared.models import count_parameters, create_model


class NeuroGripNet(_SharedNeuroGripNet):
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 6,
        dropout_rate: float = 0.3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        base_channels: int = 16,
        use_se: bool = True,
    ):
        del hidden_dim, num_layers
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            use_se=use_se,
            dropout_rate=dropout_rate,
        )


class NeuroGripNetLite(_SharedNeuroGripNetLite):
    pass


def build_model_from_config(model_config: ModelConfig, *, dropout_rate: float | None = None):
    effective_dropout = model_config.dropout_rate if dropout_rate is None else float(dropout_rate)
    payload: Dict[str, Any] = {
        "model_type": model_config.model_type,
        "in_channels": model_config.in_channels,
        "num_classes": model_config.num_classes,
        "base_channels": model_config.base_channels,
        "use_se": model_config.use_se,
        "dropout_rate": effective_dropout,
    }
    return create_model(payload)


__all__ = [
    "NeuroGripNet",
    "NeuroGripNetLite",
    "create_model",
    "count_parameters",
    "build_model_from_config",
]
