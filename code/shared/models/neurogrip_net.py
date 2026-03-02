"""
NeuroGripNet — EMG 手势识别卷积神经网络

提供两种变体：
- NeuroGripNet:     主模型（~227K 参数），使用 ParallelConvBlock + SE 注意力
- NeuroGripNetLite: 轻量模型，使用深度可分离卷积，适合更严格的资源约束

两者接收相同的输入格式 (batch, channels, freq_bins, time_frames)，
输出 (batch, num_classes) 的 logits。

典型输入形状: (N, 6, 24, 13) — 6通道EMG, 24频率bin, 13时间帧
"""

from typing import Optional, Dict, Any

try:
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

from .blocks import ParallelConvBlock, SEBlock, DepthwiseSeparableConv, _check_mindspore
from ..gestures import NUM_CLASSES


# =============================================================================
# NeuroGripNet — 主模型
# =============================================================================

if MINDSPORE_AVAILABLE:

    class NeuroGripNet(nn.Cell):
        """
        NeuroGripNet 主模型

        架构:
            Input (B, C_in, F, T)
            → ParallelConvBlock_1  → (B, base*3, F, T)
            → MaxPool2d            → (B, base*3, F/2, T/2)
            → ParallelConvBlock_2  → (B, base*2*3, F/2, T/2)
            → AdaptiveAvgPool2d    → (B, base*2*3, 1, 1)
            → Flatten              → (B, base*6)
            → Dropout
            → Dense                → (B, num_classes)

        Args:
            in_channels: 输入 EMG 通道数（默认 6）
            num_classes: 分类类别数（默认从 gestures.py 获取）
            base_channels: 基础通道数，控制模型宽度
            use_se: 是否使用 SE 注意力
            dropout_rate: Dropout 概率
        """

        def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = NUM_CLASSES,
            base_channels: int = 16,
            use_se: bool = True,
            dropout_rate: float = 0.3,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes

            # 第一层多尺度卷积: in_channels → base_channels * 3
            self.block1 = ParallelConvBlock(
                in_channels, base_channels, use_se=use_se,
            )
            ch_after_block1 = base_channels * 3  # 48

            # 池化: 下采样时频图
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # 第二层多尺度卷积: base*3 → base*2*3
            self.block2 = ParallelConvBlock(
                ch_after_block1, base_channels * 2, use_se=use_se,
            )
            ch_after_block2 = base_channels * 2 * 3  # 96

            # 全局平均池化 → 分类器
            self.global_pool = ops.AdaptiveAvgPool2D(output_size=(1, 1))
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Dense(ch_after_block2, num_classes)

        def construct(self, x):
            """
            前向传播

            Args:
                x: (batch, in_channels, freq_bins, time_frames)
                   典型形状: (N, 6, 24, 13)
            Returns:
                logits: (batch, num_classes)
            """
            x = self.block1(x)          # (B, 48, 24, 13)
            x = self.pool(x)            # (B, 48, 12, 6)
            x = self.block2(x)          # (B, 96, 12, 6)
            x = self.global_pool(x)     # (B, 96, 1, 1)
            x = self.flatten(x)         # (B, 96)
            x = self.dropout(x)
            x = self.classifier(x)      # (B, num_classes)
            return x


    # =========================================================================
    # NeuroGripNetLite — 轻量模型
    # =========================================================================

    class NeuroGripNetLite(nn.Cell):
        """
        NeuroGripNet 轻量版

        使用深度可分离卷积替代 ParallelConvBlock，大幅减少参数量。
        适合内存/算力更受限的部署场景。

        Args:
            in_channels: 输入 EMG 通道数
            num_classes: 分类类别数
            base_channels: 基础通道数
            dropout_rate: Dropout 概率
        """

        def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = NUM_CLASSES,
            base_channels: int = 16,
            dropout_rate: float = 0.3,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes

            ch1 = base_channels * 2    # 32
            ch2 = base_channels * 4    # 64

            self.features = nn.SequentialCell([
                # 层 1: 标准卷积升维
                nn.Conv2d(in_channels, base_channels, kernel_size=3,
                          has_bias=False, pad_mode='same'),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),

                # 层 2: 深度可分离卷积
                DepthwiseSeparableConv(base_channels, ch1, kernel_size=3),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # 层 3: 深度可分离卷积
                DepthwiseSeparableConv(ch1, ch2, kernel_size=3),
            ])

            self.global_pool = ops.AdaptiveAvgPool2D(output_size=(1, 1))
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Dense(ch2, num_classes)

        def construct(self, x):
            """
            Args:
                x: (batch, in_channels, freq_bins, time_frames)
            Returns:
                logits: (batch, num_classes)
            """
            x = self.features(x)
            x = self.global_pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.classifier(x)
            return x

else:
    # MindSpore 不可用时的占位符
    class NeuroGripNet:
        def __init__(self, *args, **kwargs):
            _check_mindspore()

    class NeuroGripNetLite:
        def __init__(self, *args, **kwargs):
            _check_mindspore()


# =============================================================================
# 工厂函数
# =============================================================================

def create_model(config: Dict[str, Any]) -> Any:
    """
    根据配置创建模型实例

    Args:
        config: 模型配置字典，至少包含:
            - model_type: "standard" 或 "lite"
            - in_channels: 输入通道数
            - num_classes: 类别数
            - base_channels: 基础通道数
            - use_se: 是否使用 SE 注意力（仅 standard）
            - dropout_rate: Dropout 概率

    Returns:
        模型实例
    """
    _check_mindspore()

    model_type = config.get("model_type", "standard")
    kwargs = {
        "in_channels": config.get("in_channels", 6),
        "num_classes": config.get("num_classes", NUM_CLASSES),
        "base_channels": config.get("base_channels", 16),
        "dropout_rate": config.get("dropout_rate", 0.3),
    }

    if model_type == "standard":
        kwargs["use_se"] = config.get("use_se", True)
        return NeuroGripNet(**kwargs)
    elif model_type == "lite":
        return NeuroGripNetLite(**kwargs)
    else:
        raise ValueError(f"未知模型类型: {model_type}，可选: standard / lite")


def count_parameters(model) -> int:
    """
    统计模型的可训练参数总量

    Args:
        model: MindSpore nn.Cell 实例

    Returns:
        参数总数（标量值）
    """
    _check_mindspore()
    total = 0
    for param in model.trainable_params():
        total += param.size
    return total
