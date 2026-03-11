"""
神经网络构建模块

包含组成 NeuroGripNet 的可复用构件：
- GlobalAvgPool2DCompat: Lite 兼容的全局平均池化
- ParallelConvBlock: Inception 风格多尺度并行卷积
- SEBlock: Squeeze-and-Excitation 通道注意力
- DepthwiseSeparableConv: 深度可分离卷积（轻量版使用）
"""

try:
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False


def _check_mindspore():
    """检查 MindSpore 是否可用"""
    if not MINDSPORE_AVAILABLE:
        raise ImportError(
            "MindSpore 未安装。模型构建需要 MindSpore >= 2.7.1，"
            "请运行: pip install mindspore"
        )


# =============================================================================
# Squeeze-and-Excitation 通道注意力
# =============================================================================

if MINDSPORE_AVAILABLE:
    class GlobalAvgPool2DCompat(nn.Cell):
        """
        Lite 兼容全局平均池化。

        等价于 AdaptiveAvgPool2D(output_size=(1,1))，但使用 ReduceMean
        避免在部分 MindSpore Lite CPU 后端上出现算子不支持。
        """

        def __init__(self):
            super().__init__()
            self.reduce_mean = ops.ReduceMean(keep_dims=True)

        def construct(self, x):
            # (B, C, H, W) -> (B, C, 1, 1)
            return self.reduce_mean(x, (2, 3))


    class SEBlock(nn.Cell):
        """
        Squeeze-and-Excitation 通道注意力模块

        通过全局平均池化 → 两层FC → Sigmoid 对通道重新加权，
        让网络学会"关注哪些通道更重要"。

        Args:
            channels: 输入通道数
            reduction: 压缩比（中间层通道数 = channels // reduction）
        """

        def __init__(self, channels: int, reduction: int = 4):
            super().__init__()
            mid_channels = max(channels // reduction, 1)

            self.squeeze = GlobalAvgPool2DCompat()
            self.excitation = nn.SequentialCell([
                nn.Dense(channels, mid_channels, has_bias=False),
                nn.ReLU(),
                nn.Dense(mid_channels, channels, has_bias=False),
                nn.Sigmoid(),
            ])

        def construct(self, x):
            """
            Args:
                x: (batch, channels, height, width)
            Returns:
                通道重新加权后的张量，形状不变
            """
            batch, channels, _, _ = x.shape
            # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
            scale = self.squeeze(x).reshape(batch, channels)
            # Excitation: (B, C) → (B, C)
            scale = self.excitation(scale)
            # Scale: (B, C) → (B, C, 1, 1) 广播乘回原张量
            scale = scale.reshape(batch, channels, 1, 1)
            return x * scale


    # =========================================================================
    # Inception 风格多尺度并行卷积
    # =========================================================================

    class ParallelConvBlock(nn.Cell):
        """
        多尺度并行卷积模块（Inception 风格）

        同时用 1×1、3×3、5×5 三种卷积核提取不同尺度的时频特征，
        然后沿通道维拼接，再可选地通过 SE 注意力重新加权。

        结构:
            输入 ──┬── Conv1x1 → BN → ReLU ──┐
                   ├── Conv3x3 → BN → ReLU ──┤── Concat → (SE →) Output
                   └── Conv5x5 → BN → ReLU ──┘

        Args:
            in_channels: 输入通道数
            out_channels: 每个分支的输出通道数（总输出 = 3 * out_channels）
            use_se: 是否使用 SE 注意力
            se_reduction: SE 压缩比
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_se: bool = True,
            se_reduction: int = 4,
        ):
            super().__init__()

            # 1×1 分支: 提取逐像素（逐时频点）特征
            self.branch_1x1 = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          has_bias=False, pad_mode='valid'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])

            # 3×3 分支: 提取局部时频特征
            self.branch_3x3 = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          has_bias=False, pad_mode='pad', padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])

            # 5×5 分支: 提取更大范围的时频特征
            self.branch_5x5 = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=5,
                          has_bias=False, pad_mode='pad', padding=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])

            self.concat = ops.Concat(axis=1)

            # 可选 SE 注意力
            total_channels = out_channels * 3
            self.use_se = use_se
            self.se = SEBlock(total_channels, se_reduction) if use_se else None

        def construct(self, x):
            """
            Args:
                x: (batch, in_channels, freq, time)
            Returns:
                (batch, out_channels * 3, freq, time)
            """
            out_1x1 = self.branch_1x1(x)
            out_3x3 = self.branch_3x3(x)
            out_5x5 = self.branch_5x5(x)

            out = self.concat((out_1x1, out_3x3, out_5x5))

            if self.use_se and self.se is not None:
                out = self.se(out)

            return out


    # =========================================================================
    # 深度可分离卷积（轻量化版本使用）
    # =========================================================================

    class DepthwiseSeparableConv(nn.Cell):
        """
        深度可分离卷积

        将标准卷积分解为 Depthwise（空间特征）+ Pointwise（通道混合），
        参数量约为标准卷积的 1/kernel_size²。

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
        ):
            super().__init__()

            self.depthwise = nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                group=in_channels,
                has_bias=False,
                pad_mode='same',
            )
            self.pointwise = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1,
                has_bias=False,
                pad_mode='valid',
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

        def construct(self, x):
            """
            Args:
                x: (batch, in_channels, height, width)
            Returns:
                (batch, out_channels, height, width)
            """
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

else:
    # MindSpore 不可用时的占位符
    class GlobalAvgPool2DCompat:
        def __init__(self, *args, **kwargs):
            _check_mindspore()

    class SEBlock:
        def __init__(self, *args, **kwargs):
            _check_mindspore()

    class ParallelConvBlock:
        def __init__(self, *args, **kwargs):
            _check_mindspore()

    class DepthwiseSeparableConv:
        def __init__(self, *args, **kwargs):
            _check_mindspore()
