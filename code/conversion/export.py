"""
模型导出 — .ckpt → .mindir

将 MindSpore 训练好的模型权重导出为 MindSpore Lite 可加载的 MINDIR 格式。

MINDIR 格式包含:
- 模型结构（计算图）
- 权重参数
- 输入/输出的 shape 和 dtype
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, export, load_checkpoint, load_param_into_net
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

logger = logging.getLogger(__name__)


def export_to_mindir(
    model,
    checkpoint_path: str,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 6, 24, 6),
) -> str:
    """
    将 .ckpt 模型导出为 .mindir 格式

    Args:
        model: MindSpore nn.Cell 模型实例（结构，未加载权重）
        checkpoint_path: .ckpt 权重文件路径
        output_path: 输出 .mindir 文件路径（不含扩展名）
        input_shape: 模型输入形状 (batch, channels, freq_bins, time_frames)

    Returns:
        实际保存的文件路径（含扩展名）
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore 未安装")

    # 加载权重
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    logger.info(f"已加载检查点: {checkpoint_path}")

    # 构造虚拟输入
    dummy_input = Tensor(np.zeros(input_shape, dtype=np.float32))

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出
    # MindSpore export 会自动添加 .mindir 扩展名
    output_stem = str(Path(output_path).with_suffix(''))
    export(model, dummy_input, file_name=output_stem, file_format="MINDIR")

    final_path = output_stem + ".mindir"
    logger.info(f"模型已导出: {final_path}")
    logger.info(f"  输入形状: {input_shape}")
    logger.info(f"  文件大小: {Path(final_path).stat().st_size / 1024:.1f} KB")

    return final_path
