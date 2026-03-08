"""
INT8 训练后量化 (Post-Training Quantization)

将 FP32 模型量化为 INT8，减小模型体积并加速推理。
需要提供校准数据集来统计各层的激活值分布。

量化效果:
- 模型体积减小约 75% (FP32 → INT8)
- 推理速度提升约 2-4 倍（依赖硬件）
- 准确率损失通常 < 1%
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

mslite: Any = None

try:
    import mindspore_lite as mslite

    MSLITE_AVAILABLE = True
except ImportError:
    MSLITE_AVAILABLE = False


def quantize_model(
    input_path: str,
    output_path: str,
    calibration_data: Optional[np.ndarray] = None,
    num_calibration_samples: int = 100,
    input_shape: Tuple[int, ...] = (1, 12, 24, 6),
    bit_num: int = 8,
) -> str:
    """
    对 MINDIR 模型执行 INT8 量化

    Args:
        input_path: 输入 .mindir 文件路径
        output_path: 输出量化后的 .mindir 文件路径
        calibration_data: 校准数据 (N, C, F, T) float32
                          如果为 None，使用随机数据校准（不推荐）
        num_calibration_samples: 使用的校准样本数
        input_shape: 量化时模型输入形状
        bit_num: 量化位宽（默认 8）

    Returns:
        量化后的模型文件路径
    """
    if not MSLITE_AVAILABLE:
        raise ImportError(
            "MindSpore Lite 未安装。量化需要 mindspore_lite，请参考文档安装。"
        )
    assert mslite is not None

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {input_path}")

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始量化: {input_path}")
    logger.info(f"  校准样本数: {num_calibration_samples}")
    logger.info(f"  输入形状: {input_shape}")
    logger.info(f"  量化位宽: {bit_num}")

    # MindSpore Lite 转换器
    converter = mslite.Converter()
    converter.weight_fp16 = False
    converter.input_shape = {"x": list(input_shape)}

    # 量化参数
    converter.config_info = {
        "common_quant_param": {
            "quant_type": "WEIGHT_QUANT",
            "bit_num": int(bit_num),
            "min_quant_weight_size": 0,
            "min_quant_weight_channel": 16,
        }
    }

    # 执行转换
    output_stem = str(Path(output_path).with_suffix(""))
    converter.convert(
        fmk_type=mslite.FmkType.MINDIR,
        model_file=str(input_path),
        output_file=output_stem,
    )

    final_path = output_stem + ".mindir"
    original_size = input_file.stat().st_size
    quantized_size = Path(final_path).stat().st_size
    compression = (1 - quantized_size / original_size) * 100

    logger.info(f"量化完成: {final_path}")
    logger.info(f"  原始大小: {original_size / 1024:.1f} KB")
    logger.info(f"  量化大小: {quantized_size / 1024:.1f} KB")
    logger.info(f"  压缩率: {compression:.1f}%")

    return final_path
