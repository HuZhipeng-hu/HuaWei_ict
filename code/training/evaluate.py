"""
模型评估脚本

加载训练好的模型检查点，在验证集上评估：
- 整体准确率
- 各类别准确率
- 混淆矩阵
"""

import logging
from typing import Dict, Tuple

import numpy as np

try:
    import mindspore.ops as ops
    from mindspore import Tensor, load_checkpoint, load_param_into_net
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

from shared.gestures import GestureType, NUM_CLASSES

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> Dict:
    """
    评估模型

    Args:
        model: MindSpore 模型（已加载权重）
        samples: (N, C, F, T) float32
        labels: (N,) int32
        batch_size: 评估批大小

    Returns:
        {
            "accuracy": float,
            "per_class_accuracy": {gesture_name: float, ...},
            "confusion_matrix": np.ndarray (num_classes x num_classes),
            "num_samples": int,
        }
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore 未安装")

    model.set_train(False)

    all_preds = []
    all_labels = []

    # 分批推理
    num_samples = len(samples)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_data = Tensor(samples[start:end].astype(np.float32))
        batch_labels = labels[start:end]

        logits = model(batch_data)
        preds = ops.Argmax(axis=1)(logits).asnumpy()

        all_preds.extend(preds)
        all_labels.extend(batch_labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 整体准确率
    accuracy = float(np.mean(all_preds == all_labels))

    # 混淆矩阵
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1

    # 各类别准确率
    per_class = {}
    for gesture in GestureType:
        class_mask = all_labels == gesture.value
        if class_mask.sum() == 0:
            per_class[gesture.name] = 0.0
        else:
            per_class[gesture.name] = float(
                np.mean(all_preds[class_mask] == gesture.value)
            )

    results = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion,
        "num_samples": num_samples,
    }

    # 打印结果
    _print_results(results)

    return results


def _print_results(results: Dict) -> None:
    """格式化打印评估结果"""
    logger.info(f"\n{'='*50}")
    logger.info(f"评估结果 ({results['num_samples']} 个样本)")
    logger.info(f"{'='*50}")
    logger.info(f"整体准确率: {results['accuracy']:.4f}")
    logger.info(f"\n各类别准确率:")
    for name, acc in results["per_class_accuracy"].items():
        bar = "█" * int(acc * 20)
        logger.info(f"  {name:>10}: {acc:.4f} {bar}")

    logger.info(f"\n混淆矩阵:")
    header = "        " + " ".join(
        f"{g.name[:6]:>6}" for g in GestureType
    )
    logger.info(header)
    for i, gesture in enumerate(GestureType):
        row = results["confusion_matrix"][i]
        row_str = " ".join(f"{v:>6}" for v in row)
        logger.info(f"  {gesture.name[:6]:>6} {row_str}")


def load_and_evaluate(
    model,
    checkpoint_path: str,
    samples: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """
    加载检查点并评估

    Args:
        model: 未加载权重的模型实例
        checkpoint_path: .ckpt 文件路径
        samples: 评估数据
        labels: 评估标签

    Returns:
        评估结果字典
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore 未安装")

    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(model, param_dict)
    logger.info(f"已加载检查点: {checkpoint_path}")

    return evaluate_model(model, samples, labels)
