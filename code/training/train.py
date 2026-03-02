"""
训练入口脚本

使用方式:
    python -m training.train --config configs/training.yaml --data_dir data/

参数:
    --config    训练配置 YAML 文件路径
    --data_dir  数据目录路径（包含手势子文件夹）
    --epochs    覆盖配置中的训练轮数（可选）
    --device    覆盖配置中的设备（可选）
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import (
    TrainingConfig, ModelConfig, PreprocessConfig,
    AugmentationConfig, load_training_config,
)
from shared.models import create_model, count_parameters
from shared.preprocessing import PreprocessPipeline
from shared.gestures import validate_gesture_definitions, NUM_CLASSES

from training.data.csv_dataset import CSVDatasetLoader
from training.data.augmentation import DataAugmentor
from training.trainer import Trainer
from training.evaluate import evaluate_model

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroGrip Pro V2 — 模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/training.yaml",
        help="训练配置 YAML 文件路径",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="数据目录路径（包含 Relax/, fist/, Pinch/ 等子文件夹）",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖配置中的训练轮数",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="覆盖配置中的批大小",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["CPU", "GPU", "Ascend"],
        help="覆盖配置中的训练设备",
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="禁用数据增强",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("NeuroGrip Pro V2 — 模型训练")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 加载配置
    # -------------------------------------------------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"加载配置: {config_path}")
        model_config, preprocess_config, train_config, aug_config = \
            load_training_config(str(config_path))
    else:
        logger.info("未找到配置文件，使用默认配置")
        model_config = ModelConfig()
        preprocess_config = PreprocessConfig()
        train_config = TrainingConfig()
        aug_config = AugmentationConfig()

    # CLI 参数覆盖
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.device is not None:
        train_config.device = args.device
    if args.no_augment:
        aug_config.enabled = False

    # -------------------------------------------------------------------------
    # 2. 验证手势定义
    # -------------------------------------------------------------------------
    validate_gesture_definitions()
    logger.info(f"手势定义验证通过: {NUM_CLASSES} 类")

    # -------------------------------------------------------------------------
    # 3. 构建预处理流水线
    # -------------------------------------------------------------------------
    pipeline = PreprocessPipeline(
        sampling_rate=preprocess_config.sampling_rate,
        num_channels=preprocess_config.num_channels,
        lowcut=preprocess_config.lowcut,
        highcut=preprocess_config.highcut,
        filter_order=preprocess_config.filter_order,
        stft_window_size=preprocess_config.stft_window_size,
        stft_hop_size=preprocess_config.stft_hop_size,
        stft_n_fft=preprocess_config.stft_n_fft,
    )

    # -------------------------------------------------------------------------
    # 4. 加载数据
    # -------------------------------------------------------------------------
    logger.info(f"加载数据: {args.data_dir}")
    loader = CSVDatasetLoader(
        data_dir=args.data_dir,
        preprocess=pipeline,
        num_emg_channels=preprocess_config.total_channels,
        device_sampling_rate=preprocess_config.device_sampling_rate,
        target_sampling_rate=int(preprocess_config.sampling_rate),
        segment_length=preprocess_config.segment_length,
        segment_stride=preprocess_config.segment_stride,
    )

    # 打印数据统计
    stats = loader.get_stats()
    logger.info(f"数据统计: {stats}")

    # 加载所有样本
    samples, labels = loader.load_all()
    logger.info(f"样本总数: {len(samples)}, 形状: {samples.shape}")

    # -------------------------------------------------------------------------
    # 5. 数据增强
    # -------------------------------------------------------------------------
    if aug_config.enabled:
        augmentor = DataAugmentor(
            time_warp_rate=aug_config.time_warp_rate,
            amplitude_scale=aug_config.amplitude_scale,
            noise_std=aug_config.noise_std,
            mixup_alpha=getattr(aug_config, 'mixup_alpha', 0.2),
        )
        use_mixup = getattr(aug_config, 'use_mixup', False)
        expected = len(samples) * aug_config.augment_factor
        if use_mixup:
            expected += len(samples)
        logger.info(
            f"执行数据增强 (x{aug_config.augment_factor}"
            f"{'+Mixup' if use_mixup else ''}): "
            f"{len(samples)} → ~{expected}"
        )
        samples, labels = augmentor.augment_batch(
            samples, labels,
            factor=aug_config.augment_factor,
            use_mixup=use_mixup,
        )
        logger.info(f"增强后样本数: {len(samples)}")

    # -------------------------------------------------------------------------
    # 6. 构建模型辅助函数
    # -------------------------------------------------------------------------
    def _create_fresh_model():
        """创建一个全新的模型实例"""
        return create_model({
            "model_type": model_config.model_type,
            "in_channels": model_config.in_channels,
            "num_classes": model_config.num_classes,
            "base_channels": model_config.base_channels,
            "use_se": model_config.use_se,
            "dropout_rate": model_config.dropout_rate,
        })

    # -------------------------------------------------------------------------
    # 7. 训练（支持 K-Fold 和普通模式）
    # -------------------------------------------------------------------------
    kfold = getattr(train_config, 'kfold', 0)

    if kfold > 0:
        # ===== K-Fold 交叉验证模式 =====
        logger.info(f"\n{'='*60}")
        logger.info(f"K-Fold 交叉验证: {kfold} 折")
        logger.info(f"{'='*60}")

        fold_accs = []
        for fold_idx, (train_data, val_data) in \
                CSVDatasetLoader.kfold_split(samples, labels, k=kfold):
            logger.info(f"\n--- Fold {fold_idx+1}/{kfold} ---")
            logger.info(
                f"训练: {len(train_data[0])} 样本, "
                f"验证: {len(val_data[0])} 样本"
            )

            # 每折独立模型和 trainer
            fold_model = _create_fresh_model()
            fold_trainer = Trainer(fold_model, train_config)
            fold_history = fold_trainer.train(
                train_data=train_data,
                val_data=val_data,
            )
            fold_accs.append(fold_trainer.best_val_acc)
            logger.info(
                f"Fold {fold_idx+1} 最佳 Val Acc: "
                f"{fold_trainer.best_val_acc:.4f}"
            )

        # 汇总
        import numpy as np
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        logger.info(f"\n{'='*60}")
        logger.info(
            f"K-Fold 结果: "
            f"平均 Val Acc = {mean_acc:.4f} ± {std_acc:.4f}"
        )
        logger.info(f"各折: {[f'{a:.4f}' for a in fold_accs]}")
        logger.info(f"{'='*60}")

        # K-Fold 后用全部数据做最终训练
        logger.info("\n使用全部数据做最终训练...")
        model = _create_fresh_model()
        # 用最后一折的 val 做最终评估
        (train_samples, train_labels), (val_samples, val_labels) = \
            CSVDatasetLoader.split(
                samples, labels, val_ratio=train_config.val_ratio,
            )

    else:
        # ===== 普通训练模式 =====
        (train_samples, train_labels), (val_samples, val_labels) = \
            CSVDatasetLoader.split(
                samples, labels, val_ratio=train_config.val_ratio,
            )
        model = _create_fresh_model()

    num_params = count_parameters(model)
    logger.info(
        f"模型创建完成: {model_config.model_type}, "
        f"参数量: {num_params:,}"
    )
    logger.info(
        f"数据划分: 训练 {len(train_samples)} / 验证 {len(val_samples)}"
    )

    # -------------------------------------------------------------------------
    # 8. 最终训练
    # -------------------------------------------------------------------------
    trainer = Trainer(model, train_config)
    history = trainer.train(
        train_data=(train_samples, train_labels),
        val_data=(val_samples, val_labels),
    )

    # -------------------------------------------------------------------------
    # 9. 最终评估
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("最终评估（最佳模型 + 验证集）")
    logger.info("=" * 60)

    # 重新加载最佳检查点
    from training.evaluate import load_and_evaluate
    best_ckpt = Path(train_config.checkpoint_dir) / "neurogrip_best.ckpt"
    if best_ckpt.exists():
        eval_model = create_model({
            "model_type": model_config.model_type,
            "in_channels": model_config.in_channels,
            "num_classes": model_config.num_classes,
            "base_channels": model_config.base_channels,
            "use_se": model_config.use_se,
            "dropout_rate": 0.0,  # 评估时关闭 Dropout
        })
        results = load_and_evaluate(
            eval_model, str(best_ckpt), val_samples, val_labels,
        )

    elapsed = time.time() - start_time
    logger.info(f"\n训练完成! 总耗时: {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()
