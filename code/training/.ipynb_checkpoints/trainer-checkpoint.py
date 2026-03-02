"""
训练器 — 模型训练循环
提供完整的训练流程:
- Warmup + 学习率调度（StepLR / CosineAnnealing）
- 梯度裁剪
- 早停机制
- 最优检查点自动保存
- 训练/验证指标记录
使用方式:
    trainer = Trainer(model, config)
    trainer.train(train_data, val_data)
"""
import os
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor, context
    from mindspore.train.serialization import save_checkpoint, load_checkpoint
    from mindspore.dataset import GeneratorDataset
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
from shared.config import TrainingConfig
from shared.gestures import NUM_CLASSES
logger = logging.getLogger(__name__)
# =============================================================================
# MindSpore 数据集生成器
# =============================================================================
class _ArrayDataGenerator:
    """将 numpy 数组封装为 MindSpore GeneratorDataset 所需的可迭代对象"""
    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        self.samples = samples.astype(np.float32)
        self.labels = labels.astype(np.int32)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
def create_mindspore_dataset(
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> 'GeneratorDataset':
    """
    将 numpy 数据封装为 MindSpore Dataset
    Args:
        samples: (N, C, F, T) float32
        labels: (N,) int32
        batch_size: 批大小
        shuffle: 是否打乱
    Returns:
        MindSpore GeneratorDataset
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore 未安装")
    generator = _ArrayDataGenerator(samples, labels)
    dataset = GeneratorDataset(
        source=generator,
        column_names=["data", "label"],
        shuffle=shuffle,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset
# =============================================================================
# 学习率调度
# =============================================================================
def _build_lr_schedule(config: TrainingConfig, steps_per_epoch: int) -> list:
    """
    构建学习率调度序列
    支持 Warmup + StepLR / CosineAnnealing
    Returns:
        每个 step 对应的学习率列表
    """
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    base_lr = config.learning_rate
    lr_list = []
    for step in range(total_steps):
        if step < warmup_steps:
            # 线性预热
            lr = base_lr * (step + 1) / warmup_steps
        else:
            effective_step = step - warmup_steps
            effective_total = total_steps - warmup_steps
            if config.lr_scheduler == "cosine":
                # Cosine Annealing
                lr = base_lr * 0.5 * (
                    1 + np.cos(np.pi * effective_step / effective_total)
                )
            elif config.lr_scheduler == "step":
                # Step LR
                epoch = step // steps_per_epoch
                decay = config.lr_gamma ** (
                    (epoch - config.warmup_epochs) // config.lr_step_size
                )
                lr = base_lr * decay
            else:
                lr = base_lr
        lr_list.append(float(lr))
    return lr_list
if MINDSPORE_AVAILABLE:
    class LabelSmoothingCrossEntropy(nn.Cell):
        """
        标签平滑交叉熵损失
        MindSpore 的 SoftmaxCrossEntropyWithLogits 不支持 smooth_factor 参数，
        此类通过 ops.one_hot 手动实现标签平滑，完全兼容 GRAPH_MODE。
        Args:
            num_classes: 分类类别数
            smooth_factor: 平滑系数 (0 等价于普通交叉熵)
            reduction: 'mean' / 'sum' / 'none'
        """
        def __init__(self, num_classes: int, smooth_factor: float = 0.1,
                     reduction: str = 'mean'):
            super().__init__()
            self.num_classes = num_classes
            self.on_value = Tensor(1.0 - smooth_factor, ms.float32)
            self.off_value = Tensor(smooth_factor / num_classes, ms.float32)
            self.loss = nn.SoftmaxCrossEntropyWithLogits(
                sparse=False, reduction=reduction
            )
        def construct(self, logits, labels):
            smoothed = ops.one_hot(
                labels, self.num_classes, self.on_value, self.off_value
            )
            return self.loss(logits, smoothed)
    class _TrainOneStepWithClip(nn.TrainOneStepCell):
        """带全局梯度裁剪的单步训练 Cell"""
        def __init__(self, network, optimizer, max_grad_norm: float = 1.0):
            super().__init__(network, optimizer)
            self.max_grad_norm = max_grad_norm
        # ✅ 不在 __init__ 里存函数引用，直接在 construct 里调用

        def construct(self, *inputs):
            loss = self.network(*inputs)
            sens = ops.ones_like(loss)
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            grads = self.grad_reducer(grads)

        # ✅ clip_by_global_norm 只返回裁剪后的梯度元组，不是二元组
            clipped_grads = ops.clip_by_global_norm(grads, self.max_grad_norm)

            loss = ops.depend(loss, self.optimizer(clipped_grads))
            return loss
# =============================================================================
# 训练器
# =============================================================================
class Trainer:
    """
    模型训练器
    封装完整的训练流程，包括学习率调度、梯度裁剪、早停、检查点保存。
    Args:
        model: MindSpore nn.Cell 模型实例
        config: 训练配置
    """
    def __init__(self, model, config: TrainingConfig):
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore 未安装")
        self.model = model
        self.config = config
        # 设置设备
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device)
        # 创建输出目录
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # 训练状态
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, List[float]]:
        """
        执行完整训练流程
        Args:
            train_data: (samples, labels) 训练集
            val_data: (samples, labels) 验证集
        Returns:
            训练历史 {"train_loss": [...], "val_acc": [...], ...}
        """
        train_samples, train_labels = train_data
        val_samples, val_labels = val_data
        logger.info(
            f"开始训练: "
            f"训练集 {len(train_samples)} 样本, "
            f"验证集 {len(val_samples)} 样本, "
            f"epochs={self.config.epochs}, "
            f"batch_size={self.config.batch_size}"
        )
        # 构建数据集
        train_dataset = create_mindspore_dataset(
            train_samples, train_labels,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        steps_per_epoch = train_dataset.get_dataset_size()
        # 构建学习率调度
        lr_schedule = _build_lr_schedule(self.config, steps_per_epoch)
        # 构建优化器
        optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=lr_schedule,
            weight_decay=self.config.weight_decay,
        )
        # 损失函数（支持标签平滑）
        smooth = getattr(self.config, 'label_smoothing', 0.0)
        if smooth > 0:
            loss_fn = LabelSmoothingCrossEntropy(
                num_classes=NUM_CLASSES, smooth_factor=smooth, reduction='mean',
            )
            logger.info(f"标签平滑: smooth_factor={smooth}")
        else:
            loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        # 验证专用损失函数（始终硬标签，不受标签平滑影响）
        eval_loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        # 训练网络（带梯度裁剪）
        grad_clip = getattr(self.config, 'gradient_clip', 0.0)
        if grad_clip > 0:
            train_net = _TrainOneStepWithClip(
                nn.WithLossCell(self.model, loss_fn),
                optimizer,
                max_grad_norm=grad_clip,
            )
            logger.info(f"梯度裁剪: max_norm={grad_clip}")
        else:
            train_net = nn.TrainOneStepCell(
                nn.WithLossCell(self.model, loss_fn),
                optimizer,
            )
        # 训练循环
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            # --- 训练阶段 ---
            self.model.set_train(True)
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch in train_dataset.create_dict_iterator():
                data = batch["data"]
                label = batch["label"]
                # 前向 + 反向
                loss = train_net(data, label)
                # 统计 loss
                train_loss += float(loss.asnumpy())
                train_total += len(label)
            train_loss /= max(steps_per_epoch, 1)
            # 训练集准确率在验证阶段统一计算（避免双重前向传播浪费算力）
            train_acc = 0.0 # 将在下面用验证函数计算
            # --- 训练集准确率（用评估函数，避免训练循环内双重前向）---
            _, train_acc = self._evaluate(train_samples, train_labels, eval_loss_fn)
            # --- 验证阶段 ---
            val_loss, val_acc = self._evaluate(val_samples, val_labels, eval_loss_fn)
            # 记录当前学习率
            current_lr = lr_schedule[min(
                epoch * steps_per_epoch,
                len(lr_schedule) - 1,
            )]
            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} "
                f"({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            # --- 检查点保存 ---
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.no_improve_count = 0
                self._save_checkpoint(epoch + 1, val_acc, is_best=True)
                logger.info(f" ★ 新最佳模型! Val Acc: {val_acc:.4f}")
            else:
                self.no_improve_count += 1
            # --- 早停检查 ---
            if self.no_improve_count >= self.config.early_stop_patience:
                logger.info(
                    f"早停: 连续 {self.no_improve_count} 轮无改善。"
                    f"最佳 Epoch: {self.best_epoch}, "
                    f"最佳 Val Acc: {self.best_val_acc:.4f}"
                )
                break
        logger.info(
            f"训练完成! 最佳 Val Acc: {self.best_val_acc:.4f} "
            f"(Epoch {self.best_epoch})"
        )
        # 保存训练历史
        self._save_history()
        return self.history
    def _evaluate(
        self,
        val_samples: np.ndarray,
        val_labels: np.ndarray,
        loss_fn,
    ) -> Tuple[float, float]:
        """评估模型在验证集上的表现"""
        self.model.set_train(False)
        val_dataset = create_mindspore_dataset(
            val_samples, val_labels,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        for batch in val_dataset.create_dict_iterator():
            data = batch["data"]
            label = batch["label"]
            logits = self.model(data)
            loss = loss_fn(logits, label)
            pred = ops.Argmax(axis=1)(logits)
            correct += int((pred == label).sum().asnumpy())
            total += len(label)
            total_loss += float(loss.asnumpy())
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
    def _save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """保存模型检查点"""
        if is_best:
            save_path = self.ckpt_dir / "neurogrip_best.ckpt"
        else:
            save_path = self.ckpt_dir / f"neurogrip_epoch{epoch}.ckpt"
        save_checkpoint(self.model, str(save_path))
        logger.info(f" 检查点已保存: {save_path}")
    def _save_history(self) -> None:
        """将训练历史保存为 CSV"""
        history_path = self.log_dir / "training_history.csv"
        with open(history_path, 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
            for i in range(len(self.history["train_loss"])):
                f.write(
                    f"{i+1},"
                    f"{self.history['train_loss'][i]:.6f},"
                    f"{self.history['train_acc'][i]:.6f},"
                    f"{self.history['val_loss'][i]:.6f},"
                    f"{self.history['val_acc'][i]:.6f},"
                    f"{self.history['lr'][i]:.8f}\n"
                )
        logger.info(f"训练历史已保存: {history_path}")