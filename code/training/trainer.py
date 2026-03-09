"""Model trainer with balanced sampling, focal-family losses and EMA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from shared.config import SamplerConfig, TrainingConfig
from shared.gestures import GESTURE_DEFINITIONS

try:
    import mindspore as ms
    from mindspore import Tensor, nn, ops, save_checkpoint
    import mindspore.dataset as ds
except Exception:  # pragma: no cover
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    nn = None  # type: ignore
    ops = None  # type: ignore
    save_checkpoint = None  # type: ignore
    ds = None  # type: ignore

from training.reporting import compute_classification_report

logger = logging.getLogger(__name__)


def compute_class_balanced_weights(labels: np.ndarray, num_classes: int, beta: float) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int32), minlength=num_classes).astype(np.float32)
    effective_num = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / np.sum(weights) * num_classes
    return weights.astype(np.float32)


def _resolve_confusion_class_ids(confusion_pairs: Sequence[Sequence], gesture_to_idx: Dict[str, int]) -> set[int]:
    ids: set[int] = set()
    for pair in confusion_pairs:
        if len(pair) != 2:
            continue
        for item in pair:
            if isinstance(item, int):
                ids.add(item)
            elif isinstance(item, str):
                if item in gesture_to_idx:
                    ids.add(gesture_to_idx[item])
    return ids


def build_balanced_sample_indices(
    labels: np.ndarray,
    batch_size: int,
    sampler_cfg: SamplerConfig,
    *,
    steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int32)
    class_ids = np.unique(labels)
    class_indices: Dict[int, np.ndarray] = {int(c): np.where(labels == c)[0] for c in class_ids}

    if sampler_cfg.type.lower() != "balanced":
        base = np.arange(labels.shape[0], dtype=np.int32)
        rng.shuffle(base)
        repeats = int(np.ceil((steps * batch_size) / max(1, len(base))))
        expanded = np.tile(base, repeats)[: steps * batch_size]
        return expanded.astype(np.int32)

    n_classes = len(class_ids)
    per_class = max(1, batch_size // max(1, n_classes))
    remainder = max(0, batch_size - per_class * n_classes)

    gesture_to_idx = {g.name: i for i, g in enumerate(GESTURE_DEFINITIONS)}
    hard_ids = _resolve_confusion_class_ids(sampler_cfg.confusion_pairs, gesture_to_idx)

    class_weights = np.ones(n_classes, dtype=np.float32)
    if hard_ids:
        for i, cls in enumerate(class_ids):
            if int(cls) in hard_ids:
                class_weights[i] += float(max(0.0, sampler_cfg.hard_mining_ratio))
    class_probs = class_weights / np.sum(class_weights)

    all_indices: List[int] = []
    for _ in range(steps):
        batch_idx: List[int] = []
        for cls in class_ids:
            pool = class_indices[int(cls)]
            chosen = rng.choice(pool, size=per_class, replace=len(pool) < per_class)
            batch_idx.extend(chosen.tolist())

        if remainder > 0:
            rem_classes = rng.choice(class_ids, size=remainder, replace=True, p=class_probs)
            for cls in rem_classes:
                pool = class_indices[int(cls)]
                batch_idx.append(int(rng.choice(pool)))

        rng.shuffle(batch_idx)
        all_indices.extend(batch_idx[:batch_size])

    return np.asarray(all_indices, dtype=np.int32)


def _np_loader(samples: np.ndarray, labels: np.ndarray):
    for x, y in zip(samples, labels):
        yield x, y


def create_mindspore_dataset(
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
) -> "ds.Dataset":
    if ds is None:
        raise RuntimeError("MindSpore dataset is not available")

    dataset = ds.GeneratorDataset(
        source=lambda: _np_loader(samples, labels),
        column_names=["sample", "label"],
        shuffle=shuffle,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


if nn is not None and ops is not None and Tensor is not None and ms is not None:

    class LabelSmoothingCrossEntropy(nn.Cell):
        def __init__(self, num_classes: int, smoothing: float = 0.1):
            super().__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            self.log_softmax = nn.LogSoftmax(axis=1)
            self.reduce_mean = ops.ReduceMean()
            self.on_value = Tensor(1.0 - smoothing, ms.float32)
            self.off_value = Tensor(smoothing / max(1, num_classes - 1), ms.float32)

        def construct(self, logits, labels):
            one_hot = ops.one_hot(labels, self.num_classes, self.on_value, self.off_value)
            log_probs = self.log_softmax(logits)
            loss = -ops.ReduceSum()(one_hot * log_probs, 1)
            return self.reduce_mean(loss)


    class FocalLoss(nn.Cell):
        def __init__(self, num_classes: int, gamma: float = 1.5, class_weights: Tensor | None = None):
            super().__init__()
            self.num_classes = num_classes
            self.gamma = Tensor(float(gamma), ms.float32)
            self.class_weights = class_weights
            self.one = Tensor(1.0, ms.float32)
            self.eps = Tensor(1e-6, ms.float32)

        def construct(self, logits, labels):
            probs = ops.softmax(logits, axis=1)
            one_hot = ops.one_hot(labels, self.num_classes, self.one, Tensor(0.0, ms.float32))
            pt = ops.ReduceSum()(probs * one_hot, 1)
            pt = ops.clip_by_value(pt, self.eps, self.one)
            ce = -ops.log(pt)
            focal = ops.pow(self.one - pt, self.gamma) * ce
            if self.class_weights is not None:
                gathered = ops.Gather()(self.class_weights, labels, 0)
                focal = focal * gathered
            return ops.ReduceMean()(focal)

else:

    class LabelSmoothingCrossEntropy:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MindSpore is required for LabelSmoothingCrossEntropy")


    class FocalLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MindSpore is required for FocalLoss")


@dataclass
class _EMAState:
    shadow: Dict[str, np.ndarray]
    decay: float


class ModelEMA:
    def __init__(self, decay: float = 0.999):
        self.state = _EMAState(shadow={}, decay=float(decay))

    def update(self, model: nn.Cell) -> None:
        for p in model.get_parameters():
            arr = p.asnumpy()
            if p.name not in self.state.shadow:
                self.state.shadow[p.name] = arr.copy()
            else:
                self.state.shadow[p.name] = (
                    self.state.decay * self.state.shadow[p.name] + (1.0 - self.state.decay) * arr
                )

    def apply(self, model: nn.Cell) -> Dict[str, np.ndarray]:
        backup: Dict[str, np.ndarray] = {}
        for p in model.get_parameters():
            backup[p.name] = p.asnumpy().copy()
            if p.name in self.state.shadow:
                p.set_data(Tensor(self.state.shadow[p.name], p.dtype))
        return backup

    @staticmethod
    def restore(model: nn.Cell, backup: Dict[str, np.ndarray]) -> None:
        for p in model.get_parameters():
            if p.name in backup:
                p.set_data(Tensor(backup[p.name], p.dtype))


class Trainer:
    def __init__(
        self,
        model: nn.Cell,
        config: TrainingConfig,
        class_names: Sequence[str],
        output_dir: str = ".",
    ):
        if ms is None:
            raise RuntimeError("MindSpore is required for Trainer")
        self.model = model
        self.config = config
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoints" / "neurogrip_best.ckpt"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self._ema = ModelEMA(config.ema.decay) if config.ema.enabled else None

        self.optimizer = None
        self.loss_fn_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.loss_fn: nn.Cell | None = None
        self.train_step = None

    def _build_loss(self, train_labels: np.ndarray) -> nn.Cell:
        loss_type = self.config.loss.type.lower()
        if loss_type in {"ce", "cross_entropy"}:
            if self.config.label_smoothing > 0:
                return LabelSmoothingCrossEntropy(self.num_classes, self.config.label_smoothing)
            return nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

        if loss_type == "focal":
            return FocalLoss(self.num_classes, gamma=self.config.loss.focal_gamma, class_weights=None)

        if loss_type in {"cb_focal", "class_balanced_focal"}:
            weights = compute_class_balanced_weights(
                train_labels, self.num_classes, beta=self.config.loss.class_balance_beta
            )
            tensor_weights = Tensor(weights, ms.float32)
            return FocalLoss(self.num_classes, gamma=self.config.loss.focal_gamma, class_weights=tensor_weights)

        raise ValueError(f"Unsupported loss type: {self.config.loss.type}")

    def _build_lr(self, steps_per_epoch: int) -> List[float]:
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        base = float(self.config.learning_rate)

        lrs: List[float] = []
        for step in range(total_steps):
            if warmup_steps > 0 and step < warmup_steps:
                lr = base * float(step + 1) / float(warmup_steps)
            else:
                progress = 0.0
                if total_steps > warmup_steps:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                lr = base * 0.5 * (1.0 + np.cos(np.pi * progress))
            lrs.append(float(lr))
        return lrs

    def _build_optimizer(self, steps_per_epoch: int):
        if steps_per_epoch <= 0:
            raise ValueError("Training dataset is empty; cannot build optimizer with zero steps per epoch.")

        lr_schedule = self._build_lr(steps_per_epoch)
        if not lr_schedule:
            raise ValueError("Learning-rate schedule is empty; check training epochs and batch size.")

        optimizer = nn.AdamWeightDecay(
            self.model.trainable_params(),
            learning_rate=lr_schedule,
            weight_decay=float(self.config.weight_decay),
        )
        return optimizer, lr_schedule

    def _build_train_dataset_for_epoch(
        self,
        train_samples: np.ndarray,
        train_labels: np.ndarray,
        epoch: int,
    ) -> "ds.Dataset":
        steps = int(np.ceil(len(train_labels) / self.config.batch_size))
        indices = build_balanced_sample_indices(
            train_labels,
            self.config.batch_size,
            self.config.sampler,
            steps=steps,
            seed=self.config.split_seed + epoch,
        )
        batch_samples = train_samples[indices]
        batch_labels = train_labels[indices]
        return create_mindspore_dataset(batch_samples, batch_labels, self.config.batch_size, shuffle=False)

    def _evaluate(self, dataset: "ds.Dataset") -> Dict[str, float]:
        losses: List[float] = []
        preds: List[np.ndarray] = []
        gts: List[np.ndarray] = []
        for sample, label in dataset.create_tuple_iterator():
            logits = self.model(sample)
            loss = self.loss_fn_ce(logits, label)
            losses.append(float(loss.asnumpy()))
            pred = ops.argmax(logits, axis=1).asnumpy()
            preds.append(pred.astype(np.int32))
            gts.append(label.asnumpy().astype(np.int32))

        if not preds:
            return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(gts)
        report = compute_classification_report(y_true, y_pred, self.class_names)
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "acc": float(report["accuracy"]),
            "macro_f1": float(report["macro_f1"]),
        }

    def train(
        self,
        train_samples: np.ndarray,
        train_labels: np.ndarray,
        val_samples: np.ndarray,
        val_labels: np.ndarray,
    ) -> Dict[str, List[float]]:
        self.loss_fn = self._build_loss(train_labels)
        steps_per_epoch = int(np.ceil(len(train_labels) / self.config.batch_size))
        self.optimizer, lr_schedule = self._build_optimizer(steps_per_epoch)
        self.train_step = nn.TrainOneStepCell(nn.WithLossCell(self.model, self.loss_fn), self.optimizer)
        self.train_step.set_train(True)

        val_dataset = create_mindspore_dataset(val_samples, val_labels, self.config.batch_size, shuffle=False)

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": [],
        }

        best_epoch = -1
        best_val_acc = -1.0
        best_val_f1 = -1.0
        patience = 0
        global_step = 0

        for epoch in range(1, self.config.epochs + 1):
            epoch_train_dataset = self._build_train_dataset_for_epoch(train_samples, train_labels, epoch)
            epoch_losses: List[float] = []

            for sample, label in epoch_train_dataset.create_tuple_iterator():
                loss = self.train_step(sample, label)
                epoch_losses.append(float(loss.asnumpy()))
                global_step += 1

            if self._ema is not None:
                self._ema.update(self.model)

            train_eval_dataset = create_mindspore_dataset(
                train_samples, train_labels, self.config.batch_size, shuffle=False
            )

            if self._ema is not None:
                backup = self._ema.apply(self.model)
            else:
                backup = {}

            train_metrics = self._evaluate(train_eval_dataset)
            val_metrics = self._evaluate(val_dataset)

            if self._ema is not None:
                self._ema.restore(self.model, backup)

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else train_metrics["loss"]
            train_acc = train_metrics["acc"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            val_f1 = val_metrics["macro_f1"]
            current_lr = lr_schedule[min(global_step - 1, len(lr_schedule) - 1)]

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_f1)
            history["lr"].append(current_lr)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f F1: %.4f | LR: %.6f",
                epoch,
                self.config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1,
                current_lr,
            )

            improved = (val_f1 > best_val_f1 + 1e-6) or (
                abs(val_f1 - best_val_f1) <= 1e-6 and val_acc > best_val_acc + 1e-6
            )
            if improved:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_epoch = epoch
                patience = 0
                self._save_best_checkpoint()
                logger.info(" ★ 新最佳模型! Val F1: %.4f, Val Acc: %.4f", best_val_f1, best_val_acc)
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    logger.info(
                        "早停: 连续 %d 轮无改善。最佳 Epoch: %d, 最佳 Val F1: %.4f, Val Acc: %.4f",
                        self.config.early_stopping_patience,
                        best_epoch,
                        best_val_f1,
                        best_val_acc,
                    )
                    break

        logger.info("训练完成! 最佳 Val F1: %.4f, Val Acc: %.4f (Epoch %d)", best_val_f1, best_val_acc, best_epoch)
        return history

    def _save_best_checkpoint(self) -> None:
        if self._ema is not None:
            backup = self._ema.apply(self.model)
            save_checkpoint(self.model, str(self.checkpoint_path))
            self._ema.restore(self.model, backup)
        else:
            save_checkpoint(self.model, str(self.checkpoint_path))

