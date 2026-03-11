"""Trainer for event-onset EMG+IMU models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from event_onset.config import EventModelConfig
from training.reporting import compute_classification_report
from training.trainer import FocalLoss, LabelSmoothingCrossEntropy, ModelEMA, build_balanced_sample_indices, compute_class_balanced_weights

try:
    import mindspore as ms
    from mindspore import Tensor, nn, ops, save_checkpoint
    import mindspore.dataset as ds
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    nn = None  # type: ignore
    ops = None  # type: ignore
    save_checkpoint = None  # type: ignore
    ds = None  # type: ignore

from shared.config import TrainingConfig

logger = logging.getLogger(__name__)


def _argmax(logits):
    if ops is None:
        raise RuntimeError("MindSpore ops is not available")
    if hasattr(ops, "Argmax"):
        return ops.Argmax(axis=1)(logits)
    return ops.argmax(logits, 1)


def _np_loader(emg_samples: np.ndarray, imu_samples: np.ndarray, labels: np.ndarray):
    for emg, imu, label in zip(emg_samples, imu_samples, labels):
        yield emg, imu, label


def create_event_dataset(
    emg_samples: np.ndarray,
    imu_samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
):
    if ds is None:
        raise RuntimeError("MindSpore dataset is not available")
    dataset = ds.GeneratorDataset(
        source=lambda: _np_loader(emg_samples, imu_samples, labels),
        column_names=["emg", "imu", "label"],
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=False)


if nn is not None:

    class EventWithLossCell(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super().__init__(auto_prefix=False)
            self.backbone = backbone
            self.loss_fn = loss_fn

        def construct(self, emg, imu, label):
            logits = self.backbone(emg, imu)
            return self.loss_fn(logits, label)

else:

    class EventWithLossCell:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("MindSpore is required for EventWithLossCell")


class EventTrainer:
    def __init__(
        self,
        model,
        model_config: EventModelConfig,
        config: TrainingConfig,
        class_names: Sequence[str],
        output_dir: str = ".",
    ):
        if ms is None:
            raise RuntimeError("MindSpore is required for EventTrainer")
        self.model = model
        self.model_config = model_config
        self.config = config
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoints" / "event_onset_best.ckpt"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._ema = ModelEMA(config.ema.decay) if config.ema.enabled else None
        self.loss_fn_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.loss_fn = None
        self.optimizer = None
        self.train_step = None

    def _build_loss(self, train_labels: np.ndarray):
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
            return FocalLoss(self.num_classes, gamma=self.config.loss.focal_gamma, class_weights=Tensor(weights, ms.float32))
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
        lr_schedule = self._build_lr(steps_per_epoch)
        optimizer = nn.AdamWeightDecay(
            self.model.trainable_params(),
            learning_rate=lr_schedule,
            weight_decay=float(self.config.weight_decay),
        )
        return optimizer, lr_schedule

    def _build_train_dataset_for_epoch(
        self,
        train_emg: np.ndarray,
        train_imu: np.ndarray,
        train_labels: np.ndarray,
        epoch: int,
    ):
        steps = int(np.ceil(len(train_labels) / self.config.batch_size))
        indices = build_balanced_sample_indices(
            train_labels,
            self.config.batch_size,
            self.config.sampler,
            steps=steps,
            seed=self.config.split_seed + epoch,
        )
        return create_event_dataset(train_emg[indices], train_imu[indices], train_labels[indices], self.config.batch_size, shuffle=False)

    def _evaluate(self, dataset) -> Dict[str, float]:
        losses: List[float] = []
        preds: List[np.ndarray] = []
        gts: List[np.ndarray] = []
        for emg, imu, label in dataset.create_tuple_iterator():
            logits = self.model(emg, imu)
            loss = self.loss_fn_ce(logits, label)
            losses.append(float(loss.asnumpy()))
            pred = _argmax(logits).asnumpy()
            preds.append(pred.astype(np.int32))
            gts.append(label.asnumpy().astype(np.int32))
        y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int32)
        y_true = np.concatenate(gts) if gts else np.empty(0, dtype=np.int32)
        report = compute_classification_report(y_true, y_pred, self.class_names)
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "acc": float(report["accuracy"]),
            "macro_f1": float(report["macro_f1"]),
        }

    def train(
        self,
        train_emg: np.ndarray,
        train_imu: np.ndarray,
        train_labels: np.ndarray,
        val_emg: np.ndarray,
        val_imu: np.ndarray,
        val_labels: np.ndarray,
    ) -> Dict[str, List[float]]:
        self.loss_fn = self._build_loss(train_labels)
        steps_per_epoch = int(np.ceil(len(train_labels) / self.config.batch_size))
        self.optimizer, lr_schedule = self._build_optimizer(steps_per_epoch)
        self.train_step = nn.TrainOneStepCell(EventWithLossCell(self.model, self.loss_fn), self.optimizer)
        self.train_step.set_train(True)
        val_dataset = create_event_dataset(val_emg, val_imu, val_labels, self.config.batch_size, shuffle=False)

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": [],
        }
        best_val_f1 = -1.0
        bad_epochs = 0

        for epoch in range(self.config.epochs):
            dataset = self._build_train_dataset_for_epoch(train_emg, train_imu, train_labels, epoch)
            train_losses: List[float] = []
            train_preds: List[np.ndarray] = []
            train_gts: List[np.ndarray] = []
            for emg, imu, label in dataset.create_tuple_iterator():
                loss = self.train_step(emg, imu, label)
                train_losses.append(float(loss.asnumpy()))
                logits = self.model(emg, imu)
                train_preds.append(_argmax(logits).asnumpy().astype(np.int32))
                train_gts.append(label.asnumpy().astype(np.int32))
                if self._ema is not None:
                    self._ema.update(self.model)

            y_pred = np.concatenate(train_preds) if train_preds else np.empty(0, dtype=np.int32)
            y_true = np.concatenate(train_gts) if train_gts else np.empty(0, dtype=np.int32)
            train_report = compute_classification_report(y_true, y_pred, self.class_names)

            backup = self._ema.apply(self.model) if self._ema is not None else None
            metrics = self._evaluate(val_dataset)
            if self._ema is not None and backup is not None:
                self._ema.restore(self.model, backup)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(float(np.mean(train_losses)) if train_losses else 0.0)
            history["train_acc"].append(float(train_report["accuracy"]))
            history["val_loss"].append(metrics["loss"])
            history["val_acc"].append(metrics["acc"])
            history["val_macro_f1"].append(metrics["macro_f1"])
            history["lr"].append(float(lr_schedule[min(epoch * steps_per_epoch, len(lr_schedule) - 1)]))

            logger.info(
                "Epoch %03d/%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch + 1,
                self.config.epochs,
                history["train_loss"][-1],
                history["train_acc"][-1],
                metrics["loss"],
                metrics["acc"],
                metrics["macro_f1"],
            )

            if metrics["macro_f1"] > best_val_f1:
                best_val_f1 = metrics["macro_f1"]
                bad_epochs = 0
                save_checkpoint(self.model, str(self.checkpoint_path))
            else:
                bad_epochs += 1

            if bad_epochs >= self.config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        return history
