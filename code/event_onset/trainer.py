"""Trainer for event-onset EMG+IMU models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from event_onset.config import EventModelConfig
from training.reporting import compute_classification_report
from training.trainer import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    ModelEMA,
    build_balanced_sample_indices,
    compute_class_balanced_weights,
)

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


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    epochs: int


def build_transfer_phase_schedule(total_epochs: int, freeze_emg_epochs: int) -> list[TrainingPhase]:
    total = max(0, int(total_epochs))
    freeze = max(0, min(int(freeze_emg_epochs), total))
    if total <= 0:
        return []
    if freeze <= 0:
        return [TrainingPhase(name="full", epochs=total)]
    phases: list[TrainingPhase] = [TrainingPhase(name="head_only", epochs=freeze)]
    remain = total - freeze
    if remain > 0:
        phases.append(TrainingPhase(name="unfreeze", epochs=remain))
    return phases


def is_encoder_param_name(name: str) -> bool:
    lowered = str(name)
    return lowered.startswith("emg_block1.") or lowered.startswith("emg_block2.") or lowered.startswith("imu_branch.")


def is_head_param_name(name: str) -> bool:
    return str(name).startswith("fusion.")


def phase_trainable(name: str, phase_name: str, *, unfreeze_last_blocks: bool) -> bool:
    if phase_name == "head_only":
        return not is_encoder_param_name(name)
    if phase_name == "unfreeze":
        if not unfreeze_last_blocks:
            return True
        # Keep the first EMG block frozen, unfreeze later blocks and head.
        return not str(name).startswith("emg_block1.")
    return True


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

    def _build_lr(self, steps_per_epoch: int, *, epochs: int, lr_scale: float = 1.0) -> List[float]:
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        total_epochs = max(1, int(epochs))
        total_steps = steps_per_epoch * total_epochs
        warmup_epochs = min(int(self.config.warmup_epochs), total_epochs)
        warmup_steps = steps_per_epoch * warmup_epochs
        base = float(self.config.learning_rate) * float(lr_scale)

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

    def _set_trainable_for_phase(self, phase_name: str) -> dict[str, int]:
        trainable_total = 0
        frozen_total = 0
        for param in self.model.get_parameters():
            enabled = phase_trainable(
                param.name,
                phase_name,
                unfreeze_last_blocks=bool(self.config.unfreeze_last_blocks),
            )
            param.requires_grad = bool(enabled)
            if enabled:
                trainable_total += 1
            else:
                frozen_total += 1
        return {"trainable_params": trainable_total, "frozen_params": frozen_total}

    def _collect_trainable_groups(self) -> tuple[list, list]:
        encoder_params = []
        head_params = []
        for param in self.model.get_parameters():
            if not bool(getattr(param, "requires_grad", True)):
                continue
            if is_head_param_name(param.name):
                head_params.append(param)
            elif is_encoder_param_name(param.name):
                encoder_params.append(param)
            else:
                # Default unknown parameters to head group so they are not starved by low LR.
                head_params.append(param)
        return encoder_params, head_params

    def _build_optimizer_for_phase(self, *, phase_name: str, phase_epochs: int, steps_per_epoch: int):
        if steps_per_epoch <= 0:
            raise ValueError("Training dataset is empty; cannot build optimizer with zero steps per epoch.")

        stats = self._set_trainable_for_phase(phase_name)
        encoder_params, head_params = self._collect_trainable_groups()
        if not encoder_params and not head_params:
            raise RuntimeError(f"No trainable parameters after phase setup: {phase_name}")

        wd = float(self.config.weight_decay)
        if phase_name == "head_only":
            head_lr = self._build_lr(
                steps_per_epoch,
                epochs=phase_epochs,
                lr_scale=float(self.config.head_lr_ratio),
            )
            optimizer = nn.AdamWeightDecay(head_params, learning_rate=head_lr, weight_decay=wd)
            return optimizer, head_lr, stats

        head_lr = self._build_lr(
            steps_per_epoch,
            epochs=phase_epochs,
            lr_scale=float(self.config.head_lr_ratio),
        )
        if not encoder_params:
            optimizer = nn.AdamWeightDecay(head_params, learning_rate=head_lr, weight_decay=wd)
            return optimizer, head_lr, stats

        encoder_lr = self._build_lr(
            steps_per_epoch,
            epochs=phase_epochs,
            lr_scale=float(self.config.encoder_lr_ratio),
        )
        param_groups = [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": wd},
            {"params": head_params, "lr": head_lr, "weight_decay": wd},
        ]
        optimizer = nn.AdamWeightDecay(
            param_groups,
            learning_rate=float(self.config.learning_rate),
            weight_decay=wd,
        )
        return optimizer, head_lr, stats

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
        train_emg: np.ndarray,
        train_imu: np.ndarray,
        train_labels: np.ndarray,
        val_emg: np.ndarray,
        val_imu: np.ndarray,
        val_labels: np.ndarray,
    ) -> Dict[str, List[float]]:
        self.loss_fn = self._build_loss(train_labels)
        steps_per_epoch = int(np.ceil(len(train_labels) / self.config.batch_size))
        phase_plan = build_transfer_phase_schedule(self.config.epochs, self.config.freeze_emg_epochs)
        if not phase_plan:
            raise ValueError("No training epochs configured.")

        val_dataset = create_event_dataset(val_emg, val_imu, val_labels, self.config.batch_size, shuffle=False)
        history = {
            "epoch": [],
            "phase": [],
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
        bad_epochs = 0
        global_epoch = 0

        for phase in phase_plan:
            phase_name = phase.name
            if phase_name == "unfreeze":
                phase_name = "unfreeze" if self.config.unfreeze_last_blocks else "full"

            self.optimizer, phase_lr_schedule, phase_stats = self._build_optimizer_for_phase(
                phase_name=phase_name,
                phase_epochs=phase.epochs,
                steps_per_epoch=steps_per_epoch,
            )
            self.train_step = nn.TrainOneStepCell(EventWithLossCell(self.model, self.loss_fn), self.optimizer)
            self.train_step.set_train(True)
            logger.info(
                "Start phase=%s epochs=%d trainable=%d frozen=%d",
                phase_name,
                phase.epochs,
                phase_stats["trainable_params"],
                phase_stats["frozen_params"],
            )

            local_step = 0
            for _ in range(phase.epochs):
                global_epoch += 1
                dataset = self._build_train_dataset_for_epoch(train_emg, train_imu, train_labels, global_epoch)
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
                    local_step += 1

                y_pred = np.concatenate(train_preds) if train_preds else np.empty(0, dtype=np.int32)
                y_true = np.concatenate(train_gts) if train_gts else np.empty(0, dtype=np.int32)
                train_report = compute_classification_report(y_true, y_pred, self.class_names)

                backup = self._ema.apply(self.model) if self._ema is not None else None
                metrics = self._evaluate(val_dataset)
                if self._ema is not None and backup is not None:
                    self._ema.restore(self.model, backup)

                train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                train_acc = float(train_report["accuracy"]) if train_preds else 0.0
                val_loss = metrics["loss"]
                val_acc = metrics["acc"]
                val_f1 = metrics["macro_f1"]
                current_lr = float(phase_lr_schedule[min(max(local_step - 1, 0), len(phase_lr_schedule) - 1)])

                history["epoch"].append(global_epoch)
                history["phase"].append(phase_name)
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_macro_f1"].append(val_f1)
                history["lr"].append(current_lr)

                logger.info(
                    "Epoch %03d/%03d phase=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f lr=%.6f",
                    global_epoch,
                    self.config.epochs,
                    phase_name,
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
                    best_epoch = global_epoch
                    bad_epochs = 0
                    self._save_best_checkpoint()
                    logger.info(" New best model: epoch=%d val_f1=%.4f val_acc=%.4f", best_epoch, best_val_f1, best_val_acc)
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.config.early_stopping_patience:
                        logger.info(
                            "Early stopping: patience=%d best_epoch=%d best_val_f1=%.4f best_val_acc=%.4f",
                            self.config.early_stopping_patience,
                            best_epoch,
                            best_val_f1,
                            best_val_acc,
                        )
                        return history

        logger.info(
            "Training finished. best_epoch=%d best_val_f1=%.4f best_val_acc=%.4f",
            best_epoch,
            best_val_f1,
            best_val_acc,
        )
        return history

    def _save_best_checkpoint(self) -> None:
        if self._ema is not None:
            backup = self._ema.apply(self.model)
            save_checkpoint(self.model, str(self.checkpoint_path))
            self._ema.restore(self.model, backup)
        else:
            save_checkpoint(self.model, str(self.checkpoint_path))
