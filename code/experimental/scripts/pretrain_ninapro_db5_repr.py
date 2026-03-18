"""Representation pretraining on NinaPro DB5 with supervised contrastive objective."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experimental.ninapro_db5.config import load_db5_pretrain_config
from experimental.ninapro_db5.dataset import DB5PretrainDatasetLoader
from experimental.ninapro_db5.evaluate import _set_device
from experimental.ninapro_db5.model import build_db5_encoder_model
from experimental.ninapro_db5.repr_pretrain_utils import (
    augment_batch as _augment_batch,
    build_class_source_balanced_indices as _build_class_source_balanced_indices,
    build_quality_aware_positive_pairs as _build_quality_aware_positive_pairs,
    build_training_indices as _build_training_indices,
    parse_bool_arg as _parse_bool_arg,
    resolve_augmentation_params as _resolve_augmentation_params,
    resolve_epoch_augmentation_params as _resolve_epoch_augmentation_params,
    resolve_repr_objective as _resolve_repr_objective,
    resolve_sampler_mode as _resolve_sampler_mode,
)
from shared.config import load_config
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, dump_yaml, ensure_run_dir
from training.data.split_strategy import build_manifest
from training.reporting import compute_classification_report, save_classification_report
from training.trainer import ModelEMA

try:
    import mindspore as ms
    from mindspore import Tensor, nn, ops, save_checkpoint
except Exception:  # pragma: no cover
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    nn = None  # type: ignore
    ops = None  # type: ignore
    save_checkpoint = None  # type: ignore


SUMMARY_FIELDS = [
    "run_id",
    "checkpoint_path",
    "foundation_version",
    "repr_objective",
    "num_classes",
    "best_val_epoch",
    "best_val_macro_f1",
    "best_val_acc",
    "test_macro_f1",
    "test_accuracy",
    "temperature",
    "projection_dim",
    "knn_k",
    "sampler_mode",
    "augmentation_profile",
    "augment_scale_min",
    "augment_scale_max",
    "augment_noise_std",
    "augment_channel_drop_ratio",
    "augment_time_mask_ratio",
    "augment_freq_mask_ratio",
    "ce_weight",
    "label_smoothing",
    "contrastive_weight",
    "temporal_weight",
    "recon_weight",
    "pairing_mode",
    "pair_cross_source_ratio",
    "pair_top_quality_ratio",
    "quality_sampling_mode",
    "final_curriculum_alpha",
    "learning_rate",
    "weight_decay",
    "manifest_use_source_metadata",
    "split_seed",
    "run_mode",
]


def _build_split_diagnostics(manifest, class_names: list[str]) -> dict:
    class_dist = dict(getattr(manifest, "class_distribution", {}) or {})
    by_split: dict[str, dict] = {}
    has_any_empty = False
    for split_name in ("train", "val", "test"):
        raw_counts = dict(class_dist.get(split_name, {}) or {})
        counts = {name: int(raw_counts.get(name, 0)) for name in class_names}
        empty = [name for name, count in counts.items() if int(count) <= 0]
        has_empty = bool(empty)
        has_any_empty = has_any_empty or has_empty
        by_split[split_name] = {
            "class_counts": counts,
            "min_class_count": int(min(counts.values()) if counts else 0),
            "empty_classes": empty,
            "has_empty_classes": has_empty,
        }
    return {"by_split": by_split, "overall": {"has_any_empty_classes": bool(has_any_empty)}}


def _has_group_leakage(manifest) -> bool:
    train = set(getattr(manifest, "group_keys_train", []) or [])
    val = set(getattr(manifest, "group_keys_val", []) or [])
    test = set(getattr(manifest, "group_keys_test", []) or [])
    return bool((train & val) or (train & test) or (val & test))


if nn is not None and ops is not None and Tensor is not None and ms is not None:

    class ProjectionHead(nn.Cell):
        def __init__(self, in_dim: int, projection_dim: int):
            super().__init__()
            self.layers = nn.SequentialCell(
                [
                    nn.Dense(int(in_dim), int(projection_dim)),
                    nn.ReLU(),
                    nn.Dense(int(projection_dim), int(projection_dim)),
                ]
            )

        def construct(self, x):
            return self.layers(x)


    class RepresentationNet(nn.Cell):
        def __init__(
            self,
            encoder,
            *,
            projection_dim: int,
            embedding_dim: int,
            num_classes: int,
            temporal_dim: int,
        ):
            super().__init__()
            self.encoder = encoder
            self.projector = ProjectionHead(int(embedding_dim), int(projection_dim))
            self.classifier = nn.Dense(int(embedding_dim), int(num_classes))
            self.temporal_head = nn.Dense(int(embedding_dim), int(max(4, temporal_dim)))
            self.recon_head = nn.Dense(int(embedding_dim), 3)
            self.l2 = ops.L2Normalize(axis=1) if hasattr(ops, "L2Normalize") else None
            self.reduce_sum = ops.ReduceSum(keep_dims=True)
            self.sqrt = ops.Sqrt()

        def _normalize(self, x):
            if self.l2 is not None:
                return self.l2(x)
            sq = self.reduce_sum(x * x, 1)
            sq = ops.clip_by_value(sq, Tensor(1e-12, ms.float32), Tensor(1e12, ms.float32))
            return x / self.sqrt(sq)

        def construct(self, x):
            feat = self.encoder(x)
            proj = self.projector(feat)
            logits = self.classifier(feat)
            temporal_repr = self.temporal_head(feat)
            recon_stats = self.recon_head(feat)
            return feat, self._normalize(proj), logits, temporal_repr, recon_stats


    class LabelSmoothedCrossEntropy(nn.Cell):
        def __init__(self, num_classes: int, smoothing: float):
            super().__init__()
            self.num_classes = int(num_classes)
            self.smoothing = float(max(0.0, min(0.2, smoothing)))
            self.log_softmax = nn.LogSoftmax(axis=1)
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.reduce_mean = ops.ReduceMean(keep_dims=False)
            self.on_value = Tensor(1.0 - self.smoothing, ms.float32)
            off = 0.0 if self.num_classes <= 1 else self.smoothing / float(max(1, self.num_classes - 1))
            self.off_value = Tensor(off, ms.float32)

        def construct(self, logits, labels):
            one_hot = ops.one_hot(labels, self.num_classes, self.on_value, self.off_value)
            log_probs = self.log_softmax(logits)
            loss = -self.reduce_sum(one_hot * log_probs, 1)
            return self.reduce_mean(loss)


    class SupervisedContrastiveLoss(nn.Cell):
        def __init__(self, temperature: float):
            super().__init__()
            self.temperature = Tensor(float(temperature), ms.float32)
            self.cast = ops.Cast()
            self.equal = ops.Equal()
            self.eye = ops.Eye()
            self.maximum = ops.Maximum()
            self.reduce_sum_keep = ops.ReduceSum(keep_dims=True)
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.reduce_mean = ops.ReduceMean(keep_dims=False)
            self.reduce_max = ops.ReduceMax(keep_dims=True)
            self.log = ops.Log()
            self.exp = ops.Exp()

        def construct(self, features, labels):
            labels = ops.reshape(labels, (-1, 1))
            label_t = ops.transpose(labels, (1, 0))
            mask = self.cast(self.equal(labels, label_t), ms.float32)
            count = features.shape[0]
            if int(count) <= 1:
                return Tensor(0.0, ms.float32)

            logits = ops.matmul(features, ops.transpose(features, (1, 0))) / self.temperature
            logits = logits - ops.stop_gradient(self.reduce_max(logits, 1))
            eye = self.eye(int(count), int(count), ms.float32)
            logits_mask = Tensor(1.0, ms.float32) - eye
            mask = mask * logits_mask
            exp_logits = self.exp(logits) * logits_mask
            log_prob = logits - self.log(self.reduce_sum_keep(exp_logits, 1) + Tensor(1e-12, ms.float32))
            pos_count = self.maximum(self.reduce_sum(mask, 1), Tensor(1.0, ms.float32))
            mean_log_prob_pos = self.reduce_sum(mask * log_prob, 1) / pos_count
            return -self.reduce_mean(mean_log_prob_pos)


    class ReprTrainCell(nn.Cell):
        def __init__(
            self,
            model,
            supcon_loss_fn,
            *,
            objective: str,
            use_ce: bool,
            ce_loss_fn=None,
            ce_weight: float = 0.0,
            contrastive_weight: float = 1.0,
            temporal_weight: float = 0.3,
            recon_weight: float = 0.2,
        ):
            super().__init__(auto_prefix=False)
            self.model = model
            self.supcon_loss_fn = supcon_loss_fn
            self.objective = str(objective)
            self.use_ce = bool(use_ce)
            self.ce_loss_fn = ce_loss_fn
            self.ce_weight = Tensor(float(max(0.0, ce_weight)), ms.float32)
            self.contrastive_weight = Tensor(float(max(0.0, contrastive_weight)), ms.float32)
            self.temporal_weight = Tensor(float(max(0.0, temporal_weight)), ms.float32)
            self.recon_weight = Tensor(float(max(0.0, recon_weight)), ms.float32)
            self.mul = ops.Mul()
            self.add = ops.Add()
            self.concat = ops.Concat(axis=0)
            self.cast = ops.Cast()
            self.abs = ops.Abs()
            self.square = ops.Square()
            self.sqrt = ops.Sqrt()
            self.reduce_mean = ops.ReduceMean(keep_dims=False)
            self.reduce_sum_keep = ops.ReduceSum(keep_dims=True)
            self.stack = ops.Stack(axis=1)
            self.l2 = ops.L2Normalize(axis=1) if hasattr(ops, "L2Normalize") else None

        def _mse(self, left, right):
            return self.reduce_mean(self.square(left - right))

        def _normalize(self, x):
            if self.l2 is not None:
                return self.l2(x)
            sq = self.reduce_sum_keep(x * x, 1)
            sq = ops.clip_by_value(sq, Tensor(1e-12, ms.float32), Tensor(1e12, ms.float32))
            return x / self.sqrt(sq)

        def _window_stats_target(self, x):
            # x: [B, C, F, T], build lightweight reconstruction target [B, 3]
            abs_mean = self.reduce_mean(self.abs(x), (1, 2, 3))
            raw_mean = self.reduce_mean(x, (1, 2, 3))
            centered = x - ops.reshape(raw_mean, (-1, 1, 1, 1))
            variance = self.reduce_mean(self.square(centered), (1, 2, 3))
            std = self.sqrt(variance + Tensor(1e-12, ms.float32))
            t_len = int(x.shape[3])
            if t_len > 1:
                temporal_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
                temporal_energy = self.reduce_mean(self.abs(temporal_diff), (1, 2, 3))
            else:
                temporal_energy = abs_mean * Tensor(0.0, ms.float32)
            return self.stack((abs_mean, std, temporal_energy))

        def construct(self, x1, x2, labels):
            _feat1, z1, logits1, temporal_1, recon_1 = self.model(x1)
            _feat2, z2, logits2, temporal_2, recon_2 = self.model(x2)
            all_z = self.concat((z1, z2))
            all_labels = self.concat((labels, labels))
            contrastive_loss = self.supcon_loss_fn(all_z, all_labels)

            if self.objective == "supcon":
                return contrastive_loss

            if self.objective == "supcon_ce":
                if not self.use_ce or self.ce_loss_fn is None:
                    return contrastive_loss
                ce_1 = self.ce_loss_fn(logits1, self.cast(labels, ms.int32))
                ce_2 = self.ce_loss_fn(logits2, self.cast(labels, ms.int32))
                ce = (ce_1 + ce_2) * Tensor(0.5, ms.float32)
                return self.add(contrastive_loss, self.mul(self.ce_weight, ce))

            temporal_loss = self._mse(self._normalize(temporal_1), self._normalize(temporal_2))
            recon_target_1 = self._window_stats_target(x1)
            recon_target_2 = self._window_stats_target(x2)
            recon_loss = (self._mse(recon_1, recon_target_1) + self._mse(recon_2, recon_target_2)) * Tensor(
                0.5, ms.float32
            )
            combined = self.mul(self.contrastive_weight, contrastive_loss)
            combined = self.add(combined, self.mul(self.temporal_weight, temporal_loss))
            combined = self.add(combined, self.mul(self.recon_weight, recon_loss))
            if self.use_ce and self.ce_loss_fn is not None:
                ce_1 = self.ce_loss_fn(logits1, self.cast(labels, ms.int32))
                ce_2 = self.ce_loss_fn(logits2, self.cast(labels, ms.int32))
                ce = (ce_1 + ce_2) * Tensor(0.5, ms.float32)
                combined = self.add(combined, self.mul(self.ce_weight, ce))
            return combined


def _build_lr_schedule(total_steps: int, warmup_steps: int, base_lr: float) -> list[float]:
    if total_steps <= 0:
        return []
    base = float(base_lr)
    schedule: list[float] = []
    for step in range(int(total_steps)):
        if warmup_steps > 0 and step < warmup_steps:
            lr = base * float(step + 1) / float(warmup_steps)
        else:
            progress = 0.0
            if total_steps > warmup_steps:
                progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
            lr = base * 0.5 * (1.0 + np.cos(np.pi * progress))
        schedule.append(float(lr))
    return schedule


def _encode_embeddings(encoder, samples: np.ndarray, *, batch_size: int) -> np.ndarray:
    if ms is None or Tensor is None:
        raise RuntimeError("MindSpore is not available")
    encoder.set_train(False)
    outputs: list[np.ndarray] = []
    total = int(samples.shape[0])
    for start in range(0, total, int(batch_size)):
        batch = samples[start : start + int(batch_size)]
        feat = encoder(Tensor(batch, ms.float32)).asnumpy().astype(np.float32)
        outputs.append(feat)
    emb = np.concatenate(outputs, axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (emb / norms).astype(np.float32)


def _knn_predict(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    k_eff = max(1, min(int(k), int(train_embeddings.shape[0])))
    sim = np.matmul(query_embeddings, train_embeddings.T).astype(np.float32)
    top_idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    preds: list[int] = []
    for row_idx, neighbors in enumerate(top_idx):
        labels = train_labels[neighbors].astype(np.int32)
        scores = sim[row_idx, neighbors]
        counts: dict[int, int] = {}
        score_sums: dict[int, float] = {}
        for label, score in zip(labels.tolist(), scores.tolist()):
            counts[label] = counts.get(label, 0) + 1
            score_sums[label] = score_sums.get(label, 0.0) + float(score)
        best_label = max(
            counts.keys(),
            key=lambda label: (
                counts[label],
                score_sums[label] / max(1, counts[label]),
                -label,
            ),
        )
        preds.append(int(best_label))
    return np.asarray(preds, dtype=np.int32)


def _evaluate_repr_knn(
    encoder,
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    class_names: list[str],
    batch_size: int,
    knn_k: int,
) -> dict:
    train_emb = _encode_embeddings(encoder, train_x, batch_size=batch_size)
    eval_emb = _encode_embeddings(encoder, eval_x, batch_size=batch_size)
    pred = _knn_predict(train_emb, train_y.astype(np.int32), eval_emb, k=int(knn_k))
    return compute_classification_report(eval_y.astype(np.int32), pred, class_names)


def _save_history(history: dict, out_csv: str | Path) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)


def _publish_foundation_encoder(
    *,
    foundation_dir: Path,
    run_id: str,
    run_dir: Path,
    summary: dict,
    class_names: list[str],
    data_dir: str,
) -> dict:
    foundation_dir.mkdir(parents=True, exist_ok=True)
    encoder_ckpt = foundation_dir / "checkpoints" / "db5_full53_repr_encoder.ckpt"
    encoder_ckpt.parent.mkdir(parents=True, exist_ok=True)
    source_ckpt = Path(summary["checkpoint_path"])
    if encoder_ckpt.exists():
        encoder_ckpt.chmod(0o644)
        encoder_ckpt.unlink()
    shutil.copyfile(source_ckpt, encoder_ckpt)
    encoder_ckpt.chmod(0o644)
    payload = {
        "foundation_version": str(summary.get("foundation_version", "db5_full53_repr_v1")),
        "repr_objective": str(summary.get("repr_objective", "supcon")),
        "num_classes": int(summary.get("num_classes", len(class_names))),
        "class_names": list(class_names),
        "checkpoint_path": str(encoder_ckpt),
        "source_run_id": str(run_id),
        "source_run_dir": str(run_dir),
        "source_data_dir": str(data_dir),
        "source_summary_path": str(run_dir / "offline_summary.json"),
        "created_at_unix": int(time.time()),
    }
    dump_json(foundation_dir / "foundation_repr_manifest.json", payload)
    return payload


def _build_referee_card(summary: dict, *, command: str, data_dir: str) -> str:
    return "\n".join(
        [
            "# DB5 Representation Foundation Repro Card",
            "",
            "## Command",
            "```bash",
            str(command).strip(),
            "```",
            "",
            "## Best Run",
            f"- checkpoint: `{summary.get('checkpoint_path', '')}`",
            f"- best_val_epoch: `{summary.get('best_val_epoch', '')}`",
            f"- best_val_macro_f1: `{summary.get('best_val_macro_f1', '')}`",
            f"- best_val_acc: `{summary.get('best_val_acc', '')}`",
            f"- test_macro_f1: `{summary.get('test_macro_f1', '')}`",
            f"- test_accuracy: `{summary.get('test_accuracy', '')}`",
            "",
            "## Data and Constraint",
            f"- data_dir: `{data_dir}`",
            "- no personal calibration data required",
            "",
        ]
    )


def _run_repr_once(args: argparse.Namespace, *, ms_mode: str) -> dict:
    if ms is None or nn is None or ops is None or Tensor is None:
        raise RuntimeError("MindSpore is required for representation pretraining.")

    config = load_db5_pretrain_config(args.config)
    if args.data_dir:
        config.data_dir = str(args.data_dir)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        config.training.learning_rate = float(args.learning_rate)
    if args.weight_decay is not None:
        config.training.weight_decay = float(args.weight_decay)
    if args.epochs is not None:
        config.training.epochs = int(args.epochs)
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = int(args.early_stopping_patience)
    if str(args.quality_sampling_mode or "").strip():
        config.feature.quality_sampling_mode = str(args.quality_sampling_mode).strip().lower()
    manifest_use_source_metadata_override = _parse_bool_arg(args.manifest_use_source_metadata)
    if manifest_use_source_metadata_override is not None:
        config.manifest_use_source_metadata = bool(manifest_use_source_metadata_override)

    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="db5_repr_pretrain")
    logger = logging.getLogger("ninapro_db5.repr")
    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", run_dir)
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)

    _set_device(mode=ms_mode, target=args.device_target, device_id=args.device_id)
    logger.info("Training device configured: mode=%s target=%s device_id=%d", ms_mode, args.device_target, args.device_id)

    data_dir = str(config.data_dir)
    loader = DB5PretrainDatasetLoader(data_dir, config)
    samples, labels, source_ids, source_metadata = loader.load_all_with_sources(return_metadata=True)
    class_names = loader.get_class_names()
    if not class_names:
        raise RuntimeError("DB5 class_names is empty.")
    dump_json(run_dir / "db5_window_diagnostics.json", loader.get_window_diagnostics())
    manifest = build_manifest(
        labels,
        source_ids,
        seed=config.split_seed,
        split_mode="grouped_file",
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_classes=len(class_names),
        class_names=class_names,
        manifest_strategy="v2",
        source_metadata=source_metadata if config.manifest_use_source_metadata else None,
    )
    dump_json(run_dir / "db5_manifest.json", manifest.to_dict())
    split_diagnostics = _build_split_diagnostics(manifest, class_names)
    dump_json(run_dir / "db5_split_diagnostics.json", split_diagnostics)
    if _has_group_leakage(manifest):
        raise RuntimeError("Group leakage detected in representation manifest.")

    train_idx = np.asarray(manifest.train_indices, dtype=np.int32)
    val_idx = np.asarray(manifest.val_indices, dtype=np.int32)
    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    train_x, train_y = samples[train_idx], labels[train_idx]
    val_x, val_y = samples[val_idx], labels[val_idx]
    test_x, test_y = samples[test_idx], labels[test_idx]
    train_source_ids = np.asarray(source_ids)[train_idx]
    train_metadata = [dict(source_metadata[int(idx)]) for idx in train_idx.tolist()]
    logger.info("Split sizes => train=%d val=%d test=%d", len(train_y), len(val_y), len(test_y))

    repr_objective = _resolve_repr_objective(args.repr_objective)
    sampler_mode = _resolve_sampler_mode(args.sampler_mode)
    label_smoothing = float(
        args.label_smoothing if args.label_smoothing is not None else float(config.training.label_smoothing)
    )
    ce_weight = float(args.ce_weight)
    contrastive_weight = float(args.contrastive_weight)
    temporal_weight = float(args.temporal_weight)
    recon_weight = float(args.recon_weight)
    pairing_mode = str(args.pairing_mode).strip().lower()
    if float(args.pair_cross_source_ratio) < 0.0 or float(args.pair_cross_source_ratio) > 1.0:
        raise ValueError("pair_cross_source_ratio must be in [0, 1].")
    if float(args.pair_top_quality_ratio) <= 0.0 or float(args.pair_top_quality_ratio) > 1.0:
        raise ValueError("pair_top_quality_ratio must be in (0, 1].")
    base_aug_params = _resolve_augmentation_params(
        profile=str(args.augmentation_profile),
        scale_min=args.augment_scale_min,
        scale_max=args.augment_scale_max,
        noise_std=args.augment_noise_std,
        channel_drop_ratio=args.augment_channel_drop_ratio,
        time_mask_ratio=args.augment_time_mask_ratio,
        freq_mask_ratio=args.augment_freq_mask_ratio,
    )
    logger.info(
        (
            "repr setup: objective=%s sampler_mode=%s aug_profile=%s pairing_mode=%s "
            "ce_weight=%.3f label_smoothing=%.3f loss_w={contrastive=%.2f temporal=%.2f recon=%.2f} "
            "quality_sampling_mode=%s"
        ),
        repr_objective,
        sampler_mode,
        base_aug_params.get("profile", ""),
        pairing_mode,
        ce_weight,
        label_smoothing,
        contrastive_weight,
        temporal_weight,
        recon_weight,
        str(config.feature.quality_sampling_mode),
    )

    encoder = build_db5_encoder_model(config)
    embedding_dim = int(config.base_channels * 6)
    repr_model = RepresentationNet(
        encoder,
        projection_dim=int(args.projection_dim),
        embedding_dim=embedding_dim,
        num_classes=int(len(class_names)),
        temporal_dim=int(args.temporal_dim),
    )
    supcon_loss_fn = SupervisedContrastiveLoss(float(args.temperature))
    ce_loss_fn = None
    if repr_objective == "supcon_ce" or (repr_objective == "multitask_repr" and ce_weight > 0.0):
        ce_loss_fn = LabelSmoothedCrossEntropy(int(len(class_names)), smoothing=label_smoothing)
    train_cell = ReprTrainCell(
        repr_model,
        supcon_loss_fn,
        objective=repr_objective,
        use_ce=(repr_objective == "supcon_ce" or (repr_objective == "multitask_repr" and ce_weight > 0.0)),
        ce_loss_fn=ce_loss_fn,
        ce_weight=ce_weight,
        contrastive_weight=contrastive_weight,
        temporal_weight=temporal_weight,
        recon_weight=recon_weight,
    )

    steps_per_epoch = int(np.ceil(max(1, len(train_y)) / max(1, int(config.training.batch_size))))
    total_steps = int(steps_per_epoch * int(config.training.epochs))
    warmup_steps = int(steps_per_epoch * int(config.training.warmup_epochs))
    lr_schedule = _build_lr_schedule(total_steps, warmup_steps, float(config.training.learning_rate))
    if not lr_schedule:
        raise RuntimeError("Empty learning rate schedule for representation pretraining.")

    optimizer = nn.AdamWeightDecay(
        repr_model.trainable_params(),
        learning_rate=lr_schedule,
        weight_decay=float(config.training.weight_decay),
    )
    train_step = nn.TrainOneStepCell(train_cell, optimizer)
    train_step.set_train(True)
    ema = ModelEMA(config.training.ema.decay) if bool(config.training.ema.enabled) else None
    rng = np.random.default_rng(int(config.training.split_seed))

    history = {
        "epoch": [],
        "train_loss": [],
        "val_macro_f1": [],
        "val_acc": [],
        "lr": [],
    }

    best_epoch = -1
    best_val_f1 = -1.0
    best_val_acc = -1.0
    bad_epochs = 0
    global_step = 0
    encoder_ckpt_path = run_dir / "checkpoints" / "db5_repr_encoder_best.ckpt"
    model_ckpt_path = run_dir / "checkpoints" / "db5_repr_model_best.ckpt"
    encoder_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config.training.epochs) + 1):
        epoch_losses: list[float] = []
        epoch_aug_params = _resolve_epoch_augmentation_params(
            params=base_aug_params,
            epoch=epoch,
            total_epochs=int(config.training.epochs),
        )
        batch_indices = _build_training_indices(
            labels=train_y.astype(np.int32),
            source_ids=train_source_ids,
            batch_size=int(config.training.batch_size),
            steps_per_epoch=steps_per_epoch,
            seed=int(config.training.split_seed) + epoch,
            sampler_mode=sampler_mode,
            sampler_cfg=config.training.sampler,
            class_names=class_names,
        )
        if batch_indices.size == 0:
            raise RuntimeError("Sampler returned no indices for representation training.")

        for step in range(steps_per_epoch):
            start = step * int(config.training.batch_size)
            end = start + int(config.training.batch_size)
            idx = batch_indices[start:end]
            if idx.size == 0:
                continue
            batch_x = train_x[idx]
            pair_idx = idx
            if pairing_mode == "quality_mixed":
                pair_idx = _build_quality_aware_positive_pairs(
                    anchor_indices=idx,
                    labels=train_y.astype(np.int32),
                    metadata_rows=train_metadata,
                    seed=int(config.training.split_seed) + epoch * 997 + step,
                    cross_source_ratio=float(args.pair_cross_source_ratio),
                    top_quality_ratio=float(args.pair_top_quality_ratio),
                )
            batch_x_pair = train_x[pair_idx]
            batch_y = train_y[idx].astype(np.int32)
            view1 = _augment_batch(batch_x, rng, params=epoch_aug_params)
            view2 = _augment_batch(batch_x_pair, rng, params=epoch_aug_params)
            loss = train_step(
                Tensor(view1, ms.float32),
                Tensor(view2, ms.float32),
                Tensor(batch_y, ms.int32),
            )
            epoch_losses.append(float(loss.asnumpy()))
            if ema is not None:
                ema.update(repr_model)
            global_step += 1

        if ema is not None:
            backup = ema.apply(repr_model)
        else:
            backup = {}
        val_report = _evaluate_repr_knn(
            repr_model.encoder,
            train_x=train_x,
            train_y=train_y,
            eval_x=val_x,
            eval_y=val_y,
            class_names=class_names,
            batch_size=int(config.training.batch_size),
            knn_k=int(args.knn_k),
        )
        if ema is not None:
            ema.restore(repr_model, backup)

        val_f1 = float(val_report["macro_f1"])
        val_acc = float(val_report["accuracy"])
        current_lr = float(lr_schedule[min(max(global_step - 1, 0), len(lr_schedule) - 1)])
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        history["val_macro_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %d/%d | ReprLoss %.4f | Val Acc %.4f F1 %.4f | LR %.6f | AugAlpha %.2f",
            epoch,
            int(config.training.epochs),
            history["train_loss"][-1],
            val_acc,
            val_f1,
            current_lr,
            float(epoch_aug_params.get("curriculum_alpha", 1.0)),
        )

        improved = (val_f1 > best_val_f1 + 1e-6) or (
            abs(val_f1 - best_val_f1) <= 1e-6 and val_acc > best_val_acc + 1e-6
        )
        if improved:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            if ema is not None:
                backup = ema.apply(repr_model)
                save_checkpoint(repr_model.encoder, str(encoder_ckpt_path))
                save_checkpoint(repr_model, str(model_ckpt_path))
                ema.restore(repr_model, backup)
            else:
                save_checkpoint(repr_model.encoder, str(encoder_ckpt_path))
                save_checkpoint(repr_model, str(model_ckpt_path))
            logger.info("New best representation checkpoint: epoch=%d val_f1=%.4f", best_epoch, best_val_f1)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config.training.early_stopping_patience):
                logger.info(
                    "Early stopping: patience=%d best_epoch=%d best_val_f1=%.4f",
                    int(config.training.early_stopping_patience),
                    best_epoch,
                    best_val_f1,
                )
                break

    if not encoder_ckpt_path.exists():
        raise RuntimeError("Representation training finished without a best checkpoint.")

    _set_device(mode=ms_mode, target=args.device_target, device_id=args.device_id)
    best_encoder = build_db5_encoder_model(config)
    from mindspore import load_checkpoint, load_param_into_net

    enc_params = load_checkpoint(str(encoder_ckpt_path))
    load_param_into_net(best_encoder, enc_params)
    best_encoder.set_train(False)

    val_report = _evaluate_repr_knn(
        best_encoder,
        train_x=train_x,
        train_y=train_y,
        eval_x=val_x,
        eval_y=val_y,
        class_names=class_names,
        batch_size=int(config.training.batch_size),
        knn_k=int(args.knn_k),
    )
    test_report = _evaluate_repr_knn(
        best_encoder,
        train_x=train_x,
        train_y=train_y,
        eval_x=test_x,
        eval_y=test_y,
        class_names=class_names,
        batch_size=int(config.training.batch_size),
        knn_k=int(args.knn_k),
    )

    _save_history(history, run_dir / "training_history.csv")
    repr_eval_dir = run_dir / "repr_eval"
    repr_eval_dir.mkdir(parents=True, exist_ok=True)
    val_outputs = save_classification_report(val_report, out_dir=repr_eval_dir, prefix="repr_knn_val")
    test_outputs = save_classification_report(test_report, out_dir=repr_eval_dir, prefix="repr_knn_test")
    dump_json(
        repr_eval_dir / "repr_eval_summary.json",
        {
            "knn_k": int(args.knn_k),
            "repr_objective": repr_objective,
            "sampler_mode": sampler_mode,
            "best_val_epoch": int(best_epoch),
            "val_macro_f1": float(val_report["macro_f1"]),
            "val_acc": float(val_report["accuracy"]),
            "test_macro_f1": float(test_report["macro_f1"]),
            "test_accuracy": float(test_report["accuracy"]),
            "val_outputs": val_outputs,
            "test_outputs": test_outputs,
        },
    )

    sampler_cfg = config.training.sampler
    sampler_snapshot = {
        "type": str(getattr(sampler_cfg, "type", "")),
        "hard_mining_ratio": float(getattr(sampler_cfg, "hard_mining_ratio", 0.0) or 0.0),
        "confusion_pairs": list(getattr(sampler_cfg, "confusion_pairs", []) or []),
    }

    summary = {
        "run_id": run_id,
        "checkpoint_path": str(encoder_ckpt_path),
        "foundation_version": str(args.foundation_version or "db5_full53_repr_v1"),
        "repr_objective": repr_objective,
        "num_classes": int(len(class_names)),
        "best_val_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "test_macro_f1": float(test_report["macro_f1"]),
        "test_accuracy": float(test_report["accuracy"]),
        "temperature": float(args.temperature),
        "projection_dim": int(args.projection_dim),
        "knn_k": int(args.knn_k),
        "sampler_mode": sampler_mode,
        "sampler_snapshot": sampler_snapshot,
        "augmentation_profile": str(base_aug_params["profile"]),
        "augment_scale_min": float(base_aug_params["scale_min"]),
        "augment_scale_max": float(base_aug_params["scale_max"]),
        "augment_noise_std": float(base_aug_params["noise_std"]),
        "augment_channel_drop_ratio": float(base_aug_params["channel_drop_ratio"]),
        "augment_time_mask_ratio": float(base_aug_params["time_mask_ratio"]),
        "augment_freq_mask_ratio": float(base_aug_params["freq_mask_ratio"]),
        "ce_weight": float(ce_weight if (repr_objective == "supcon_ce" or ce_weight > 0.0) else 0.0),
        "label_smoothing": float(label_smoothing if (repr_objective == "supcon_ce" or ce_weight > 0.0) else 0.0),
        "contrastive_weight": float(contrastive_weight if repr_objective == "multitask_repr" else 1.0),
        "temporal_weight": float(temporal_weight if repr_objective == "multitask_repr" else 0.0),
        "recon_weight": float(recon_weight if repr_objective == "multitask_repr" else 0.0),
        "pairing_mode": pairing_mode,
        "pair_cross_source_ratio": float(args.pair_cross_source_ratio),
        "pair_top_quality_ratio": float(args.pair_top_quality_ratio),
        "quality_sampling_mode": str(config.feature.quality_sampling_mode),
        "final_curriculum_alpha": float(
            _resolve_epoch_augmentation_params(
                params=base_aug_params,
                epoch=int(config.training.epochs),
                total_epochs=int(config.training.epochs),
            ).get("curriculum_alpha", 1.0)
        ),
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "manifest_use_source_metadata": bool(config.manifest_use_source_metadata),
        "split_seed": int(config.split_seed),
        "run_mode": str(ms_mode),
        "class_names": class_names,
    }
    dump_json(run_dir / "offline_summary.json", summary)
    append_csv_row(Path(args.run_root) / "db5_repr_pretrain_results.csv", SUMMARY_FIELDS, summary)

    foundation_manifest = _publish_foundation_encoder(
        foundation_dir=Path(args.foundation_dir),
        run_id=run_id,
        run_dir=run_dir,
        summary=summary,
        class_names=class_names,
        data_dir=data_dir,
    )
    dump_json(run_dir / "foundation_repr_manifest.json", foundation_manifest)
    referee_card_path = run_dir / "referee_repro_card.md"
    referee_card_path.write_text(
        _build_referee_card(
            summary,
            command="python " + " ".join(shlex.quote(item) for item in sys.argv),
            data_dir=data_dir,
        ),
        encoding="utf-8",
    )
    dump_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "data_dir": data_dir,
            "mode": str(ms_mode),
            "repr_objective": repr_objective,
            "sampler_mode": sampler_mode,
            "pairing_mode": pairing_mode,
            "quality_sampling_mode": str(config.feature.quality_sampling_mode),
            "encoder_checkpoint_path": str(encoder_ckpt_path),
            "model_checkpoint_path": str(model_ckpt_path),
            "repr_eval_dir": str(repr_eval_dir),
            "referee_repro_card": str(referee_card_path),
        },
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DB5 representation pretraining with supervised contrastive loss.")
    parser.add_argument("--config", default="experimental/configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--ms_mode", default="graph", choices=["graph", "pynative"])
    parser.add_argument("--auto_fallback_pynative", choices=["true", "false"], default="true")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--foundation_version", default="db5_full53_repr_v1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--manifest_use_source_metadata", choices=["true", "false"], default=None)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--temporal_dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--repr_objective", default="supcon_ce", choices=["supcon", "supcon_ce", "multitask_repr"])
    parser.add_argument("--ce_weight", type=float, default=0.35)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--temporal_weight", type=float, default=0.3)
    parser.add_argument("--recon_weight", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument(
        "--sampler_mode",
        default="class_source_balanced",
        choices=["class_balanced", "class_source_balanced", "balanced", "source_balanced"],
    )
    parser.add_argument("--augmentation_profile", default="strong", choices=["mild", "strong", "curriculum"])
    parser.add_argument("--pairing_mode", default="same_window_aug", choices=["same_window_aug", "quality_mixed"])
    parser.add_argument("--pair_cross_source_ratio", type=float, default=0.6)
    parser.add_argument("--pair_top_quality_ratio", type=float, default=0.4)
    parser.add_argument("--quality_sampling_mode", default=None, choices=["quality", "uniform"])
    parser.add_argument("--augment_scale_min", type=float, default=None)
    parser.add_argument("--augment_scale_max", type=float, default=None)
    parser.add_argument("--augment_noise_std", type=float, default=None)
    parser.add_argument("--augment_channel_drop_ratio", type=float, default=None)
    parser.add_argument("--augment_time_mask_ratio", type=float, default=None)
    parser.add_argument("--augment_freq_mask_ratio", type=float, default=None)
    parser.add_argument("--run_downstream_fewshot", choices=["true", "false"], default="false")
    parser.add_argument("--fewshot_script", default="scripts/evaluate_event_fewshot_curve.py")
    parser.add_argument("--fewshot_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--fewshot_data_dir", default="../data")
    parser.add_argument("--fewshot_recordings_manifest", default=None)
    parser.add_argument("--fewshot_target_db5_keys", default="E1_G01,E1_G02,E1_G03,E1_G04")
    parser.add_argument("--fewshot_budgets", default="10,20,35,60")
    parser.add_argument("--fewshot_seeds", default="11,22,33")
    return parser


def _resolve_existing_recordings_manifest(
    *,
    data_dir: str,
    config_path: str,
    manifest_arg: str | None,
) -> str:
    if str(manifest_arg or "").strip():
        raw = Path(str(manifest_arg).strip())
    else:
        root = load_config(config_path)
        data_cfg = dict(root.get("data", {}) or {})
        default_rel = str(data_cfg.get("recordings_manifest_path") or "recordings_manifest.csv")
        raw = Path(default_rel)

    candidates = [raw]
    if not raw.is_absolute():
        candidates.insert(0, Path(data_dir) / raw)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

    expected = candidates[0]
    raise FileNotFoundError(
        "run_downstream_fewshot=true requires recordings manifest with event metadata. "
        f"Missing file: {expected}. "
        "Fix: pass --fewshot_recordings_manifest <path/to/recordings_manifest.csv>."
    )


def _run_optional_fewshot(args: argparse.Namespace, *, encoder_checkpoint: str, run_dir: Path) -> None:
    enabled = bool(str(args.run_downstream_fewshot).strip().lower() == "true")
    if not enabled:
        return
    manifest_path = _resolve_existing_recordings_manifest(
        data_dir=str(args.fewshot_data_dir),
        config_path=str(args.fewshot_config),
        manifest_arg=args.fewshot_recordings_manifest,
    )
    cmd = [
        sys.executable,
        str(args.fewshot_script),
        "--config",
        str(args.fewshot_config),
        "--data_dir",
        str(args.fewshot_data_dir),
        "--run_root",
        str(args.run_root),
        "--run_id",
        f"{Path(run_dir).name}_fewshot",
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--pretrained_emg_checkpoint",
        str(encoder_checkpoint),
        "--target_db5_keys",
        str(args.fewshot_target_db5_keys),
        "--budgets",
        str(args.fewshot_budgets),
        "--seeds",
        str(args.fewshot_seeds),
        "--recordings_manifest",
        str(manifest_path),
    ]
    subprocess.run(cmd, cwd=str(CODE_ROOT), check=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    logger = logging.getLogger("ninapro_db5.repr")
    start = time.time()
    repro_command = "python " + " ".join(shlex.quote(item) for item in sys.argv)

    fallback = bool(str(args.auto_fallback_pynative).strip().lower() == "true")
    try:
        try:
            result = _run_repr_once(args, ms_mode=str(args.ms_mode))
            run_dir = Path(result["run_dir"])
        except Exception as exc:
            if str(args.ms_mode).lower() == "graph" and fallback:
                logger.warning("Graph mode failed; retrying in pynative mode. root_cause=%s", exc)
                result = _run_repr_once(args, ms_mode="pynative")
                run_dir = Path(result["run_dir"])
                dump_json(
                    run_dir / "graph_fallback_report.json",
                    {
                        "fallback_triggered": True,
                        "initial_mode": "graph",
                        "retry_mode": "pynative",
                        "root_cause": str(exc),
                    },
                )
            else:
                raise

        summary = dict(result["summary"])
        _run_optional_fewshot(args, encoder_checkpoint=str(summary["checkpoint_path"]), run_dir=run_dir)
        dump_json(
            run_dir / "run_completion.json",
            {
                "run_id": str(summary["run_id"]),
                "run_dir": str(run_dir),
                "elapsed_minutes": float((time.time() - start) / 60.0),
                "checkpoint_path": str(summary["checkpoint_path"]),
            },
        )
    except Exception as exc:
        failure_dir = Path(args.run_root) / f"{(args.run_id or 'db5_repr')}_failure"
        dump_json(
            failure_dir / "db5_repr_failure_report.json",
            {
                "status": "failed",
                "stage": "repr_pretrain",
                "root_cause": str(exc),
                "next_command": repro_command,
                "generated_at_unix": int(time.time()),
            },
        )
        raise


if __name__ == "__main__":
    main()
