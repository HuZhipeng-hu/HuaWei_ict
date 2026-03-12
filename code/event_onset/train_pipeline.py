"""Training pipeline for event-onset experiments."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from event_onset.config import EventModelConfig, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.evaluate import load_and_evaluate_event
from event_onset.model import build_event_model
from event_onset.trainer import EventTrainer
from ninapro_db5.model import load_emg_encoder_from_db5_checkpoint
from shared.label_modes import get_label_mode_spec
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, dump_yaml, ensure_run_dir
from training.reporting import save_classification_report
from training.data.split_strategy import SplitManifest, build_manifest, load_manifest, save_manifest

logger = logging.getLogger("training.event_onset")

OFFLINE_SUMMARY_FIELDS = [
    "run_id",
    "manifest_path",
    "checkpoint_path",
    "model_type",
    "base_channels",
    "use_se",
    "loss_type",
    "hard_mining_ratio",
    "augment_enabled",
    "augment_factor",
    "use_mixup",
    "budget_per_class",
    "budget_seed",
    "used_pretrained_init",
    "test_accuracy",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
]


class _ManifestLoadIssue(ValueError):
    """Raised when a manifest cannot be safely reused for the current event training run."""


_MANIFEST_FIELD_NAMES = {item.name for item in dataclass_fields(SplitManifest)}


def _apply_cli_overrides(args, model_cfg: EventModelConfig, train_cfg, augmentation_cfg):
    if args.base_channels is not None:
        model_cfg.base_channels = int(args.base_channels)
    if args.use_se is not None:
        model_cfg.use_se = bool(args.use_se)
    if args.loss_type is not None:
        train_cfg.loss.type = args.loss_type
    if args.hard_mining_ratio is not None:
        train_cfg.sampler.hard_mining_ratio = float(args.hard_mining_ratio)
    if args.augment_factor is not None:
        augmentation_cfg.augment_factor = int(args.augment_factor)
    if args.use_mixup is not None:
        augmentation_cfg.use_mixup = bool(args.use_mixup)
    if args.augmentation_enabled is not None:
        augmentation_cfg.enabled = bool(args.augmentation_enabled)
    if args.split_seed is not None:
        train_cfg.split_seed = int(args.split_seed)
    if getattr(args, "pretrained_emg_checkpoint", None):
        model_cfg.pretrained_emg_checkpoint = args.pretrained_emg_checkpoint
    return model_cfg, train_cfg, augmentation_cfg


def _apply_train_budget_to_manifest(
    manifest: SplitManifest,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    budget_per_class: int,
    budget_seed: int,
) -> tuple[SplitManifest, dict[str, dict[str, int]]]:
    if budget_per_class <= 0:
        raise ValueError("budget_per_class must be > 0")

    rng = np.random.default_rng(int(budget_seed))
    train_idx = np.asarray(manifest.train_indices, dtype=np.int32)
    kept: list[int] = []
    report: dict[str, dict[str, int]] = {}
    missing_classes: list[str] = []
    for class_id, class_name in enumerate(class_names):
        class_train_idx = train_idx[labels[train_idx] == class_id]
        available = int(class_train_idx.shape[0])
        if available <= 0:
            missing_classes.append(str(class_name))
        selected = min(available, int(budget_per_class))
        if selected > 0:
            if available > selected:
                chosen = rng.choice(class_train_idx, size=selected, replace=False)
            else:
                chosen = class_train_idx
            kept.extend(int(idx) for idx in chosen.tolist())
        report[class_name] = {"available": available, "selected": selected}

    if missing_classes:
        raise RuntimeError(
            "Budgeting failed: train split has zero samples for classes: "
            + ", ".join(missing_classes)
        )

    if not kept:
        raise RuntimeError("Budgeted train subset is empty; check data quality and budget_per_class.")

    # Preserve split metadata and only shrink train indices.
    budget_manifest = SplitManifest(
        train_indices=sorted(kept),
        val_indices=list(manifest.val_indices),
        test_indices=list(manifest.test_indices),
        train_sources=list(manifest.train_sources),
        val_sources=list(manifest.val_sources),
        test_sources=list(manifest.test_sources),
        seed=manifest.seed,
        split_mode=manifest.split_mode,
        manifest_strategy=manifest.manifest_strategy,
        num_samples=manifest.num_samples,
        val_ratio=manifest.val_ratio,
        test_ratio=manifest.test_ratio,
        class_distribution=dict(manifest.class_distribution),
        group_keys_train=list(manifest.group_keys_train),
        group_keys_val=list(manifest.group_keys_val),
        group_keys_test=list(manifest.group_keys_test),
    )
    return budget_manifest, report


def _save_history(history: dict, out_csv: str | Path) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)


def _build_current_manifest(
    *,
    labels: np.ndarray,
    source_ids: np.ndarray,
    source_metadata: Sequence[dict],
    seed: int,
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    class_names: Sequence[str],
    manifest_strategy: str,
) -> SplitManifest:
    return build_manifest(
        labels,
        source_ids,
        seed=seed,
        split_mode=split_mode,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        num_classes=len(class_names),
        class_names=class_names,
        manifest_strategy=manifest_strategy,
        source_metadata=source_metadata,
    )


def _read_manifest_json(in_path: str) -> dict:
    try:
        with open(in_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise _ManifestLoadIssue(f"manifest is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise _ManifestLoadIssue("manifest root must be a JSON object")
    return payload


def _load_manifest_for_training(in_path: str, *, current_num_samples: int) -> SplitManifest:
    payload = _read_manifest_json(in_path)
    unknown_fields = sorted(key for key in payload.keys() if key not in _MANIFEST_FIELD_NAMES)
    if unknown_fields:
        raise _ManifestLoadIssue(f"unsupported legacy fields: {', '.join(unknown_fields)}")
    try:
        manifest = load_manifest(in_path)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _ManifestLoadIssue(f"failed validation: {exc}") from exc
    if manifest.num_samples != current_num_samples:
        raise _ManifestLoadIssue(
            f"sample count mismatch: manifest={manifest.num_samples}, loaded={current_num_samples}"
        )
    return manifest


def _prepare_manifest(
    *,
    labels: np.ndarray,
    source_ids: np.ndarray,
    source_metadata: Sequence[dict],
    seed: int,
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    class_names: Sequence[str],
    manifest_in_cli: Optional[str],
    manifest_in_config: Optional[str],
    manifest_out_cli: Optional[str],
    manifest_strategy: str,
) -> tuple[SplitManifest, Optional[str]]:
    current_num_samples = int(labels.shape[0])
    manifest_out_path = manifest_out_cli or manifest_in_config

    if manifest_in_cli:
        if not Path(manifest_in_cli).exists():
            raise FileNotFoundError(
                "Explicit --split_manifest_in path does not exist: "
                f"{manifest_in_cli}. Fix: generate first with --split_manifest_out <path> "
                "or use an existing --split_manifest_in <path>."
            )
        try:
            manifest = _load_manifest_for_training(manifest_in_cli, current_num_samples=current_num_samples)
        except _ManifestLoadIssue as exc:
            raise ValueError(
                "Explicit --split_manifest_in is not compatible with the current dataset: "
                f"{manifest_in_cli} ({exc}). Fix: regenerate it with --split_manifest_out <path> and retry."
            ) from exc
        logger.info("Using split manifest from CLI: %s", manifest_in_cli)
        return manifest, manifest_in_cli

    if manifest_in_config:
        if Path(manifest_in_config).exists():
            try:
                manifest = _load_manifest_for_training(manifest_in_config, current_num_samples=current_num_samples)
            except _ManifestLoadIssue as exc:
                logger.warning(
                    "Configured split manifest at %s is legacy/stale/invalid (%s). Rebuilding with strategy=%s.",
                    manifest_in_config,
                    exc,
                    manifest_strategy,
                )
                manifest = _build_current_manifest(
                    labels=labels,
                    source_ids=source_ids,
                    source_metadata=source_metadata,
                    seed=seed,
                    split_mode=split_mode,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    class_names=class_names,
                    manifest_strategy=manifest_strategy,
                )
                saved = save_manifest(manifest, manifest_in_config)
                logger.info("Rebuilt split manifest at %s", saved)
                return manifest, str(saved)
            logger.info("Using split manifest from config: %s", manifest_in_config)
            return manifest, manifest_in_config

        logger.info(
            "Configured split manifest does not exist (%s). Building one with strategy=%s.",
            manifest_in_config,
            manifest_strategy,
        )
        manifest = _build_current_manifest(
            labels=labels,
            source_ids=source_ids,
            source_metadata=source_metadata,
            seed=seed,
            split_mode=split_mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            class_names=class_names,
            manifest_strategy=manifest_strategy,
        )
        saved = save_manifest(manifest, manifest_out_path or manifest_in_config)
        logger.info("Auto-generated split manifest at %s", saved)
        return manifest, str(saved)

    manifest = _build_current_manifest(
        labels=labels,
        source_ids=source_ids,
        source_metadata=source_metadata,
        seed=seed,
        split_mode=split_mode,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        class_names=class_names,
        manifest_strategy=manifest_strategy,
    )
    saved_path = None
    if manifest_out_path:
        saved = save_manifest(manifest, manifest_out_path)
        logger.info("Saved split manifest: %s", saved)
        saved_path = str(saved)
    return manifest, saved_path


def _split_arrays_by_manifest(
    emg_samples: np.ndarray,
    imu_samples: np.ndarray,
    labels: np.ndarray,
    manifest: SplitManifest,
):
    train_idx = np.asarray(manifest.train_indices, dtype=np.int32)
    val_idx = np.asarray(manifest.val_indices, dtype=np.int32)
    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    return (
        (emg_samples[train_idx], imu_samples[train_idx], labels[train_idx]),
        (emg_samples[val_idx], imu_samples[val_idx], labels[val_idx]),
        (emg_samples[test_idx], imu_samples[test_idx], labels[test_idx]),
    )


def _top_confusion_pair_text(report: dict) -> str:
    pairs = report.get("top_confusion_pairs") or []
    if not pairs:
        return ""
    pair = pairs[0]
    return f"{pair['pair'][0]}<->{pair['pair'][1]}:{pair['count']}"


def run_event_training(args) -> None:
    start = time.time()
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="event_train")
    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", run_dir)

    model_cfg, data_cfg, train_cfg, augmentation_cfg = load_event_training_config(args.config)
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)
    model_cfg, train_cfg, augmentation_cfg = _apply_cli_overrides(args, model_cfg, train_cfg, augmentation_cfg)
    label_spec = get_label_mode_spec(data_cfg.label_mode)

    dump_yaml(
        run_dir / "config_snapshots" / "effective_overrides.yaml",
        {
            "run_id": run_id,
            "model": {
                "model_type": model_cfg.model_type,
                "num_classes": model_cfg.num_classes,
                "base_channels": model_cfg.base_channels,
                "use_se": model_cfg.use_se,
                "dropout_rate": model_cfg.dropout_rate,
                "pretrained_emg_checkpoint": model_cfg.pretrained_emg_checkpoint,
            },
            "training": {
                "loss_type": train_cfg.loss.type,
                "hard_mining_ratio": train_cfg.sampler.hard_mining_ratio,
                "split_seed": train_cfg.split_seed,
                "freeze_emg_epochs": train_cfg.freeze_emg_epochs,
                "unfreeze_last_blocks": train_cfg.unfreeze_last_blocks,
                "encoder_lr_ratio": train_cfg.encoder_lr_ratio,
                "head_lr_ratio": train_cfg.head_lr_ratio,
            },
            "data": {
                "label_mode": data_cfg.label_mode,
                "capture_mode_filter": data_cfg.capture_mode_filter,
                "device_sampling_rate_hz": data_cfg.device_sampling_rate_hz,
                "imu_sampling_rate_hz": data_cfg.imu_sampling_rate_hz,
                "context_window_ms": data_cfg.feature.context_window_ms,
                "window_step_ms": data_cfg.feature.window_step_ms,
                "top_k_windows_per_clip": data_cfg.top_k_windows_per_clip,
                "idle_top_k_windows_per_clip": data_cfg.idle_top_k_windows_per_clip,
                "use_imu": data_cfg.use_imu,
                "budget_per_class": int(getattr(args, "budget_per_class", 0) or 0),
                "budget_seed": int(getattr(args, "budget_seed", train_cfg.split_seed) or train_cfg.split_seed),
            },
        },
    )

    recordings_manifest_path = args.recordings_manifest or data_cfg.recordings_manifest_path
    loader = EventClipDatasetLoader(args.data_dir, data_cfg, recordings_manifest_path=recordings_manifest_path)
    dataset_stats = loader.get_stats()
    logger.info("Event dataset stats: %s", dataset_stats)
    emg_samples, imu_samples, labels, source_ids, source_meta = loader.load_all_with_sources(return_metadata=True)
    logger.info(
        "Loaded event samples: emg=%s imu=%s labels=%d",
        tuple(emg_samples.shape),
        tuple(imu_samples.shape),
        labels.shape[0],
    )

    quality_report_path = Path(args.quality_report_out) if args.quality_report_out else run_dir / "quality" / "quality_report.json"
    q_path = loader.save_quality_report(quality_report_path)
    logger.info("Saved quality report: %s", q_path)

    manifest, manifest_path = _prepare_manifest(
        labels=labels,
        source_ids=source_ids,
        source_metadata=source_meta,
        seed=train_cfg.split_seed,
        split_mode=data_cfg.split_mode,
        val_ratio=train_cfg.val_ratio,
        test_ratio=train_cfg.test_ratio,
        class_names=label_spec.class_names,
        manifest_in_cli=args.split_manifest_in,
        manifest_in_config=data_cfg.split_manifest_path,
        manifest_out_cli=args.split_manifest_out,
        manifest_strategy=args.manifest_strategy,
    )
    if manifest_path:
        copy_config_snapshot(manifest_path, run_dir / "manifests" / Path(manifest_path).name)

    budget_per_class = int(getattr(args, "budget_per_class", 0) or 0)
    budget_seed = int(getattr(args, "budget_seed", train_cfg.split_seed) or train_cfg.split_seed)
    if budget_per_class > 0:
        manifest, budget_report = _apply_train_budget_to_manifest(
            manifest,
            labels,
            label_spec.class_names,
            budget_per_class=budget_per_class,
            budget_seed=budget_seed,
        )
        dump_json(
            run_dir / "manifests" / "train_budget_report.json",
            {
                "budget_per_class": budget_per_class,
                "budget_seed": budget_seed,
                "class_budget": budget_report,
                "train_samples_after_budget": len(manifest.train_indices),
            },
        )
        logger.info(
            "Applied train budget: per_class=%d seed=%d train_samples=%d",
            budget_per_class,
            budget_seed,
            len(manifest.train_indices),
        )

    (train_emg, train_imu, train_y), (val_emg, val_imu, val_y), (test_emg, test_imu, test_y) = _split_arrays_by_manifest(
        emg_samples,
        imu_samples,
        labels,
        manifest,
    )
    logger.info("Split sizes => train=%d, val=%d, test=%d", len(train_y), len(val_y), len(test_y))

    if augmentation_cfg.enabled and augmentation_cfg.augment_factor > 1:
        logger.warning("Event-onset pipeline does not yet augment EMG+IMU jointly. Ignoring augmentation factor=%s.", augmentation_cfg.augment_factor)

    model = build_event_model(model_cfg)
    if model_cfg.pretrained_emg_checkpoint:
        transferred = load_emg_encoder_from_db5_checkpoint(model, model_cfg.pretrained_emg_checkpoint)
        logger.info(
            "Loaded DB5 EMG encoder weights from %s (loaded=%d skipped=%d)",
            model_cfg.pretrained_emg_checkpoint,
            transferred["loaded"],
            transferred["skipped"],
        )
    trainer = EventTrainer(model, model_cfg, train_cfg, label_spec.class_names, output_dir=str(run_dir))
    history = trainer.train(train_emg, train_imu, train_y, val_emg, val_imu, val_y)
    history_path = run_dir / "training_history.csv"
    _save_history(history, history_path)

    report = load_and_evaluate_event(
        ckpt_path=trainer.checkpoint_path,
        emg_samples=test_emg,
        imu_samples=test_imu,
        labels=test_y,
        class_names=label_spec.class_names,
        model_config=model_cfg,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    report.update(
        {
            "eval_protocol": args.eval_protocol,
            "manifest_path": manifest_path,
            "checkpoint_path": str(trainer.checkpoint_path),
            "run_id": run_id,
        }
    )
    report_paths = save_classification_report(report, out_dir=run_dir / "evaluation", prefix="test")

    summary = {
        "run_id": run_id,
        "manifest_path": manifest_path or "",
        "checkpoint_path": str(trainer.checkpoint_path),
        "model_type": model_cfg.model_type,
        "base_channels": model_cfg.base_channels,
        "use_se": model_cfg.use_se,
        "loss_type": train_cfg.loss.type,
        "hard_mining_ratio": train_cfg.sampler.hard_mining_ratio,
        "augment_enabled": False,
        "augment_factor": 1,
        "use_mixup": False,
        "budget_per_class": int(budget_per_class),
        "budget_seed": int(budget_seed),
        "used_pretrained_init": bool(model_cfg.pretrained_emg_checkpoint),
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro_f1"],
        "test_macro_recall": report["macro_recall"],
        "top_confusion_pair": _top_confusion_pair_text(report),
    }
    dump_json(run_dir / "offline_summary.json", summary)
    append_csv_row(Path(args.run_root) / "offline_results.csv", OFFLINE_SUMMARY_FIELDS, summary)
    dump_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "config_path": args.config,
            "recordings_manifest_path": str(recordings_manifest_path),
            "quality_report": str(q_path),
            "training_history": str(history_path),
            "evaluation_outputs": report_paths,
            "elapsed_minutes": (time.time() - start) / 60.0,
        },
    )
