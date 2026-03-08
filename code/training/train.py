"""Training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import mindspore as ms
    from mindspore import context
except Exception:  # pragma: no cover
    ms = None  # type: ignore
    context = None  # type: ignore

from shared.config import load_training_config, load_training_data_config
from shared.gestures import GESTURE_DEFINITIONS
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, dump_yaml, ensure_run_dir
from training.data import CSVDatasetLoader, DataAugmentor, split_and_optionally_augment
from training.data.split_strategy import SplitManifest, build_manifest, load_manifest, save_manifest
from training.evaluate import load_and_evaluate
from training.model import build_model_from_config
from training.reporting import save_classification_report
from training.trainer import Trainer

logger = logging.getLogger("training")

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
    "test_accuracy",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
]


class _ManifestLoadIssue(ValueError):
    """Raised when a manifest cannot be safely reused for the current training run."""


_MANIFEST_FIELD_NAMES = {item.name for item in dataclass_fields(SplitManifest)}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    def _parse_optional_bool(value: str) -> bool:
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = argparse.ArgumentParser(description="Train NeuroGrip model")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--data_dir", required=True, help="Dataset root folder")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--run_id", default=None, help="Stable experiment run id")
    parser.add_argument("--run_root", default="artifacts/runs", help="Base directory for run artifacts")
    parser.add_argument("--recordings_manifest", default=None, help="Optional recordings_manifest.csv override")
    parser.add_argument("--model_type", default=None, choices=["standard", "lite"], help="Override model.model_type")
    parser.add_argument("--base_channels", type=int, default=None, help="Override model.base_channels")
    parser.add_argument("--use_se", type=_parse_optional_bool, default=None, help="Override model.use_se")
    parser.add_argument("--loss_type", default=None, help="Override training.loss.type")
    parser.add_argument("--hard_mining_ratio", type=float, default=None, help="Override training.sampler.hard_mining_ratio")
    parser.add_argument("--augment_factor", type=int, default=None, help="Override augmentation.augment_factor")
    parser.add_argument("--use_mixup", type=_parse_optional_bool, default=None, help="Override augmentation.use_mixup")
    parser.add_argument(
        "--augmentation_enabled",
        type=_parse_optional_bool,
        default=None,
        help="Override augmentation.enabled",
    )
    parser.add_argument("--split_seed", type=int, default=None, help="Override training.split_seed")

    parser.add_argument("--split_manifest_in", default=None, help="Load split manifest from path")
    parser.add_argument("--split_manifest_out", default=None, help="Save built split manifest to path")
    parser.add_argument(
        "--manifest_strategy",
        default="v2",
        choices=["v1", "v2"],
        help="Manifest build strategy when split is generated",
    )
    parser.add_argument("--quality_report_out", default=None, help="Path to save quality filter report JSON")
    parser.add_argument(
        "--eval_protocol",
        default="same_user_same_day_v1",
        help="Evaluation protocol tag recorded in output reports",
    )
    return parser.parse_args()


def _set_device(device_target: str, device_id: int) -> None:
    if ms is None:
        raise RuntimeError("MindSpore is not available")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target=device_target)
    if device_target == "GPU":
        context.set_context(device_id=device_id)


def _gesture_mappings() -> tuple[dict[str, int], list[str]]:
    class_names = [g.name for g in GESTURE_DEFINITIONS]
    gesture_to_idx = {name: i for i, name in enumerate(class_names)}
    return gesture_to_idx, class_names


def _normalize_model_config(model_cfg, preprocess_cfg):
    if preprocess_cfg.dual_branch.enabled and model_cfg.in_channels != 12:
        logger.warning(
            "dual_branch.enabled=true requires in_channels=12. Overriding model.in_channels from %d to 12.",
            model_cfg.in_channels,
        )
        model_cfg.in_channels = 12
    return model_cfg


def _apply_cli_overrides(args: argparse.Namespace, model_cfg, train_cfg, augmentation_cfg):
    if args.model_type is not None:
        model_cfg.model_type = args.model_type
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
    return model_cfg, train_cfg, augmentation_cfg


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
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
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
) -> tuple:
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


def _save_history(history: dict, out_csv: str | Path) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = zip(*(history[k] for k in keys))
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def _build_augmentor(augmentation_cfg, seed: int) -> Optional[DataAugmentor]:
    if not augmentation_cfg.enabled or augmentation_cfg.augment_factor <= 1 and not augmentation_cfg.use_mixup:
        return None
    scale_min = float(getattr(augmentation_cfg, "scale_min", 0.9))
    scale_max = float(getattr(augmentation_cfg, "scale_max", 1.1))
    return DataAugmentor(
        temporal_shift_max=getattr(augmentation_cfg, "temporal_shift_max", 2),
        scale_min=scale_min,
        scale_max=scale_max,
        noise_std=getattr(augmentation_cfg, "noise_std", 0.02),
        mixup_alpha=getattr(augmentation_cfg, "mixup_alpha", 0.2),
        seed=seed,
    )


def _top_confusion_pair_text(report: dict) -> str:
    pairs = report.get("top_confusion_pairs") or []
    if not pairs:
        return ""
    pair = pairs[0]
    return f"{pair['pair'][0]}<->{pair['pair'][1]}:{pair['count']}"


def main() -> None:
    args = parse_args()
    _setup_logging()
    start = time.time()

    logger.info("================================================================")
    logger.info("NeuroGrip model training")
    logger.info("================================================================")
    logger.info("Loading config: %s", args.config)

    if ms is None:
        raise RuntimeError("MindSpore is not installed")

    _set_device(args.device_target, args.device_id)
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="train")
    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", run_dir)

    model_cfg, preprocess_cfg, train_cfg, augmentation_cfg = load_training_config(args.config)
    data_cfg = load_training_data_config(args.config)
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)
    model_cfg, train_cfg, augmentation_cfg = _apply_cli_overrides(args, model_cfg, train_cfg, augmentation_cfg)
    model_cfg = _normalize_model_config(model_cfg, preprocess_cfg)
    dump_yaml(
        run_dir / "config_snapshots" / "effective_overrides.yaml",
        {
            "run_id": run_id,
            "model": {
                "model_type": model_cfg.model_type,
                "in_channels": model_cfg.in_channels,
                "num_classes": model_cfg.num_classes,
                "base_channels": model_cfg.base_channels,
                "use_se": model_cfg.use_se,
                "dropout_rate": model_cfg.dropout_rate,
            },
            "training": {
                "loss_type": train_cfg.loss.type,
                "hard_mining_ratio": train_cfg.sampler.hard_mining_ratio,
                "split_seed": train_cfg.split_seed,
            },
            "augmentation": {
                "enabled": augmentation_cfg.enabled,
                "augment_factor": augmentation_cfg.augment_factor,
                "use_mixup": augmentation_cfg.use_mixup,
            },
        },
    )

    gesture_to_idx, class_names = _gesture_mappings()
    if len(class_names) != model_cfg.num_classes:
        raise ValueError(f"num_classes mismatch: model={model_cfg.num_classes}, gestures={len(class_names)}")
    logger.info("Gesture definition check passed: %d classes", len(class_names))

    recordings_manifest_path = args.recordings_manifest or data_cfg.recordings_manifest_path

    logger.info("Loading dataset from: %s", args.data_dir)
    loader = CSVDatasetLoader(
        args.data_dir,
        gesture_to_idx,
        preprocess_cfg,
        quality_filter=train_cfg.quality_filter,
        recordings_manifest_path=recordings_manifest_path,
    )
    dataset_stats = loader.get_stats()
    logger.info("Dataset stats: %s", dataset_stats)
    samples, labels, source_ids, source_meta = loader.load_all_with_sources(return_metadata=True)
    logger.info("Loaded samples: %d, shape=%s", samples.shape[0], tuple(samples.shape))

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
        class_names=class_names,
        manifest_in_cli=args.split_manifest_in,
        manifest_in_config=data_cfg.split_manifest_path,
        manifest_out_cli=args.split_manifest_out,
        manifest_strategy=args.manifest_strategy,
    )

    if manifest.num_samples != int(samples.shape[0]):
        raise ValueError(
            "Manifest sample count mismatch: "
            f"manifest={manifest.num_samples}, loaded={samples.shape[0]}. "
            "This usually means preprocess/split settings changed. "
            "Please regenerate manifest with --split_manifest_out and retrain."
        )

    if manifest_path:
        copy_config_snapshot(manifest_path, run_dir / "manifests" / Path(manifest_path).name)

    augmentor = _build_augmentor(augmentation_cfg, seed=train_cfg.split_seed)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_and_optionally_augment(
        samples=samples,
        labels=labels,
        manifest=manifest,
        augmentor=augmentor,
        augment_factor=augmentation_cfg.augment_factor,
        use_mixup=augmentation_cfg.use_mixup,
    )
    logger.info(
        "Split sizes => train=%d, val=%d, test=%d",
        len(train_y),
        len(val_y),
        len(test_y),
    )

    model = build_model_from_config(model_cfg)
    trainer = Trainer(model, train_cfg, class_names, output_dir=str(run_dir))
    history = trainer.train(train_x, train_y, val_x, val_y)
    history_path = run_dir / "training_history.csv"
    _save_history(history, history_path)
    logger.info("Saved training history: %s", history_path)

    logger.info("")
    logger.info("================================================================")
    logger.info("Final evaluation on test split")
    logger.info("================================================================")
    report = load_and_evaluate(
        ckpt_path=trainer.checkpoint_path,
        samples=test_x,
        labels=test_y,
        class_names=class_names,
        model_config=model_cfg,
        dropout_rate=model_cfg.dropout_rate,
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
    logger.info("Saved evaluation report: %s", report_paths)

    summary = {
        "run_id": run_id,
        "manifest_path": manifest_path or "",
        "checkpoint_path": str(trainer.checkpoint_path),
        "model_type": model_cfg.model_type,
        "base_channels": model_cfg.base_channels,
        "use_se": model_cfg.use_se,
        "loss_type": train_cfg.loss.type,
        "hard_mining_ratio": train_cfg.sampler.hard_mining_ratio,
        "augment_enabled": augmentation_cfg.enabled,
        "augment_factor": augmentation_cfg.augment_factor,
        "use_mixup": augmentation_cfg.use_mixup,
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro_f1"],
        "test_macro_recall": report["macro_recall"],
        "top_confusion_pair": _top_confusion_pair_text(report),
    }
    dump_json(run_dir / "offline_summary.json", summary)
    append_csv_row(Path(args.run_root) / "offline_results.csv", OFFLINE_SUMMARY_FIELDS, summary)

    run_metadata = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_path": args.config,
        "recordings_manifest_path": str(recordings_manifest_path) if recordings_manifest_path else None,
        "quality_report": str(q_path),
        "training_history": str(history_path),
        "evaluation_outputs": report_paths,
        "elapsed_minutes": (time.time() - start) / 60.0,
    }
    dump_json(run_dir / "run_metadata.json", run_metadata)

    elapsed_min = (time.time() - start) / 60.0
    logger.info("Training finished in %.1f min", elapsed_min)


if __name__ == "__main__":
    main()
