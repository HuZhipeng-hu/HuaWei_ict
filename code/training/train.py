"""Training entrypoint."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
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
from shared.preprocessing import PreprocessPipeline
from training.data.csv_dataset import CSVDatasetLoader
from training.data.split_strategy import build_manifest, load_manifest, save_manifest
from training.evaluate import load_and_evaluate
from training.model import NeuroGripNet
from training.reporting import save_classification_report
from training.trainer import Trainer

logger = logging.getLogger("training")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuroGrip model")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--data_dir", required=True, help="Dataset root folder")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)

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
    manifest_out_path = manifest_out_cli or manifest_in_config

    if manifest_in_cli:
        if not os.path.exists(manifest_in_cli):
            raise FileNotFoundError(
                "Explicit --split_manifest_in path does not exist: "
                f"{manifest_in_cli}. Fix: generate first with --split_manifest_out <path> "
                "or use an existing --split_manifest_in <path>."
            )
        logger.info("Using split manifest from CLI: %s", manifest_in_cli)
        manifest = load_manifest(manifest_in_cli)
        return manifest, manifest_in_cli

    if manifest_in_config:
        if os.path.exists(manifest_in_config):
            logger.info("Using split manifest from config: %s", manifest_in_config)
            manifest = load_manifest(manifest_in_config)
            return manifest, manifest_in_config

        logger.info(
            "Configured split manifest does not exist (%s). Building one with strategy=%s.",
            manifest_in_config,
            manifest_strategy,
        )
        manifest = build_manifest(
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
        saved = save_manifest(manifest, manifest_out_path or manifest_in_config)
        logger.info("Auto-generated split manifest at %s", saved)
        return manifest, str(saved)

    manifest = build_manifest(
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


def _split_by_manifest(
    samples: np.ndarray,
    labels: np.ndarray,
    manifest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx = np.asarray(manifest.train_indices, dtype=np.int32)
    val_idx = np.asarray(manifest.val_indices, dtype=np.int32)
    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    return (
        samples[train_idx],
        labels[train_idx],
        samples[val_idx],
        labels[val_idx],
        samples[test_idx],
        labels[test_idx],
    )


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

    model_cfg, preprocess_cfg, train_cfg, _ = load_training_config(args.config)
    data_cfg = load_training_data_config(args.config)
    gesture_to_idx, class_names = _gesture_mappings()
    if len(class_names) != model_cfg.num_classes:
        raise ValueError(f"num_classes mismatch: model={model_cfg.num_classes}, gestures={len(class_names)}")
    logger.info("Gesture definition check passed: %d classes", len(class_names))

    if preprocess_cfg.dual_branch.enabled and model_cfg.in_channels != 12:
        logger.warning(
            "dual_branch.enabled=true requires in_channels=12. Overriding model.in_channels from %d to 12.",
            model_cfg.in_channels,
        )
        model_cfg.in_channels = 12

    logger.info("Loading dataset from: %s", args.data_dir)
    loader = CSVDatasetLoader(
        args.data_dir,
        gesture_to_idx,
        preprocess_cfg,
        quality_filter=train_cfg.quality_filter,
    )
    dataset_stats = loader.get_stats()
    logger.info("Dataset stats: %s", dataset_stats)
    samples, labels, source_ids, source_meta = loader.load_all_with_sources(return_metadata=True)
    logger.info("Loaded samples: %d, shape=%s", samples.shape[0], tuple(samples.shape))

    quality_report_path = args.quality_report_out or "logs/quality/quality_report.json"
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

    train_x, train_y, val_x, val_y, test_x, test_y = _split_by_manifest(samples, labels, manifest)
    logger.info("Split sizes => train=%d, val=%d, test=%d", len(train_y), len(val_y), len(test_y))

    model = NeuroGripNet(
        in_channels=model_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        dropout_rate=model_cfg.dropout_rate,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
    )

    trainer = Trainer(model, train_cfg, class_names, output_dir=".")
    history = trainer.train(train_x, train_y, val_x, val_y)
    _save_history(history, "logs/training_history.csv")
    logger.info("训练历史已保存: logs/training_history.csv")

    logger.info("")
    logger.info("================================================================")
    logger.info("Final evaluation on test split")
    logger.info("================================================================")
    report = load_and_evaluate(
        ckpt_path="checkpoints/neurogrip_best.ckpt",
        samples=test_x,
        labels=test_y,
        class_names=class_names,
        in_channels=model_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        dropout_rate=model_cfg.dropout_rate,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        device_target=args.device_target,
        device_id=args.device_id,
    )

    report["eval_protocol"] = args.eval_protocol
    report["manifest_path"] = manifest_path
    report_paths = save_classification_report(report, out_dir="logs/evaluation", prefix="test")
    logger.info("Saved evaluation report: %s", report_paths)

    elapsed_min = (time.time() - start) / 60.0
    logger.info("Training finished in %.1f min", elapsed_min)


if __name__ == "__main__":
    main()

