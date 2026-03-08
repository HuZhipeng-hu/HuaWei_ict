"""Evaluate checkpoint with fixed split manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from shared.config import load_training_config
from shared.gestures import GESTURE_DEFINITIONS
from training.data.csv_dataset import CSVDatasetLoader
from training.data.split_strategy import load_manifest
from training.evaluate import load_and_evaluate
from training.reporting import save_classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test split")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_dir", default="logs/evaluation")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--eval_protocol", default="same_user_same_day_v1")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    logger = logging.getLogger("eval_ckpt")
    logger.info("Loading config: %s", args.config)

    model_cfg, preprocess_cfg, train_cfg, _ = load_training_config(args.config)
    gesture_to_idx = {g.name: i for i, g in enumerate(GESTURE_DEFINITIONS)}
    class_names = [g.name for g in GESTURE_DEFINITIONS]

    if preprocess_cfg.dual_branch.enabled and model_cfg.in_channels != 12:
        logger.warning("dual_branch enabled; overriding model.in_channels=%d -> 12", model_cfg.in_channels)
        model_cfg.in_channels = 12

    loader = CSVDatasetLoader(
        args.data_dir,
        gesture_to_idx,
        preprocess_cfg,
        quality_filter=train_cfg.quality_filter,
    )
    samples, labels, _, _ = loader.load_all_with_sources(return_metadata=True)
    manifest = load_manifest(args.split_manifest)
    if manifest.num_samples != int(samples.shape[0]):
        raise ValueError(
            f"Manifest sample count mismatch: manifest={manifest.num_samples}, loaded={samples.shape[0]}. "
            "Regenerate manifest with current preprocess settings."
        )

    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    test_x = samples[test_idx]
    test_y = labels[test_idx]

    report = load_and_evaluate(
        ckpt_path=args.checkpoint,
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
    report["manifest_path"] = args.split_manifest
    outputs = save_classification_report(report, out_dir=args.output_dir, prefix="test")
    logger.info("Saved evaluation report: %s", outputs)


if __name__ == "__main__":
    main()

