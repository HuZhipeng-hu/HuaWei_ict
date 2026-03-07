"""
Training entrypoint.

Examples:
    python -m training.train --config configs/training.yaml --data_dir ../data
    python -m training.train --data_dir ../data --split_mode grouped_file --test_ratio 0.2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import (  # noqa: E402
    AugmentationConfig,
    ModelConfig,
    PreprocessConfig,
    TrainingConfig,
    load_training_config,
)
from shared.gestures import NUM_CLASSES, validate_gesture_definitions  # noqa: E402
from shared.models import count_parameters, create_model  # noqa: E402
from shared.preprocessing import PreprocessPipeline  # noqa: E402
from training.data.augmentation import DataAugmentor  # noqa: E402
from training.data.csv_dataset import CSVDatasetLoader  # noqa: E402
from training.data.split_strategy import (  # noqa: E402
    SPLIT_MODES,
    build_manifest,
    grouped_kfold_indices,
    legacy_kfold_indices,
    load_manifest,
    save_manifest,
    split_and_optionally_augment,
    split_arrays_from_manifest,
)
from training.evaluate import load_and_evaluate  # noqa: E402
from training.reporting import save_classification_report  # noqa: E402
from training.trainer import Trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroGrip model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Training YAML path")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["CPU", "GPU", "Ascend"],
        help="Override training device",
    )
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")

    parser.add_argument(
        "--split_mode",
        type=str,
        default=None,
        choices=list(SPLIT_MODES),
        help="Split mode: legacy or grouped_file",
    )
    parser.add_argument("--test_ratio", type=float, default=None, help="Test split ratio")
    parser.add_argument("--split_manifest_in", type=str, default=None, help="Load split manifest from path")
    parser.add_argument("--split_manifest_out", type=str, default=None, help="Write split manifest to path")

    return parser.parse_args()


def _create_model(model_config: ModelConfig, for_eval: bool = False):
    return create_model(
        {
            "model_type": model_config.model_type,
            "in_channels": model_config.in_channels,
            "num_classes": model_config.num_classes,
            "base_channels": model_config.base_channels,
            "use_se": model_config.use_se,
            "dropout_rate": 0.0 if for_eval else model_config.dropout_rate,
        }
    )


def _build_augmentor_if_enabled(aug_config: AugmentationConfig):
    if not aug_config.enabled:
        return None
    return DataAugmentor(
        time_warp_rate=aug_config.time_warp_rate,
        amplitude_scale=aug_config.amplitude_scale,
        noise_std=aug_config.noise_std,
        mixup_alpha=getattr(aug_config, "mixup_alpha", 0.2),
    )


def _log_split_summary(split_name: str, labels: np.ndarray) -> None:
    counts = [int(np.sum(labels == class_id)) for class_id in range(NUM_CLASSES)]
    logger.info("%s split: %s samples, class_counts=%s", split_name, len(labels), counts)


def _prepare_split_manifest(
    *,
    labels: np.ndarray,
    source_ids: np.ndarray,
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    data_dir: str,
    cli_manifest_in: str | None,
    config_manifest_path: str | None,
    manifest_out: str | None,
):
    """
    Resolve/load/build split manifest with portable fallback behavior.

    Rules:
    - Explicit --split_manifest_in is strict: missing file raises FileNotFoundError.
    - Config split_manifest_path is soft: missing file triggers auto build + save.
    - If neither input path exists, build in-memory and save only when manifest_out is provided.
    """
    explicit_manifest_path = Path(cli_manifest_in) if cli_manifest_in else None
    config_manifest = Path(config_manifest_path) if config_manifest_path else None
    manifest_out_path = Path(manifest_out) if manifest_out else None

    def _build() -> object:
        built_manifest = build_manifest(
            labels=labels,
            source_ids=source_ids,
            split_mode=split_mode,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
            data_dir=str(Path(data_dir).resolve()),
        )
        logger.info(
            "Built split manifest mode=%s seed=%s val_ratio=%.3f test_ratio=%.3f",
            built_manifest.split_mode,
            built_manifest.seed,
            built_manifest.val_ratio,
            built_manifest.test_ratio,
        )
        return built_manifest

    if explicit_manifest_path is not None:
        if not explicit_manifest_path.exists():
            raise FileNotFoundError(
                "Split manifest file not found for explicit --split_manifest_in: "
                f"{explicit_manifest_path}. "
                "To create one first, run training once with "
                "--split_manifest_out <path>; or pass an existing file via "
                "--split_manifest_in <path>."
            )

        manifest = load_manifest(explicit_manifest_path)
        logger.info("Using split manifest from explicit --split_manifest_in: %s", explicit_manifest_path)
        if manifest_out_path is not None:
            save_manifest(manifest, manifest_out_path)
            logger.info("Saved split manifest copy to: %s", manifest_out_path)
        return manifest

    if config_manifest is not None and config_manifest.exists():
        manifest = load_manifest(config_manifest)
        logger.info("Using split manifest from config: %s", config_manifest)
        if manifest_out_path is not None:
            save_manifest(manifest, manifest_out_path)
            logger.info("Saved split manifest copy to: %s", manifest_out_path)
        return manifest

    manifest = _build()

    if config_manifest is not None and not config_manifest.exists():
        save_target = manifest_out_path or config_manifest
        save_manifest(manifest, save_target)
        logger.info("Auto-generated split manifest at %s", save_target)
        return manifest

    if manifest_out_path is not None:
        save_manifest(manifest, manifest_out_path)
        logger.info("Saved split manifest to: %s", manifest_out_path)

    return manifest


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 64)
    logger.info("NeuroGrip model training")
    logger.info("=" * 64)

    config_path = Path(args.config)
    if config_path.exists():
        logger.info("Loading config: %s", config_path)
        model_config, preprocess_config, train_config, aug_config = load_training_config(str(config_path))
    else:
        logger.info("Config not found, using default config.")
        model_config = ModelConfig()
        preprocess_config = PreprocessConfig()
        train_config = TrainingConfig()
        aug_config = AugmentationConfig()

    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.device is not None:
        train_config.device = args.device
    if args.no_augment:
        aug_config.enabled = False
    if args.split_mode is not None:
        train_config.split_mode = args.split_mode
    if args.test_ratio is not None:
        train_config.test_ratio = args.test_ratio

    if train_config.split_mode not in SPLIT_MODES:
        raise ValueError(f"Invalid split_mode={train_config.split_mode!r}, expected one of {SPLIT_MODES}")

    dual_cfg = getattr(preprocess_config, "dual_branch", None)
    if isinstance(dual_cfg, dict):
        dual_branch_enabled = bool(dual_cfg.get("enabled", False))
    else:
        dual_branch_enabled = bool(getattr(dual_cfg, "enabled", False))
    if dual_branch_enabled:
        expected_channels = int(preprocess_config.num_channels) * 2
        if model_config.in_channels != expected_channels:
            logger.warning(
                "dual_branch enabled, overriding model.in_channels from %s to %s",
                model_config.in_channels,
                expected_channels,
            )
            model_config.in_channels = expected_channels

    validate_gesture_definitions()
    logger.info("Gesture definition check passed: %s classes", NUM_CLASSES)

    pipeline = PreprocessPipeline(
        sampling_rate=preprocess_config.sampling_rate,
        num_channels=preprocess_config.num_channels,
        lowcut=preprocess_config.lowcut,
        highcut=preprocess_config.highcut,
        filter_order=preprocess_config.filter_order,
        stft_window_size=preprocess_config.stft_window_size,
        stft_hop_size=preprocess_config.stft_hop_size,
        stft_n_fft=preprocess_config.stft_n_fft,
        device_sampling_rate=preprocess_config.device_sampling_rate,
        segment_length=preprocess_config.segment_length,
        segment_stride=preprocess_config.segment_stride,
        dual_branch=preprocess_config.dual_branch,
    )

    logger.info("Loading dataset from: %s", args.data_dir)
    loader = CSVDatasetLoader(
        data_dir=args.data_dir,
        preprocess=pipeline,
        num_emg_channels=preprocess_config.total_channels,
        device_sampling_rate=preprocess_config.device_sampling_rate,
        target_sampling_rate=int(preprocess_config.sampling_rate),
        segment_length=preprocess_config.segment_length,
        segment_stride=preprocess_config.segment_stride,
    )

    logger.info("Dataset stats: %s", loader.get_stats())
    samples, labels, source_ids = loader.load_all_with_sources()
    logger.info("Loaded samples: %s, shape=%s", len(samples), samples.shape)

    manifest = _prepare_split_manifest(
        labels=labels,
        source_ids=source_ids,
        split_mode=train_config.split_mode,
        val_ratio=train_config.val_ratio,
        test_ratio=train_config.test_ratio,
        split_seed=train_config.split_seed,
        data_dir=args.data_dir,
        cli_manifest_in=args.split_manifest_in,
        config_manifest_path=train_config.split_manifest_path,
        manifest_out=args.split_manifest_out,
    )
    if manifest.num_samples and manifest.num_samples != len(samples):
        raise ValueError(
            "Manifest sample count mismatch: "
            f"manifest={manifest.num_samples}, loaded={len(samples)}. "
            "This usually means preprocess/split settings changed. "
            "Please regenerate manifest with --split_manifest_out and retrain."
        )

    if manifest.split_mode == "grouped_file":
        train_src = set(manifest.train_sources)
        val_src = set(manifest.val_sources)
        test_src = set(manifest.test_sources)
        if train_src & val_src or train_src & test_src or val_src & test_src:
            raise RuntimeError("Grouped split source leakage detected in manifest")

    augmentor = _build_augmentor_if_enabled(aug_config)
    use_mixup = getattr(aug_config, "use_mixup", False)
    augment_factor = int(getattr(aug_config, "augment_factor", 1))

    (train_samples, train_labels), (val_samples, val_labels), (test_samples, test_labels) = split_and_optionally_augment(
        samples=samples,
        labels=labels,
        manifest=manifest,
        augmentor=augmentor,
        augment_factor=augment_factor,
        use_mixup=use_mixup,
    )

    if len(train_samples) == 0:
        raise RuntimeError("Train split is empty after split")
    if len(val_samples) == 0:
        raise RuntimeError("Validation split is empty after split")
    if len(test_samples) == 0:
        raise RuntimeError("Test split is empty after split")

    _log_split_summary("Train", train_labels)
    _log_split_summary("Val", val_labels)
    _log_split_summary("Test", test_labels)

    if train_config.kfold > 1:
        logger.info("=" * 64)
        logger.info("Running KFold validation: k=%s (with fixed independent test split)", train_config.kfold)
        logger.info("=" * 64)

        dev_indices = np.asarray(manifest.train_indices + manifest.val_indices, dtype=np.int64)
        if manifest.split_mode == "grouped_file":
            fold_iterator = grouped_kfold_indices(
                labels=labels,
                source_ids=source_ids,
                base_indices=dev_indices,
                k=train_config.kfold,
                seed=train_config.split_seed,
            )
        else:
            fold_iterator = legacy_kfold_indices(
                labels=labels,
                base_indices=dev_indices,
                k=train_config.kfold,
                seed=train_config.split_seed,
            )

        fold_val_accs = []
        for fold_idx, fold_train_idx, fold_val_idx in fold_iterator:
            logger.info("Fold %s/%s", fold_idx + 1, train_config.kfold)
            fold_train_samples = samples[fold_train_idx]
            fold_train_labels = labels[fold_train_idx]
            fold_val_samples = samples[fold_val_idx]
            fold_val_labels = labels[fold_val_idx]

            if augmentor is not None and (augment_factor > 1 or use_mixup):
                fold_train_samples, fold_train_labels = augmentor.augment_batch(
                    fold_train_samples,
                    fold_train_labels,
                    factor=augment_factor,
                    use_mixup=use_mixup,
                )

            fold_model = _create_model(model_config, for_eval=False)
            fold_trainer = Trainer(fold_model, train_config)
            fold_trainer.train(
                train_data=(fold_train_samples, fold_train_labels),
                val_data=(fold_val_samples, fold_val_labels),
            )
            fold_val_accs.append(float(fold_trainer.best_val_acc))
            logger.info("Fold %s best val acc: %.4f", fold_idx + 1, fold_trainer.best_val_acc)

        logger.info(
            "KFold done: mean=%.4f std=%.4f values=%s",
            float(np.mean(fold_val_accs)),
            float(np.std(fold_val_accs)),
            [f"{v:.4f}" for v in fold_val_accs],
        )

    model = _create_model(model_config, for_eval=False)
    logger.info(
        "Model created: type=%s params=%s",
        model_config.model_type,
        f"{count_parameters(model):,}",
    )

    trainer = Trainer(model, train_config)
    trainer.train(
        train_data=(train_samples, train_labels),
        val_data=(val_samples, val_labels),
    )

    logger.info("\n" + "=" * 64)
    logger.info("Final evaluation on test split")
    logger.info("=" * 64)

    best_ckpt = Path(train_config.checkpoint_dir) / "neurogrip_best.ckpt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    eval_model = _create_model(model_config, for_eval=True)
    test_results = load_and_evaluate(
        eval_model,
        str(best_ckpt),
        test_samples,
        test_labels,
    )

    report = test_results.get("report")
    if isinstance(report, dict):
        output_dir = Path(train_config.log_dir) / "evaluation"
        artifacts = save_classification_report(report, output_dir=output_dir, prefix="test")
        logger.info("Saved evaluation report: %s", artifacts)

    elapsed = time.time() - start_time
    logger.info("Training finished in %.1f min", elapsed / 60.0)


if __name__ == "__main__":
    main()
