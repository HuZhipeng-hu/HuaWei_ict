"""
Evaluate a checkpoint against a fixed split manifest.

Example:
    python scripts/evaluate_ckpt.py \
      --checkpoint checkpoints/neurogrip_best.ckpt \
      --split_manifest artifacts/splits/default_manifest.json \
      --output_dir logs/eval_run_01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import ModelConfig, load_training_config
from shared.models import create_model
from shared.preprocessing import PreprocessPipeline
from training.data.csv_dataset import CSVDatasetLoader
from training.data.split_strategy import load_manifest, split_arrays_from_manifest
from training.evaluate import load_and_evaluate
from training.reporting import save_classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_ckpt")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with split manifest")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--split_manifest", type=str, required=True, help="Path to split manifest JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for evaluation artifacts")

    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Training YAML for model/preprocess config")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset root (default from manifest)")
    return parser.parse_args()


def _create_eval_model(model_config: ModelConfig):
    return create_model(
        {
            "model_type": model_config.model_type,
            "in_channels": model_config.in_channels,
            "num_classes": model_config.num_classes,
            "base_channels": model_config.base_channels,
            "use_se": model_config.use_se,
            "dropout_rate": 0.0,
        }
    )


def main():
    args = parse_args()

    model_config, preprocess_config, _, _ = load_training_config(args.config)
    dual_cfg = preprocess_config.dual_branch
    dual_enabled = bool(dual_cfg.get("enabled", False)) if isinstance(dual_cfg, dict) else bool(getattr(dual_cfg, "enabled", False))
    if dual_enabled:
        expected_channels = int(preprocess_config.num_channels) * 2
        if model_config.in_channels != expected_channels:
            logger.warning(
                "dual_branch enabled, overriding model.in_channels from %s to %s",
                model_config.in_channels,
                expected_channels,
            )
            model_config.in_channels = expected_channels
    manifest = load_manifest(args.split_manifest)

    data_dir = args.data_dir or manifest.data_dir
    if not data_dir:
        raise ValueError("data_dir is required when manifest does not include data_dir")

    logger.info("Using dataset: %s", data_dir)
    logger.info("Using manifest: %s", args.split_manifest)

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

    loader = CSVDatasetLoader(
        data_dir=data_dir,
        preprocess=pipeline,
        num_emg_channels=preprocess_config.total_channels,
        device_sampling_rate=preprocess_config.device_sampling_rate,
        target_sampling_rate=int(preprocess_config.sampling_rate),
        segment_length=preprocess_config.segment_length,
        segment_stride=preprocess_config.segment_stride,
    )

    samples, labels, _ = loader.load_all_with_sources()
    if manifest.num_samples and manifest.num_samples != len(samples):
        raise ValueError(
            "Manifest sample count mismatch: "
            f"manifest={manifest.num_samples}, loaded={len(samples)}. "
            "Please evaluate with a manifest generated under the same preprocess settings."
        )

    _, _, (test_samples, test_labels) = split_arrays_from_manifest(samples, labels, manifest)
    if len(test_samples) == 0:
        raise RuntimeError("Test split is empty in manifest")

    model = _create_eval_model(model_config)
    results = load_and_evaluate(model, args.checkpoint, test_samples, test_labels)

    report = results.get("report")
    if not isinstance(report, dict):
        raise RuntimeError("Evaluation did not return structured report")

    artifacts = save_classification_report(report, output_dir=args.output_dir, prefix="test")
    logger.info("Saved artifacts: %s", artifacts)


if __name__ == "__main__":
    main()
