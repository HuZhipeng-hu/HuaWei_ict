"""
Model conversion entrypoint.

Examples:
    python -m conversion.convert --checkpoint checkpoints/neurogrip_best.ckpt --output models/neurogrip
    python -m conversion.convert --checkpoint checkpoints/neurogrip_best.ckpt --output models/neurogrip --quantize
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversion.export import export_to_mindir
from conversion.quantize import quantize_model
from shared.config import ModelConfig, load_config, load_training_config
from shared.models import count_parameters, create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("conversion")


def _derive_expected_input_shape_from_training(training_config_path: Path):
    if not training_config_path.exists():
        return None

    train_model_cfg, train_pp_cfg, _, _ = load_training_config(str(training_config_path))
    freq_bins = train_pp_cfg.stft_n_fft // 2 + 1
    time_frames = max(
        1,
        (train_pp_cfg.segment_length - train_pp_cfg.stft_window_size)
        // train_pp_cfg.stft_hop_size
        + 1,
    )
    return (
        1,
        train_pp_cfg.num_channels,
        freq_bins,
        time_frames,
        train_model_cfg,
        train_pp_cfg,
    )


def _validate_input_shape(input_shape, model_config: ModelConfig) -> None:
    if len(input_shape) != 4:
        raise ValueError(f"input_shape must be 4D (N,C,F,T), got {input_shape}")
    if input_shape[0] != 1:
        logger.warning(
            "input_shape batch=%s (not 1). Export supports it, but runtime usually uses batch=1.",
            input_shape[0],
        )
    if input_shape[1] != model_config.in_channels:
        logger.warning(
            "input_shape channels=%s mismatch model in_channels=%s.",
            input_shape[1],
            model_config.in_channels,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroGrip Pro V2 model conversion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt model file.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path stem (extension .mindir added automatically).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/conversion.yaml",
        help="Path to conversion YAML config.",
    )
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 post-training quantization.")
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Model input shape as comma-separated values, e.g. 1,6,24,6.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NeuroGrip Pro V2 - Model Conversion")
    logger.info("=" * 60)

    config_path = Path(args.config)
    raw_config = {}
    if config_path.exists():
        raw = load_config(str(config_path))
        if isinstance(raw, dict):
            raw_config = raw
        else:
            logger.warning("Config is not a dict-like structure, using defaults.")

    model_section = raw_config.get("model", {})
    export_section = raw_config.get("export", {})
    quant_section = raw_config.get("quantize", {})

    if args.input_shape:
        input_shape = tuple(int(x.strip()) for x in args.input_shape.split(","))
    else:
        config_input_shape = export_section.get("input_shape", [1, 6, 24, 6])
        input_shape = tuple(int(x) for x in config_input_shape)

    model_config = ModelConfig(**model_section) if model_section else ModelConfig()
    _validate_input_shape(input_shape, model_config)
    logger.info("Using input shape: %s", input_shape)

    # Warn when conversion input shape drifts from training preprocessing.
    training_cfg_path = config_path.parent / "training.yaml"
    expected = _derive_expected_input_shape_from_training(training_cfg_path)
    if expected is not None:
        expected_shape = expected[:4]
        train_model_cfg = expected[4]
        if expected_shape != input_shape:
            logger.warning(
                "input_shape %s differs from training-derived shape %s from %s. "
                "This can cause runtime mismatch.",
                input_shape,
                expected_shape,
                training_cfg_path,
            )
        if train_model_cfg.in_channels != model_config.in_channels:
            logger.warning(
                "conversion.model.in_channels=%s differs from training.model.in_channels=%s.",
                model_config.in_channels,
                train_model_cfg.in_channels,
            )
        if train_model_cfg.num_classes != model_config.num_classes:
            logger.warning(
                "conversion.model.num_classes=%s differs from training.model.num_classes=%s.",
                model_config.num_classes,
                train_model_cfg.num_classes,
            )
    else:
        logger.info("No training.yaml found for cross-check, skipping shape consistency check.")

    quantize_enabled = args.quantize or bool(quant_section.get("enabled", False))
    quantize_bit_num = int(quant_section.get("bit_num", 8))
    logger.info("Quantization: %s (bit_num=%s)", "enabled" if quantize_enabled else "disabled", quantize_bit_num)

    model = create_model(
        {
            "model_type": model_config.model_type,
            "in_channels": model_config.in_channels,
            "num_classes": model_config.num_classes,
            "base_channels": model_config.base_channels,
            "use_se": model_config.use_se,
            "dropout_rate": 0.0,
        }
    )
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    mindir_path = export_to_mindir(
        model=model,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=input_shape,
    )

    if quantize_enabled:
        quantized_path = args.output + "_int8"
        quantize_model(
            input_path=mindir_path,
            output_path=quantized_path,
            input_shape=input_shape,
            bit_num=quantize_bit_num,
        )
        logger.info("Quantized model saved: %s.mindir", quantized_path)

    logger.info("Conversion completed.")


if __name__ == "__main__":
    main()
