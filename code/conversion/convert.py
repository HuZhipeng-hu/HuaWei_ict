"""Model conversion entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversion.export import export_to_mindir
from conversion.quantize import quantize_model
from shared.config import ConversionConfig, ModelConfig, load_config, load_conversion_config, load_training_config
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, ensure_run_dir
from shared.models import count_parameters
from training.model import build_model_from_config
from shared.preprocessing import PreprocessPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("conversion")

CONVERSION_SUMMARY_FIELDS = [
    "run_id",
    "checkpoint_path",
    "output_path",
    "input_shape",
    "model_type",
    "base_channels",
    "use_se",
    "quantized_output",
]


def _derive_training_protocol(
    training_config_path: Path,
) -> Optional[tuple[Tuple[int, int, int, int], ModelConfig]]:
    if not training_config_path.exists():
        return None

    train_model_cfg, train_pp_cfg, _, _ = load_training_config(str(training_config_path))
    pipeline = PreprocessPipeline(train_pp_cfg)
    expected_shape = (1,) + tuple(pipeline.get_output_shape())
    return expected_shape, train_model_cfg


def _validate_input_shape(input_shape: Tuple[int, ...], model_config: ModelConfig) -> None:
    if len(input_shape) != 4:
        raise ValueError(f"input_shape must be 4D (N,C,F,T), got {input_shape}")
    if input_shape[0] != 1:
        logger.warning(
            "input_shape batch=%s (not 1). Export supports it, but runtime usually uses batch=1.",
            input_shape[0],
        )
    if input_shape[1] != model_config.in_channels:
        raise ValueError(
            "input_shape channels=%s mismatch model in_channels=%s."
            % (input_shape[1], model_config.in_channels)
        )


def _validate_training_shape_strict(
    input_shape: Tuple[int, ...],
    model_config: ModelConfig,
    training_protocol: Optional[tuple[Tuple[int, int, int, int], ModelConfig]],
    training_cfg_path: Path,
) -> None:
    if training_protocol is None:
        return

    expected_shape, train_model_cfg = training_protocol

    if expected_shape != input_shape:
        raise ValueError(
            "input_shape %s differs from training-derived shape %s from %s. "
            "Please align conversion config with training preprocess settings."
            % (input_shape, expected_shape, training_cfg_path)
        )

    if train_model_cfg.in_channels != model_config.in_channels:
        raise ValueError(
            "conversion.model.in_channels=%s differs from training.model.in_channels=%s."
            % (model_config.in_channels, train_model_cfg.in_channels)
        )

    if train_model_cfg.num_classes != model_config.num_classes:
        raise ValueError(
            "conversion.model.num_classes=%s differs from training.model.num_classes=%s."
            % (model_config.num_classes, train_model_cfg.num_classes)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroGrip Pro V2 model conversion")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt model file.")
    parser.add_argument("--output", type=str, default=None, help="Output path stem (extension .mindir added automatically).")
    parser.add_argument("--config", type=str, default="configs/conversion.yaml", help="Path to conversion YAML config.")
    parser.add_argument("--run_id", default=None, help="Stable experiment run id")
    parser.add_argument("--run_root", default="artifacts/runs", help="Base directory for run artifacts")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 post-training quantization.")
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Model input shape as comma-separated values, e.g. 1,12,24,6.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NeuroGrip Pro V2 - Model Conversion")
    logger.info("=" * 60)

    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="convert")
    config_path = Path(args.config)
    if config_path.exists():
        conversion_cfg = load_conversion_config(str(config_path))
        raw_config = load_config(str(config_path))
        copy_config_snapshot(config_path, run_dir / "config_snapshots" / config_path.name)
    else:
        logger.warning("Config not found: %s. Using default ConversionConfig.", config_path)
        conversion_cfg = ConversionConfig()
        raw_config = {}

    training_cfg_path = config_path.parent / "training.yaml"
    training_protocol = _derive_training_protocol(training_cfg_path)
    if training_cfg_path.exists():
        copy_config_snapshot(training_cfg_path, run_dir / "config_snapshots" / training_cfg_path.name)

    model_section = raw_config.get("model", {})
    if model_section:
        model_config = ModelConfig(**model_section)
    elif training_protocol is not None:
        _, model_config = training_protocol
    else:
        model_config = ModelConfig()

    checkpoint_path = args.checkpoint or conversion_cfg.checkpoint_path
    if args.output:
        output_path = args.output
    else:
        default_name = Path(conversion_cfg.output_path).with_suffix("").name
        output_path = str(run_dir / "conversion" / default_name)
    if not checkpoint_path:
        raise ValueError("checkpoint path is required via --checkpoint or conversion.checkpoint_path")

    if args.input_shape:
        input_shape = tuple(int(x.strip()) for x in args.input_shape.split(","))
    else:
        input_shape = tuple(int(x) for x in conversion_cfg.input_shape)

    _validate_input_shape(input_shape, model_config)
    _validate_training_shape_strict(input_shape, model_config, training_protocol, training_cfg_path)

    quant_section = raw_config.get("quantize", {})
    quantize_enabled = args.quantize or bool(quant_section.get("enabled", False))
    quantize_bit_num = int(quant_section.get("bit_num", 8))

    logger.info("Run ID: %s", run_id)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Output: %s", output_path)
    logger.info("Using input shape: %s", input_shape)
    logger.info("Quantization: %s (bit_num=%s)", "enabled" if quantize_enabled else "disabled", quantize_bit_num)

    model = build_model_from_config(model_config, dropout_rate=0.0)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    mindir_path = export_to_mindir(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=input_shape,
    )

    quantized_path = None
    if quantize_enabled:
        quantized_output = str(Path(output_path).with_suffix("")) + "_int8"
        quantized_path = quantize_model(
            input_path=mindir_path,
            output_path=quantized_output,
            input_shape=input_shape,
            bit_num=quantize_bit_num,
        )
        logger.info("Quantized model saved: %s", quantized_path)

    summary = {
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "output_path": mindir_path,
        "input_shape": list(input_shape),
        "model_type": model_config.model_type,
        "base_channels": model_config.base_channels,
        "use_se": model_config.use_se,
        "quantized_output": quantized_path or "",
    }
    dump_json(run_dir / "conversion" / "conversion_summary.json", summary)
    append_csv_row(Path(args.run_root) / "conversion_results.csv", CONVERSION_SUMMARY_FIELDS, summary)

    logger.info("Conversion completed.")


if __name__ == "__main__":
    main()
