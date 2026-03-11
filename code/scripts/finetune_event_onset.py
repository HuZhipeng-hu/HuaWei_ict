"""Unified finetuning entrypoint for event-onset training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.train_pipeline import run_event_training


def _parse_optional_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune event-onset model on wearer data")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--recordings_manifest", default=None)

    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--use_se", type=_parse_optional_bool, default=None)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--hard_mining_ratio", type=float, default=None)
    parser.add_argument("--augment_factor", type=int, default=None)
    parser.add_argument("--use_mixup", type=_parse_optional_bool, default=None)
    parser.add_argument("--augmentation_enabled", type=_parse_optional_bool, default=None)
    parser.add_argument("--split_seed", type=int, default=None)

    parser.add_argument("--split_manifest_in", default=None)
    parser.add_argument("--split_manifest_out", default=None)
    parser.add_argument("--manifest_strategy", default="v2", choices=["v1", "v2"])
    parser.add_argument("--quality_report_out", default=None)
    parser.add_argument("--eval_protocol", default="same_user_same_day_v1")

    parser.add_argument(
        "--pretrained_emg_checkpoint",
        default=None,
        help="Optional DB5 pretrained checkpoint for EMG encoder warm start.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    logger = logging.getLogger("event_onset.finetune")
    if args.pretrained_emg_checkpoint:
        logger.info("Using pretrained EMG checkpoint: %s", args.pretrained_emg_checkpoint)
    else:
        logger.warning("No pretrained EMG checkpoint supplied. Finetune will run from random initialization.")
    run_event_training(args)


if __name__ == "__main__":
    main()
