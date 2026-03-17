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
from shared.config import load_config


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
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument(
        "--target_db5_keys",
        default=None,
        help=(
            "Comma-separated action keys, "
            "e.g. TENSE_OPEN,V_SIGN,OK_SIGN,THUMB_UP,WRIST_CW,WRIST_CCW. CONTINUE/RELAX is implicit."
        ),
    )

    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--use_se", type=_parse_optional_bool, default=None)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--hard_mining_ratio", type=float, default=None)
    parser.add_argument("--freeze_emg_epochs", type=int, default=None)
    parser.add_argument("--encoder_lr_ratio", type=float, default=None)
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
        "--budget_per_class",
        type=int,
        default=60,
        help="Train-only sample budget per class (default: 60). Validation/test splits are unchanged.",
    )
    parser.add_argument(
        "--budget_seed",
        type=int,
        default=42,
        help="Random seed for budgeted train subset sampling.",
    )

    parser.add_argument(
        "--pretrained_emg_checkpoint",
        default=None,
        help="Optional DB5 pretrained checkpoint for EMG encoder warm start.",
    )
    parser.add_argument(
        "--incremental_from_checkpoint",
        default=None,
        help="Optional previous event-onset checkpoint for incremental head expansion.",
    )
    parser.add_argument(
        "--incremental_old_target_db5_keys",
        default=None,
        help="Comma-separated old action keys used by incremental_from_checkpoint (CONTINUE/RELAX is implicit).",
    )
    parser.add_argument(
        "--incremental_head_only",
        type=_parse_optional_bool,
        default=True,
        help="When true, prioritize head-only phase before unfreezing for incremental action extension.",
    )
    parser.add_argument(
        "--incremental_init_seed",
        type=int,
        default=42,
        help="Seed used for incremental-class head row initialization.",
    )
    return parser


def _resolve_recordings_manifest_path(*, data_dir: str, config_path: str, manifest_arg: str | None) -> str:
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
        "Event-onset finetune requires recordings manifest with event metadata. "
        f"Missing file: {expected}. "
        "Fix: pass --recordings_manifest <path/to/recordings_manifest.csv>."
    )


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
    if args.incremental_from_checkpoint:
        logger.info(
            "Incremental mode enabled: checkpoint=%s old_target_db5_keys=%s",
            args.incremental_from_checkpoint,
            args.incremental_old_target_db5_keys or "(not provided)",
        )
    args.recordings_manifest = _resolve_recordings_manifest_path(
        data_dir=str(args.data_dir),
        config_path=str(args.config),
        manifest_arg=args.recordings_manifest,
    )
    logger.info("Using recordings_manifest: %s", args.recordings_manifest)
    logger.info(
        "Budget mode: budget_per_class=%d budget_seed=%d",
        int(args.budget_per_class),
        int(args.budget_seed),
    )
    run_event_training(args)


if __name__ == "__main__":
    main()
