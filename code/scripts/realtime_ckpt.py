"""Compatibility wrapper for CKPT debug runtime in event-onset pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from scripts.run_event_runtime import main as run_event_runtime_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run event-onset debug runtime with CKPT backend")
    parser.add_argument("--config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--ckpt", default=None, help="Alias of --checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--device", default=None, choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--source_csv", default=None)
    parser.add_argument("--duration_sec", type=float, default=0.0)
    parser.add_argument("--standalone", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    checkpoint = args.checkpoint or args.ckpt

    forwarded = [
        "run_event_runtime.py",
        "--config",
        args.config,
        "--backend",
        "ckpt",
    ]
    if checkpoint:
        forwarded.extend(["--checkpoint", checkpoint])
    if args.port:
        forwarded.extend(["--port", args.port])
    if args.device:
        forwarded.extend(["--device", args.device])
    if args.source_csv:
        forwarded.extend(["--source_csv", args.source_csv])
    if args.duration_sec:
        forwarded.extend(["--duration_sec", str(args.duration_sec)])
    if args.standalone:
        forwarded.append("--standalone")

    logging.getLogger("event_runtime_ckpt").warning(
        "Legacy realtime_ckpt entry now forwards to scripts/run_event_runtime.py --backend ckpt."
    )
    sys.argv = forwarded
    run_event_runtime_main()


if __name__ == "__main__":
    main()
