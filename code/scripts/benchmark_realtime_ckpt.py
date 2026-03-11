"""Compatibility wrapper to event runtime benchmark."""

from __future__ import annotations

import logging
import sys

from scripts.benchmark_event_runtime_ckpt import main as benchmark_event_main


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("benchmark_realtime_ckpt").warning(
        "Legacy benchmark_realtime_ckpt now forwards to benchmark_event_runtime_ckpt."
    )
    if "--backend" not in sys.argv:
        sys.argv.extend(["--backend", "both"])
    benchmark_event_main()


if __name__ == "__main__":
    main()
