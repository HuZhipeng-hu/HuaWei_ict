"""Compatibility wrapper to event-onset runtime entrypoint."""

from __future__ import annotations

import logging

from scripts.run_event_runtime import main as run_event_runtime_main


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("runtime")
    logger.warning(
        "Legacy 6-gesture runtime path has been removed. "
        "Forwarding to event-onset runtime with default backend=lite."
    )
    run_event_runtime_main()


if __name__ == "__main__":
    main()
