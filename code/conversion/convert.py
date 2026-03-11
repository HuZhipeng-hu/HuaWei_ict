"""Compatibility wrapper to event-onset conversion entrypoint."""

from __future__ import annotations

import logging

from scripts.convert_event_onset import main as convert_event_onset_main


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("conversion")
    logger.warning(
        "Legacy 6-gesture conversion path has been removed. "
        "Forwarding to event-onset conversion."
    )
    convert_event_onset_main()


if __name__ == "__main__":
    main()
