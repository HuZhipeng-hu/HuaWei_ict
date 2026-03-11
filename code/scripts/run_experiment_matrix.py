"""Deprecated legacy experiment-matrix entrypoint."""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "run_experiment_matrix.py is removed for the event-onset branch. "
        "Use scripts/finetune_event_onset.py and scripts/benchmark_event_runtime_ckpt.py instead."
    )


if __name__ == "__main__":
    main()
