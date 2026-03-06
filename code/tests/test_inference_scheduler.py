"""Unit tests for inference rate scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.inference.scheduler import InferenceRateScheduler


def test_scheduler_unlimited_mode_is_compatible():
    scheduler = InferenceRateScheduler(infer_rate_hz=0)
    # Unlimited mode should always allow inference.
    assert scheduler.should_run(now=1.0)
    assert scheduler.should_run(now=1.00001)
    assert scheduler.should_run(now=1.00002)


def test_scheduler_rate_limit_blocks_until_interval():
    scheduler = InferenceRateScheduler(infer_rate_hz=20.0)  # 50ms interval
    assert scheduler.should_run(now=1.0)
    assert not scheduler.should_run(now=1.02)
    assert not scheduler.should_run(now=1.049)
    assert scheduler.should_run(now=1.051)
