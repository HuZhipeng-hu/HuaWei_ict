"""Unit tests for inference rate scheduler."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.inference import InferenceRateScheduler, PredictionScheduler, SlidingWindowVoter, TemporalVoter


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


def test_prediction_scheduler_matches_legacy_timing():
    legacy = InferenceRateScheduler(infer_rate_hz=20.0)
    modern = PredictionScheduler(inference_interval_ms=50)

    timeline = [1.0, 1.02, 1.049, 1.051]
    legacy_results = [legacy.should_run(now=t) for t in timeline]
    modern_results = [modern.should_run(now=t) for t in timeline]

    assert modern_results == legacy_results


def test_voter_aliases_stabilize_same_gesture():
    legacy = SlidingWindowVoter(window_size=3, min_count=2, confidence_threshold=0.5)
    modern = TemporalVoter(history_window_ms=300, hysteresis_count=2)

    assert legacy.update(1, 0.9).value == 1
    assert modern.update(1, 0.9, now=1.0) is None
    assert legacy.update(1, 0.9).value == 1
    assert modern.update(1, 0.9, now=1.1) == 1
