from __future__ import annotations

from scripts.evaluate_event_demo_control import _compute_sanity_flags


def test_compute_sanity_flags_ok() -> None:
    flags = _compute_sanity_flags(command_success_rate=0.02, false_trigger_rate=0.98)
    assert flags["metric_invariant_ok"] is True


def test_compute_sanity_flags_detects_conflict() -> None:
    flags = _compute_sanity_flags(command_success_rate=0.10, false_trigger_rate=0.98)
    assert flags["metric_invariant_ok"] is False

