from __future__ import annotations

import pytest

from scripts.tune_event_runtime_thresholds import (
    _parse_float_tokens,
    _parse_int_tokens,
    _rank_key,
)


def test_parse_float_tokens() -> None:
    values = _parse_float_tokens("0.82, 0.86,0.90", name="confidence_thresholds")
    assert values == [0.82, 0.86, 0.9]


def test_parse_int_tokens() -> None:
    values = _parse_int_tokens("3,5", name="vote_windows")
    assert values == [3, 5]


def test_parse_tokens_reject_empty() -> None:
    with pytest.raises(ValueError):
        _parse_float_tokens(" , ", name="confidence_thresholds")
    with pytest.raises(ValueError):
        _parse_int_tokens(" , ", name="vote_windows")


def test_rank_key_prefers_high_success_low_error() -> None:
    row_a = {"command_success_rate": 0.80, "false_trigger_rate": 0.08, "false_release_rate": 0.10}
    row_b = {"command_success_rate": 0.78, "false_trigger_rate": 0.00, "false_release_rate": 0.00}
    ranked = sorted([row_a, row_b], key=_rank_key, reverse=True)
    assert ranked[0] is row_a
