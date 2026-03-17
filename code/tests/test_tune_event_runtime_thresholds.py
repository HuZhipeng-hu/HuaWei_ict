from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.tune_event_runtime_thresholds as tune

from scripts.tune_event_runtime_thresholds import (
    _parse_float_tokens,
    _parse_int_tokens,
    _rank_key,
)
from shared.gestures import GestureType


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


def test_evaluate_combo_excludes_release_command_from_action_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Decision:
        def __init__(self, state: GestureType, changed: bool = True) -> None:
            self.state = state
            self.changed = changed

    class _Step:
        def __init__(self, decision: _Decision) -> None:
            self.decision = decision

    scenarios: list[tuple[list[GestureType], GestureType]] = [
        ([GestureType.RELAX], GestureType.RELAX),  # release command clip success
        ([GestureType.THUMB_UP], GestureType.THUMB_UP),  # normal action clip success
    ]

    class _FakeController:
        def __init__(self, **_kwargs) -> None:
            transitions, final_state = scenarios.pop(0)
            self.state_machine = SimpleNamespace(current_label=0, current_state=GestureType.RELAX)
            self.current_state = final_state
            self._transitions = transitions

        def ingest_rows(self, _matrix: np.ndarray) -> list[_Step]:
            return [_Step(_Decision(state=item, changed=True)) for item in self._transitions]

    monkeypatch.setattr(tune, "EventOnsetController", _FakeController)

    metrics = tune._evaluate_combo(
        clips=[
            (2, 1, np.zeros((16, 14), dtype=np.float32)),  # THUMB_UP -> TENSE_OPEN(release)
            (0, 2, np.zeros((16, 14), dtype=np.float32)),  # RELAX -> THUMB_UP
        ],
        class_names=["RELAX", "TENSE_OPEN", "THUMB_UP"],
        label_to_state={0: GestureType.RELAX, 1: GestureType.RELAX, 2: GestureType.THUMB_UP},
        data_cfg=SimpleNamespace(),
        runtime_cfg=SimpleNamespace(
            inference=SimpleNamespace(
                confidence_threshold=0.8,
                gate_confidence_threshold=0.85,
                command_confidence_threshold=0.8,
                activation_margin_threshold=0.1,
                vote_window=3,
                vote_min_count=2,
                switch_confidence_boost=0.1,
            ),
            runtime=SimpleNamespace(),
        ),
        predict_proba=lambda *_args, **_kwargs: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        predict_detail=None,
        params={
            "confidence_threshold": 0.9,
            "gate_confidence_threshold": 0.88,
            "command_confidence_threshold": 0.84,
            "activation_margin_threshold": 0.2,
            "vote_window": 5,
            "vote_min_count": 3,
            "switch_confidence_boost": 0.12,
        },
    )
    assert metrics["total_clip_count"] == 2
    assert metrics["action_clip_count"] == 1
    assert metrics["continue_clip_count"] == 0
    assert metrics["release_command_clip_count"] == 1
    assert metrics["command_success_rate"] == 1.0
    assert metrics["false_release_rate"] == 0.0
