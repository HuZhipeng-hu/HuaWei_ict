from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.evaluate_event_demo_control import _evaluate_control_metrics
from scripts.evaluate_event_demo_control import _compute_sanity_flags
from shared.gestures import GestureType


def test_compute_sanity_flags_ok() -> None:
    flags = _compute_sanity_flags(command_success_rate=0.02, false_trigger_rate=0.98)
    assert flags["metric_invariant_ok"] is True


def test_compute_sanity_flags_detects_conflict() -> None:
    flags = _compute_sanity_flags(command_success_rate=0.10, false_trigger_rate=0.98)
    assert flags["metric_invariant_ok"] is False


def test_control_metrics_treats_release_command_as_non_action() -> None:
    class _Decision:
        def __init__(self, state: GestureType, changed: bool = True) -> None:
            self.state = state
            self.changed = changed

    class _Step:
        def __init__(self, decision: _Decision) -> None:
            self.decision = decision

    class _Controller:
        def __init__(self, transitions: list[GestureType], final_state: GestureType) -> None:
            self.state_machine = SimpleNamespace(current_label=0, current_state=GestureType.RELAX)
            self.current_state = final_state
            self._transitions = transitions

        def ingest_rows(self, _matrix: np.ndarray) -> list[_Step]:
            return [_Step(_Decision(state=item, changed=True)) for item in self._transitions]

    class _Loader:
        def iter_clips(self):
            rows = np.zeros((32, 14), dtype=np.float32)
            # release command clip: THUMB_UP -> TENSE_OPEN(->RELAX)
            yield "THUMB_UP", "TENSE_OPEN", rows, {"relative_path": "clip_release"}
            # normal action clip
            yield "RELAX", "THUMB_UP", rows, {"relative_path": "clip_action"}
            # pure relax clip
            yield "RELAX", "RELAX", rows, {"relative_path": "clip_relax"}

    controllers = iter(
        [
            _Controller([GestureType.RELAX], GestureType.RELAX),
            _Controller([GestureType.THUMB_UP], GestureType.THUMB_UP),
            _Controller([], GestureType.RELAX),
        ]
    )

    def _controller_factory():
        return next(controllers)

    metrics = _evaluate_control_metrics(
        controller_factory=_controller_factory,
        loader=_Loader(),
        test_sources={"clip_release", "clip_action", "clip_relax"},
        class_names=["RELAX", "TENSE_OPEN", "THUMB_UP"],
        label_to_state={
            0: GestureType.RELAX,
            1: GestureType.RELAX,  # release command label
            2: GestureType.THUMB_UP,
        },
    )
    assert metrics["total_clip_count"] == 3
    assert metrics["action_clip_count"] == 1
    assert metrics["release_command_clip_count"] == 1
    assert metrics["relax_clip_count"] == 1
    assert metrics["command_success_rate"] == 1.0
    assert metrics["false_release_rate"] == 0.0
    assert metrics["false_trigger_rate"] == 0.0
