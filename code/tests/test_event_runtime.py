from event_onset.config import EventInferenceConfig, EventRuntimeBehaviorConfig
from event_onset.runtime import EventRuntimeStateMachine
from shared.gestures import GestureType


def _build_machine() -> EventRuntimeStateMachine:
    class_names = ["RELAX", "E1_G01", "E1_G02", "E2_G05"]
    label_to_state = {
        0: GestureType.RELAX,
        1: GestureType.FIST,
        2: GestureType.PINCH,
        3: GestureType.OK,
    }
    return EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(
            confidence_threshold=0.75,
            per_class_confidence_thresholds={
                "E1_G01": 0.75,
                "E1_G02": 0.82,
                "E2_G05": 0.80,
            },
            vote_window=3,
            vote_min_count=2,
            activation_margin_threshold=0.08,
            switch_confidence_boost=0.06,
            switch_margin_threshold=0.12,
        ),
        runtime_config=EventRuntimeBehaviorConfig(
            idle_release_hold_ms=700,
            min_transition_gap_ms=120,
            post_transition_lock_ms=220,
            low_energy_release_threshold=2.5,
        ),
        low_energy_threshold=2.5,
        class_names=class_names,
        label_to_state=label_to_state,
    )


def test_event_runtime_switches_and_holds_until_idle_timeout():
    machine = _build_machine()

    machine.update(1, 0.90, 8.0, 0.0)
    decision = machine.update(1, 0.92, 8.0, 40.0)
    assert decision.state == GestureType.FIST
    assert decision.changed is True
    assert decision.emitted_class_name == "E1_G01"

    idle = machine.update(0, 0.20, 0.5, 200.0)
    assert idle.state == GestureType.FIST
    assert idle.changed is False

    released = machine.update(0, 0.10, 0.2, 950.0)
    assert released.state == GestureType.RELAX
    assert released.changed is True
    assert released.emitted_class_name == "CONTINUE"


def test_event_runtime_supports_direct_action_to_action_switch():
    machine = _build_machine()

    machine.update(1, 0.90, 8.0, 0.0, current_state_confidence=0.1, top2_confidence=0.1)
    machine.update(1, 0.91, 8.0, 40.0, current_state_confidence=0.1, top2_confidence=0.1)
    assert machine.current_state == GestureType.FIST

    machine.update(2, 0.95, 8.2, 300.0, current_state_confidence=0.6, top2_confidence=0.6)
    switched = machine.update(2, 0.96, 8.4, 340.0, current_state_confidence=0.6, top2_confidence=0.6)
    assert switched.changed is True
    assert switched.state == GestureType.PINCH
    assert switched.emitted_class_name == "E1_G02"


def test_event_runtime_rejects_single_noise_pulse():
    machine = _build_machine()

    pulse = machine.update(3, 0.91, 9.0, 0.0)
    assert pulse.changed is False
    assert machine.current_state == GestureType.RELAX

    reset = machine.update(0, 0.2, 0.4, 40.0)
    assert reset.state == GestureType.RELAX
    assert reset.changed is False


def test_event_runtime_rejects_weak_cross_class_switch_signal():
    machine = _build_machine()

    machine.update(1, 0.88, 8.0, 0.0, current_state_confidence=0.1, top2_confidence=0.1)
    machine.update(1, 0.89, 8.0, 40.0, current_state_confidence=0.1, top2_confidence=0.1)
    assert machine.current_state == GestureType.FIST

    weak_switch = machine.update(2, 0.84, 8.0, 300.0, current_state_confidence=0.76, top2_confidence=0.76)
    assert weak_switch.changed is False
    assert machine.current_state == GestureType.FIST


def test_event_runtime_supports_wrist_direction_actions():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(
            confidence_threshold=0.75,
            per_class_confidence_thresholds={"WRIST_CW": 0.8, "WRIST_CCW": 0.8},
            vote_window=2,
            vote_min_count=2,
            activation_margin_threshold=0.08,
            switch_confidence_boost=0.06,
            switch_margin_threshold=0.12,
        ),
        runtime_config=EventRuntimeBehaviorConfig(
            idle_release_hold_ms=700,
            min_transition_gap_ms=120,
            post_transition_lock_ms=220,
            low_energy_release_threshold=2.5,
        ),
        low_energy_threshold=2.5,
        class_names=["RELAX", "WRIST_CW", "WRIST_CCW"],
        label_to_state={0: GestureType.RELAX, 1: GestureType.WRIST_CW, 2: GestureType.WRIST_CCW},
    )

    machine.update(1, 0.91, 8.0, 0.0, current_state_confidence=0.1, top2_confidence=0.1)
    cw = machine.update(1, 0.92, 8.0, 40.0, current_state_confidence=0.1, top2_confidence=0.1)
    assert cw.changed is True
    assert cw.state == GestureType.WRIST_CW
    assert cw.emitted_class_name == "WRIST_CW"

    machine.update(2, 0.95, 8.0, 300.0, current_state_confidence=0.7, top2_confidence=0.7)
    ccw = machine.update(2, 0.96, 8.0, 340.0, current_state_confidence=0.7, top2_confidence=0.7)
    assert ccw.changed is True
    assert ccw.state == GestureType.WRIST_CCW
    assert ccw.emitted_class_name == "WRIST_CCW"


def test_event_runtime_latch_and_tense_open_release_without_idle_auto_fallback():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(
            confidence_threshold=0.75,
            per_class_confidence_thresholds={"TENSE_OPEN": 0.80, "THUMB_UP": 0.82},
            vote_window=2,
            vote_min_count=2,
            activation_margin_threshold=0.08,
            switch_confidence_boost=0.06,
            switch_margin_threshold=0.12,
        ),
        runtime_config=EventRuntimeBehaviorConfig(
            idle_release_hold_ms=10_000_000,
            min_transition_gap_ms=120,
            post_transition_lock_ms=220,
            low_energy_release_threshold=2.5,
            release_mode="command_only",
        ),
        low_energy_threshold=2.5,
        class_names=["RELAX", "TENSE_OPEN", "THUMB_UP"],
        label_to_state={0: GestureType.RELAX, 1: GestureType.RELAX, 2: GestureType.THUMB_UP},
    )

    machine.update(2, 0.95, 8.0, 0.0, current_state_confidence=0.10, top2_confidence=0.10)
    thumb = machine.update(2, 0.96, 8.1, 40.0, current_state_confidence=0.10, top2_confidence=0.10)
    assert thumb.changed is True
    assert thumb.state == GestureType.THUMB_UP

    idle = machine.update(0, 0.20, 0.2, 2_000.0)
    assert idle.changed is False
    assert machine.current_state == GestureType.THUMB_UP

    machine.update(1, 0.92, 8.2, 2_400.0, current_state_confidence=0.60, top2_confidence=0.60)
    released = machine.update(1, 0.93, 8.2, 2_450.0, current_state_confidence=0.60, top2_confidence=0.60)
    assert released.changed is True
    assert released.state == GestureType.RELAX
    assert released.emitted_class_name == "TENSE_OPEN"


def test_event_runtime_two_stage_gate_and_command_thresholds():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(
            confidence_threshold=0.75,
            gate_confidence_threshold=0.85,
            command_confidence_threshold=0.80,
            per_class_confidence_thresholds={"THUMB_UP": 0.78},
            vote_window=2,
            vote_min_count=2,
            activation_margin_threshold=0.08,
            switch_confidence_boost=0.06,
            switch_margin_threshold=0.12,
        ),
        runtime_config=EventRuntimeBehaviorConfig(
            idle_release_hold_ms=10_000_000,
            min_transition_gap_ms=120,
            post_transition_lock_ms=220,
            low_energy_release_threshold=2.5,
            release_mode="command_only",
        ),
        low_energy_threshold=2.5,
        class_names=["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW"],
        label_to_state={
            0: GestureType.RELAX,
            1: GestureType.RELAX,
            2: GestureType.THUMB_UP,
            3: GestureType.WRIST_CW,
        },
    )

    blocked = machine.update(
        2,
        0.92,
        8.0,
        0.0,
        gate_confidence=0.82,
        command_confidence=0.92,
        top2_confidence=0.10,
    )
    assert blocked.changed is False
    assert machine.current_state == GestureType.RELAX

    machine.update(
        2,
        0.91,
        8.1,
        40.0,
        gate_confidence=0.90,
        command_confidence=0.79,
        top2_confidence=0.10,
    )
    weak_command = machine.update(
        2,
        0.91,
        8.1,
        80.0,
        gate_confidence=0.90,
        command_confidence=0.79,
        top2_confidence=0.10,
    )
    assert weak_command.changed is False
    assert machine.current_state == GestureType.RELAX

    machine.update(
        2,
        0.93,
        8.2,
        240.0,
        gate_confidence=0.91,
        command_confidence=0.93,
        top2_confidence=0.20,
    )
    accepted = machine.update(
        2,
        0.94,
        8.3,
        280.0,
        gate_confidence=0.92,
        command_confidence=0.94,
        top2_confidence=0.20,
    )
    assert accepted.changed is True
    assert accepted.state == GestureType.THUMB_UP
