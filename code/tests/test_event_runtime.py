from event_onset.config import EventInferenceConfig, EventRuntimeBehaviorConfig
from event_onset.runtime import EventRuntimeStateMachine
from shared.gestures import GestureType


def test_event_runtime_switches_and_holds_until_idle_timeout():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(confidence_threshold=0.75, vote_window=3, vote_min_count=2),
        runtime_config=EventRuntimeBehaviorConfig(idle_release_hold_ms=700, min_transition_gap_ms=120, low_energy_release_threshold=2.5),
        low_energy_threshold=2.5,
    )

    machine.update(1, 0.9, 8.0, 0.0)
    decision = machine.update(1, 0.92, 8.0, 40.0)
    assert decision.state == GestureType.FIST
    assert decision.changed is True

    idle = machine.update(0, 0.2, 0.5, 200.0)
    assert idle.state == GestureType.FIST
    assert idle.changed is False

    released = machine.update(0, 0.1, 0.2, 950.0)
    assert released.state == GestureType.RELAX
    assert released.changed is True


def test_event_runtime_supports_direct_fist_to_pinch_switch():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(
            confidence_threshold=0.75,
            fist_confidence_threshold=0.75,
            pinch_confidence_threshold=0.82,
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
    )

    machine.update(1, 0.9, 8.0, 0.0, current_state_confidence=0.1, top2_confidence=0.1)
    machine.update(1, 0.9, 8.0, 40.0, current_state_confidence=0.1, top2_confidence=0.1)
    assert machine.current_state == GestureType.FIST

    machine.update(2, 0.95, 8.2, 300.0, current_state_confidence=0.6, top2_confidence=0.6)
    switched = machine.update(2, 0.96, 8.4, 340.0, current_state_confidence=0.6, top2_confidence=0.6)
    assert switched.changed is True
    assert switched.state == GestureType.PINCH


def test_event_runtime_rejects_single_noise_pulse():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(confidence_threshold=0.75, vote_window=3, vote_min_count=2),
        runtime_config=EventRuntimeBehaviorConfig(idle_release_hold_ms=700, min_transition_gap_ms=120, low_energy_release_threshold=2.5),
        low_energy_threshold=2.5,
    )

    pulse = machine.update(1, 0.91, 9.0, 0.0)
    assert pulse.changed is False
    assert machine.current_state == GestureType.RELAX

    reset = machine.update(0, 0.2, 0.4, 40.0)
    assert reset.state == GestureType.RELAX
    assert reset.changed is False


def test_event_runtime_rejects_weak_cross_class_switch_signal():
    machine = EventRuntimeStateMachine(
        inference_config=EventInferenceConfig(
            confidence_threshold=0.75,
            fist_confidence_threshold=0.75,
            pinch_confidence_threshold=0.82,
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
    )

    machine.update(1, 0.88, 8.0, 0.0, current_state_confidence=0.1, top2_confidence=0.1)
    machine.update(1, 0.89, 8.0, 40.0, current_state_confidence=0.1, top2_confidence=0.1)
    assert machine.current_state == GestureType.FIST

    weak_switch = machine.update(2, 0.84, 8.0, 300.0, current_state_confidence=0.76, top2_confidence=0.76)
    assert weak_switch.changed is False
    assert machine.current_state == GestureType.FIST
