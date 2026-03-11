import numpy as np

from runtime.hardware.factory import StandaloneSensor
from scripts.benchmark_event_runtime_ckpt import _build_mock_predictor, _compare_metrics, _merge_gate
from event_onset.config import load_event_runtime_config


def test_mock_predictor_outputs_probability_distribution():
    predictor = _build_mock_predictor(seed=42, num_classes=3)
    probs = predictor(np.zeros((8, 24, 3), dtype=np.float32), np.zeros((6, 16), dtype=np.float32))

    assert probs.shape == (3,)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0)


def test_compare_metrics_returns_expected_deltas():
    ckpt = {
        "transition_hit_rate": 0.80,
        "false_trigger_rate": 0.10,
        "state_hold_accuracy": 0.90,
        "release_accuracy": 0.88,
        "latency_p50_ms": 120.0,
        "latency_p95_ms": 220.0,
    }
    lite = {
        "transition_hit_rate": 0.78,
        "false_trigger_rate": 0.12,
        "state_hold_accuracy": 0.89,
        "release_accuracy": 0.90,
        "latency_p50_ms": 130.0,
        "latency_p95_ms": 260.0,
    }
    delta = _compare_metrics(ckpt, lite)

    assert delta["delta_transition_hit_rate"] == lite["transition_hit_rate"] - ckpt["transition_hit_rate"]
    assert delta["delta_false_trigger_rate"] == lite["false_trigger_rate"] - ckpt["false_trigger_rate"]
    assert delta["delta_latency_p95_ms"] == lite["latency_p95_ms"] - ckpt["latency_p95_ms"]
    gate = _merge_gate(ckpt, lite)
    assert gate["passed"] is True


def test_event_runtime_config_has_model_paths():
    runtime_cfg = load_event_runtime_config("configs/runtime_event_onset.yaml")
    assert runtime_cfg.model_path.endswith(".mindir")
    assert runtime_cfg.model_metadata_path.endswith(".json")


def test_standalone_sensor_uses_eight_channel_protocol():
    sensor = StandaloneSensor()
    assert sensor.connect() is True
    try:
        window = sensor.read_window(5)
    finally:
        sensor.disconnect()

    assert window is not None
    assert window.shape == (5, 8)
