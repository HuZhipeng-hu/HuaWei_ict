from runtime.hardware.factory import StandaloneSensor
from shared.config import RuntimeConfig, load_runtime_config
from scripts.benchmark_realtime_ckpt import build_runtime_benchmark_plan, _iter_control_windows


def test_realtime_benchmark_plan_matches_dual_branch_shape():
    runtime_cfg = RuntimeConfig()
    plan = build_runtime_benchmark_plan(runtime_cfg)

    assert plan["base_window"] == 420
    assert plan["stride"] == 210
    assert plan["read_window_size"] > plan["base_window"]
    assert plan["expected_input_shape"] == (1, 16, 24, 6)


def test_iter_control_windows_respects_read_size_and_cycle_step():
    signal = __import__("numpy").random.randn(620, 8).astype("float32")
    windows = list(_iter_control_windows(signal, read_window_size=560, cycle_step_samples=20))

    assert len(windows) == 4
    assert all(window.shape == (560, 8) for _, window in windows)


def test_iter_control_windows_caps_read_size_for_short_recordings():
    signal = __import__("numpy").random.randn(540, 8).astype("float32")
    windows = list(_iter_control_windows(signal, read_window_size=559, cycle_step_samples=20, min_window_size=420))

    assert len(windows) == 1
    assert windows[0][1].shape == (540, 8)


def test_runtime_config_prefers_inference_infer_rate_hz(tmp_path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
inference:
  infer_rate_hz: 25
""".strip(),
        encoding="utf-8",
    )

    runtime_cfg = load_runtime_config(cfg_path)

    assert runtime_cfg.inference.infer_rate_hz == 25.0
    assert runtime_cfg.infer_rate_hz == 25.0


def test_runtime_config_accepts_legacy_root_infer_rate_hz(tmp_path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
infer_rate_hz: 15
""".strip(),
        encoding="utf-8",
    )

    runtime_cfg = load_runtime_config(cfg_path)

    assert runtime_cfg.inference.infer_rate_hz == 15.0
    assert runtime_cfg.infer_rate_hz == 15.0


def test_standalone_sensor_uses_eight_channel_protocol():
    sensor = StandaloneSensor()
    assert sensor.connect() is True
    try:
        window = sensor.read_window(5)
    finally:
        sensor.disconnect()

    assert window is not None
    assert window.shape == (5, 8)
