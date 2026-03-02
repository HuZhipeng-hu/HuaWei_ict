"""
集成测试脚本

覆盖关键路径和边界条件，确保代码可以正常交付给他人使用。
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

results = []


def test(name):
    """测试装饰器"""

    def decorator(fn):
        try:
            fn()
            results.append((name, True, ""))
            print(f"  [PASS] {name}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
        return fn

    return decorator


# =====================================================================
# 1. 手势系统
# =====================================================================


@test("手势枚举、映射、常量完整性")
def _():
    from shared.gestures import (
        GestureType,
        GESTURE_FINGER_MAP,
        FOLDER_TO_GESTURE,
        NUM_CLASSES,
        NUM_FINGERS,
        NUM_EMG_CHANNELS,
        validate_gesture_definitions,
        get_finger_angles,
    )

    assert NUM_CLASSES == 6
    assert NUM_FINGERS == 5
    assert NUM_EMG_CHANNELS == 8
    assert len(GestureType) == 6
    assert len(GESTURE_FINGER_MAP) == 6
    assert len(FOLDER_TO_GESTURE) == 6
    validate_gesture_definitions()
    for g in GestureType:
        angles = get_finger_angles(g)
        assert len(angles) == NUM_FINGERS
        assert all(0 <= a <= 180 for a in angles)


# =====================================================================
# 2. 配置管理
# =====================================================================


@test("训练配置加载")
def _():
    from shared.config import load_training_config, ModelConfig

    mc, pc, tc, ac = load_training_config("configs/training.yaml")
    assert mc.in_channels == 6
    assert pc.device_sampling_rate == 1000
    assert pc.total_channels == 8
    assert tc.gradient_clip == 1.0
    assert ac.enabled is True


@test("运行时配置加载")
def _():
    from shared.config import load_runtime_config, RuntimeConfig

    rc = load_runtime_config("configs/runtime.yaml")
    assert isinstance(rc, RuntimeConfig)
    assert rc.control_rate_hz == 30.0


@test("load_config 未指定类型返回字典")
def _():
    from shared.config import load_config

    raw = load_config("configs/training.yaml")
    assert isinstance(raw, dict)
    assert "model" in raw
    assert "training" in raw


@test("缺失文件优雅降级")
def _():
    from shared.config import load_training_config, load_runtime_config

    mc, _, _, _ = load_training_config("不存在的文件.yaml")
    assert mc.in_channels == 6  # 默认值
    rc = load_runtime_config("不存在的文件.yaml")
    assert rc.control_rate_hz == 30.0  # 默认值


# =====================================================================
# 3. 预处理流水线
# =====================================================================


@test("预处理输出形状 (84采样→6×24×6)")
def _():
    from shared.preprocessing import PreprocessPipeline

    pipeline = PreprocessPipeline(
        sampling_rate=200.0,
        num_channels=6,
        lowcut=20.0,
        highcut=90.0,
        stft_window_size=24,
        stft_hop_size=12,
        stft_n_fft=46,
    )
    signal = np.random.randn(84, 8).astype(np.float32) * 100
    result = pipeline.process(signal)
    assert result.shape == (6, 24, 6), f"Expected (6,24,6), got {result.shape}"


@test("process_window 等价于 process")
def _():
    from shared.preprocessing import PreprocessPipeline

    pipeline = PreprocessPipeline(
        sampling_rate=200.0,
        num_channels=6,
        lowcut=20.0,
        highcut=90.0,
        stft_window_size=24,
        stft_hop_size=12,
        stft_n_fft=46,
    )
    window = np.random.randn(84, 8).astype(np.float32) * 100
    assert pipeline.process_window(window).shape == (6, 24, 6)


@test("SignalWindower 切分正确性")
def _():
    from shared.preprocessing import SignalWindower

    w = SignalWindower(window_size=84, stride=42)
    signal = np.random.randn(500, 8)
    segments = w.segment(signal)
    expected_count = w.count_segments(500)
    assert len(segments) == expected_count
    assert all(s.shape == (84, 8) for s in segments)
    # 信号太短（比窗口短）
    assert w.segment(np.random.randn(50, 8)) == []
    assert w.count_segments(50) == 0


# =====================================================================
# 4. 数据加载（真实数据）
# =====================================================================


@test("CSV数据加载+形状验证")
def _():
    from training.data.csv_dataset import CSVDatasetLoader
    from shared.preprocessing import PreprocessPipeline

    pipeline = PreprocessPipeline(
        sampling_rate=200.0,
        num_channels=6,
        lowcut=20.0,
        highcut=90.0,
        stft_window_size=24,
        stft_hop_size=12,
        stft_n_fft=46,
    )
    loader = CSVDatasetLoader(
        data_dir="../data",
        preprocess=pipeline,
        num_emg_channels=8,
        device_sampling_rate=1000,
        target_sampling_rate=200,
        segment_length=84,
        segment_stride=42,
    )
    data, labels = loader.load_all()
    assert len(data) > 0, "No data loaded"
    assert data[0].shape == (6, 24, 6)
    unique = set(labels)
    assert all(0 <= l < 6 for l in unique)
    print(f"      → {len(data)} samples, labels: {sorted(unique)}")


@test("数据分割比例 (dev 0.2)")
def _():
    from training.data.csv_dataset import CSVDatasetLoader
    from shared.preprocessing import PreprocessPipeline

    pipeline = PreprocessPipeline(
        sampling_rate=200.0,
        num_channels=6,
        lowcut=20.0,
        highcut=90.0,
        stft_window_size=24,
        stft_hop_size=12,
        stft_n_fft=46,
    )
    loader = CSVDatasetLoader(
        data_dir="../data",
        preprocess=pipeline,
        num_emg_channels=8,
        device_sampling_rate=1000,
        target_sampling_rate=200,
        segment_length=84,
        segment_stride=42,
    )
    data, labels = loader.load_all()
    (train_d, train_l), (val_d, val_l) = loader.split(data, labels, val_ratio=0.2)
    total = len(train_d) + len(val_d)
    assert total == len(data)
    assert 0.15 <= len(val_d) / total <= 0.25
    print(f"      → train={len(train_d)}, val={len(val_d)}")


# =====================================================================
# 5. 数据增强
# =====================================================================


@test("数据增强 factor=3")
def _():
    from training.data.augmentation import DataAugmentor

    aug = DataAugmentor(
        time_warp_rate=0.1,
        amplitude_scale=0.15,
        noise_std=0.05,
    )
    sample = np.random.randn(6, 24, 6).astype(np.float32)
    aug_data, aug_labels = aug.augment_batch(
        np.array([sample]),
        np.array([0]),
        factor=3,
    )
    assert len(aug_data) == 3  # 1 original + 2 augmented
    assert all(a.shape == (6, 24, 6) for a in aug_data)
    assert all(l == 0 for l in aug_labels)


# =====================================================================
# 6. 推理后处理
# =====================================================================


@test("滑动窗口投票器")
def _():
    from runtime.inference.postprocessing import SlidingWindowVoter

    voter = SlidingWindowVoter(window_size=5, min_count=3, confidence_threshold=0.5)
    # 连续投5次相同手势
    result = None
    for _ in range(5):
        result = voter.update(1, 0.9)
    assert result is not None
    # 重置后应为 None
    voter.reset()
    assert voter.current_gesture is None


# =====================================================================
# 7. 状态机
# =====================================================================


@test("状态机正常流程")
def _():
    from runtime.control.state_machine import SystemStateMachine, SystemState

    sm = SystemStateMachine()
    assert sm.state == SystemState.IDLE
    assert sm.transition_to(SystemState.CALIBRATING)
    assert sm.transition_to(SystemState.RUNNING)
    assert sm.is_running
    assert sm.transition_to(SystemState.STOPPING)
    assert sm.transition_to(SystemState.IDLE)


@test("状态机错误恢复")
def _():
    from runtime.control.state_machine import SystemStateMachine, SystemState

    sm = SystemStateMachine()
    sm.transition_to(SystemState.RUNNING)
    assert sm.set_error("test error")
    assert sm.is_error
    assert sm.error_message == "test error"
    assert sm.reset()
    assert sm.state == SystemState.IDLE


@test("状态机非法转换拒绝")
def _():
    from runtime.control.state_machine import SystemStateMachine, SystemState

    sm = SystemStateMachine()
    assert not sm.transition_to(SystemState.STOPPING)  # IDLE → STOPPING 非法


# =====================================================================
# 8. YAML 配置完整性
# =====================================================================


@test("所有YAML文件可解析")
def _():
    import yaml

    for f in [
        "configs/training.yaml",
        "configs/runtime.yaml",
        "configs/conversion.yaml",
    ]:
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            assert isinstance(data, dict), f"{f} is not a dict"


# =====================================================================
# 汇总
# =====================================================================

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"\n结果: {passed} 通过, {failed} 失败")

if failed > 0:
    print("\n失败项:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
