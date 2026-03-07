"""Integration-style smoke tests for core wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_gesture_definitions_are_consistent():
    from shared.gestures import NUM_CLASSES, GestureType, validate_gesture_definitions

    assert NUM_CLASSES == 6
    assert len(GestureType) == 6
    assert validate_gesture_definitions() is True


def test_config_loading_smoke():
    from shared.config import RuntimeConfig, load_runtime_config, load_training_config

    mc, pc, tc, ac = load_training_config("configs/training.yaml")
    assert mc.in_channels == 12
    assert pc.device_sampling_rate == 1000
    assert pc.dual_branch.enabled is True
    assert tc.split_mode in {"legacy", "grouped_file"}
    assert 0.0 <= tc.test_ratio < 1.0
    assert ac.enabled in {True, False}

    rc = load_runtime_config("configs/runtime.yaml")
    assert isinstance(rc, RuntimeConfig)
    assert rc.control_rate_hz > 0
    assert rc.infer_rate_hz >= 0


def test_all_yaml_configs_parse():
    import yaml

    for config_name in ("training.yaml", "runtime.yaml", "conversion.yaml"):
        path = Path("configs") / config_name
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)


def test_preprocess_pipeline_shape_smoke():
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
    assert result.shape == (6, 24, 6)


def test_csv_loader_real_data_smoke():
    from shared.preprocessing import PreprocessPipeline
    from training.data.csv_dataset import CSVDatasetLoader

    data_dir = Path(__file__).resolve().parents[2] / "data"
    if not data_dir.exists():
        # Skip gracefully if repository snapshot has no dataset.
        return

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
        data_dir=str(data_dir),
        preprocess=pipeline,
        num_emg_channels=8,
        device_sampling_rate=1000,
        target_sampling_rate=200,
        segment_length=84,
        segment_stride=42,
    )

    samples, labels, source_ids = loader.load_all_with_sources()
    assert len(samples) == len(labels) == len(source_ids)
    assert samples[0].shape == (6, 24, 6)


def test_dual_branch_shape_consistency_between_training_runtime_conversion():
    import yaml

    from shared.config import load_runtime_config, load_training_config
    from shared.preprocessing import PreprocessPipeline

    _, train_pp, _, _ = load_training_config("configs/training.yaml")
    runtime_cfg = load_runtime_config("configs/runtime.yaml")

    train_pipeline = PreprocessPipeline(
        sampling_rate=train_pp.sampling_rate,
        num_channels=train_pp.num_channels,
        lowcut=train_pp.lowcut,
        highcut=train_pp.highcut,
        filter_order=train_pp.filter_order,
        stft_window_size=train_pp.stft_window_size,
        stft_hop_size=train_pp.stft_hop_size,
        stft_n_fft=train_pp.stft_n_fft,
        device_sampling_rate=train_pp.device_sampling_rate,
        segment_length=train_pp.segment_length,
        segment_stride=train_pp.segment_stride,
        dual_branch=train_pp.dual_branch,
    )
    runtime_pp = runtime_cfg.preprocess
    runtime_pipeline = PreprocessPipeline(
        sampling_rate=runtime_pp.sampling_rate,
        num_channels=runtime_pp.num_channels,
        lowcut=runtime_pp.lowcut,
        highcut=runtime_pp.highcut,
        filter_order=runtime_pp.filter_order,
        stft_window_size=runtime_pp.stft_window_size,
        stft_hop_size=runtime_pp.stft_hop_size,
        stft_n_fft=runtime_pp.stft_n_fft,
        device_sampling_rate=runtime_pp.device_sampling_rate,
        segment_length=runtime_pp.segment_length,
        segment_stride=runtime_pp.segment_stride,
        dual_branch=runtime_pp.dual_branch,
    )

    train_shape = (1, *train_pipeline.get_output_shape())
    runtime_shape = (1, *runtime_pipeline.get_output_shape())
    assert train_shape == runtime_shape

    with open(Path("configs/conversion.yaml"), "r", encoding="utf-8") as f:
        conversion_cfg = yaml.safe_load(f)
    conversion_shape = tuple(int(x) for x in conversion_cfg["export"]["input_shape"])
    assert conversion_shape == train_shape


def test_voter_smoke():
    from runtime.inference.postprocessing import SlidingWindowVoter

    voter = SlidingWindowVoter(window_size=5, min_count=3, confidence_threshold=0.5)
    output = None
    for _ in range(5):
        output = voter.update(1, 0.9)
    assert output is not None
    voter.reset()
    assert voter.current_gesture is None
