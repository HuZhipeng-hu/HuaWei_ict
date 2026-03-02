"""
Unit tests for preprocessing pipeline.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.preprocessing.filters import BandpassFilter, normalize, rectify
from shared.preprocessing.stft import PreprocessPipeline, STFTProcessor


def test_bandpass_1d():
    bp = BandpassFilter(lowcut=20, highcut=90, sampling_rate=200)
    signal = np.random.randn(200).astype(np.float64)
    filtered = bp(signal)
    assert filtered.shape == signal.shape
    assert not np.isnan(filtered).any()


def test_bandpass_2d():
    bp = BandpassFilter(lowcut=20, highcut=90, sampling_rate=200)
    signal = np.random.randn(200, 6).astype(np.float64)
    filtered = bp(signal)
    assert filtered.shape == (200, 6)
    assert not np.isnan(filtered).any()


def test_bandpass_short_signal():
    bp = BandpassFilter(lowcut=20, highcut=90, sampling_rate=200)
    signal = np.random.randn(10).astype(np.float64)
    filtered = bp(signal)
    assert filtered.shape == signal.shape


def test_rectify():
    signal = np.array([-1.0, 0.5, -0.3, 2.0])
    result = rectify(signal)
    expected = np.array([1.0, 0.5, 0.3, 2.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_normalize():
    signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    normalized, _, _ = normalize(signal)
    assert np.abs(normalized.mean(axis=0)).max() < 1e-6
    assert np.abs(normalized.std(axis=0) - 1.0).max() < 1e-6


def test_normalize_with_stats():
    signal = np.array([[10.0, 20.0], [30.0, 40.0]])
    mean = np.array([20.0, 30.0])
    std = np.array([10.0, 10.0])
    normalized, _, _ = normalize(signal, mean=mean, std=std)
    expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_array_almost_equal(normalized, expected)


def test_stft_output_shape():
    stft = STFTProcessor(window_size=24, hop_size=12, n_fft=46)
    signal = np.random.randn(168).astype(np.float64)
    spec = stft(signal)
    freq_bins, time_frames = stft.compute_output_shape(168)
    assert spec.shape[0] == freq_bins
    assert spec.shape[1] == time_frames


def test_stft_short_signal():
    stft = STFTProcessor(window_size=24, hop_size=12, n_fft=46)
    signal = np.random.randn(10).astype(np.float64)
    spec = stft(signal)
    assert spec.shape[0] == 24  # n_fft // 2 + 1


def test_pipeline_output_shape():
    pipeline = PreprocessPipeline(sampling_rate=200.0, num_channels=6)
    signal = np.random.randn(168, 8).astype(np.float64)
    spec = pipeline.process(signal)
    assert spec.ndim == 3
    assert spec.shape[0] == 6
    assert spec.dtype == np.float32


def test_pipeline_process_window():
    pipeline = PreprocessPipeline(sampling_rate=200.0, num_channels=6)
    window = np.random.randn(168, 6).astype(np.float64)
    spec1 = pipeline.process(window)
    spec2 = pipeline.process_window(window)
    np.testing.assert_array_equal(spec1, spec2)


if __name__ == "__main__":
    tests = [
        test_bandpass_1d,
        test_bandpass_2d,
        test_bandpass_short_signal,
        test_rectify,
        test_normalize,
        test_normalize_with_stats,
        test_stft_output_shape,
        test_stft_short_signal,
        test_pipeline_output_shape,
        test_pipeline_process_window,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {test.__name__}: {exc}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
