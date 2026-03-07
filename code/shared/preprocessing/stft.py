"""
STFT transform and preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .filters import BandpassFilter, normalize, rectify


class SignalWindower:
    def __init__(self, window_size: int, stride: int):
        self.window_size = window_size
        self.stride = stride

    def segment(self, signal: np.ndarray) -> List[np.ndarray]:
        total = signal.shape[0]
        segments = []
        start = 0
        while start + self.window_size <= total:
            segments.append(signal[start : start + self.window_size])
            start += self.stride
        return segments

    def count_segments(self, signal_length: int) -> int:
        if signal_length < self.window_size:
            return 0
        return (signal_length - self.window_size) // self.stride + 1


class STFTProcessor:
    def __init__(self, window_size: int = 24, hop_size: int = 12, n_fft: int = 46):
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        num_frames = max(0, (len(signal) - self.window_size) // self.hop_size + 1)
        if num_frames == 0:
            padded = np.zeros(self.window_size)
            padded[: len(signal)] = signal
            frame = padded * np.hanning(self.window_size)
            spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))
            return spectrum.reshape(-1, 1)

        frames = []
        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size
            frame = signal[start : start + self.window_size]
            frame = frame * np.hanning(self.window_size)
            spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))
            frames.append(spectrum)
        return np.array(frames).T

    def compute_output_shape(self, signal_length: int) -> Tuple[int, int]:
        num_frames = max(1, (signal_length - self.window_size) // self.hop_size + 1)
        return (self.freq_bins, num_frames)


def _default_dual_branch_cfg() -> Dict:
    return {
        "enabled": True,
        "fuse_mode": "concat_channels",
        "low_rate": 200,
        "high_rate": 1000,
        "high_segment_length": 420,
        "high_segment_stride": 210,
        "high_stft_window_size": 120,
        "high_stft_hop_size": 60,
        "high_stft_n_fft": 230,
        "high_freq_bins_out": 24,
        "multi_phase_offsets": [0.0, 0.33, 0.66],
    }


class PreprocessPipeline:
    def __init__(
        self,
        sampling_rate: float = 200.0,
        num_channels: int = 6,
        lowcut: float = 20.0,
        highcut: float = 90.0,
        filter_order: int = 4,
        stft_window_size: int = 24,
        stft_hop_size: int = 12,
        stft_n_fft: int = 46,
        device_sampling_rate: int = 1000,
        segment_length: int = 84,
        segment_stride: int = 42,
        dual_branch: Optional[object] = None,
    ):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.device_sampling_rate = int(device_sampling_rate)
        self.segment_length = int(segment_length)
        self.segment_stride = int(segment_stride)

        if dual_branch is None:
            dual_dict = {"enabled": False}
        elif is_dataclass(dual_branch):
            dual_dict = asdict(dual_branch)
        elif isinstance(dual_branch, dict):
            dual_dict = dict(dual_branch)
        else:
            raise TypeError(f"Unsupported dual_branch type: {type(dual_branch)!r}")

        merged = _default_dual_branch_cfg()
        merged.update(dual_dict)
        self.dual_branch = merged
        self.dual_branch_enabled = bool(self.dual_branch.get("enabled", False))
        self.fuse_mode = str(self.dual_branch.get("fuse_mode", "concat_channels"))
        if self.fuse_mode != "concat_channels":
            raise ValueError(f"Unsupported dual-branch fuse_mode={self.fuse_mode!r}")

        self.bandpass = BandpassFilter(
            lowcut=lowcut,
            highcut=highcut,
            sampling_rate=sampling_rate,
            order=filter_order,
        )
        self.stft = STFTProcessor(
            window_size=stft_window_size,
            hop_size=stft_hop_size,
            n_fft=stft_n_fft,
        )

        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

        self.low_rate = int(self.dual_branch.get("low_rate", int(sampling_rate)))
        self.high_rate = int(self.dual_branch.get("high_rate", self.device_sampling_rate))
        self.high_segment_length = int(self.dual_branch.get("high_segment_length", 420))
        self.high_segment_stride = int(self.dual_branch.get("high_segment_stride", 210))
        self.high_freq_bins_out = int(self.dual_branch.get("high_freq_bins_out", self.stft.freq_bins))
        self.multi_phase_offsets = [
            float(v) for v in self.dual_branch.get("multi_phase_offsets", [0.0, 0.33, 0.66])
        ]

        self._low_decimate_ratio = max(1, int(round(self.high_rate / max(self.low_rate, 1))))

        if self.dual_branch_enabled:
            self.high_bandpass = BandpassFilter(
                lowcut=lowcut,
                highcut=highcut,
                sampling_rate=self.high_rate,
                order=filter_order,
            )
            self.high_stft = STFTProcessor(
                window_size=int(self.dual_branch.get("high_stft_window_size", 120)),
                hop_size=int(self.dual_branch.get("high_stft_hop_size", 60)),
                n_fft=int(self.dual_branch.get("high_stft_n_fft", 230)),
            )
        else:
            self.high_bandpass = None
            self.high_stft = None

    def get_dual_branch_spec(self) -> Dict:
        return dict(self.dual_branch)

    def get_required_window_size(self) -> int:
        if self.dual_branch_enabled:
            return self.high_segment_length
        return self.segment_length

    def get_required_window_stride(self) -> int:
        if self.dual_branch_enabled:
            return self.high_segment_stride
        return self.segment_stride

    def get_output_shape(self) -> Tuple[int, int, int]:
        if self.dual_branch_enabled:
            high_stft = self.high_stft
            assert high_stft is not None
            _, low_time = self.stft.compute_output_shape(self.segment_length)
            _, high_time = high_stft.compute_output_shape(self.high_segment_length)
            if low_time != high_time:
                raise ValueError(
                    f"Dual-branch time frame mismatch: low={low_time}, high={high_time}. "
                    "Please align segment and STFT settings."
                )
            return (self.num_channels * 2, int(self.high_freq_bins_out), int(low_time))
        freq_bins, time_frames = self.stft.compute_output_shape(self.segment_length)
        return (self.num_channels, int(freq_bins), int(time_frames))

    def set_normalization_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._norm_mean = mean
        self._norm_std = std

    def _to_2d_channels(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        if signal.shape[1] > self.num_channels:
            signal = signal[:, : self.num_channels]
        return signal.astype(np.float64)

    @staticmethod
    def _resize_freq_axis(spec: np.ndarray, target_bins: int) -> np.ndarray:
        src_bins, time_frames = spec.shape
        if src_bins == target_bins:
            return spec
        src_x = np.linspace(0.0, 1.0, src_bins)
        dst_x = np.linspace(0.0, 1.0, target_bins)
        resized = np.empty((target_bins, time_frames), dtype=np.float64)
        for frame_idx in range(time_frames):
            resized[:, frame_idx] = np.interp(dst_x, src_x, spec[:, frame_idx])
        return resized

    def _single_branch(self, signal: np.ndarray) -> np.ndarray:
        signal = self._to_2d_channels(signal)
        filtered = self.bandpass(signal)
        rectified = rectify(filtered)
        normalized, _, _ = normalize(rectified, mean=self._norm_mean, std=self._norm_std)

        spectrograms = [self.stft(normalized[:, ch]) for ch in range(normalized.shape[1])]
        result = np.stack(spectrograms, axis=0)
        return result.astype(np.float32)

    def _dual_branch(self, raw_signal: np.ndarray) -> np.ndarray:
        raw_signal = self._to_2d_channels(raw_signal)
        if raw_signal.shape[0] < self.high_segment_length:
            raise ValueError(
                f"Dual-branch requires at least {self.high_segment_length} samples, "
                f"got {raw_signal.shape[0]}"
            )
        raw_signal = raw_signal[: self.high_segment_length]

        low_signal = raw_signal[:: self._low_decimate_ratio]
        if low_signal.shape[0] < self.segment_length:
            raise ValueError(
                f"Low branch requires at least {self.segment_length} samples after decimation, "
                f"got {low_signal.shape[0]}"
            )
        low_signal = low_signal[: self.segment_length]

        high_signal = raw_signal

        low_filtered = self.bandpass(low_signal)
        low_rectified = rectify(low_filtered)
        low_norm, _, _ = normalize(low_rectified, mean=self._norm_mean, std=self._norm_std)
        low_specs = [self.stft(low_norm[:, ch]) for ch in range(low_norm.shape[1])]
        low_stack = np.stack(low_specs, axis=0)

        high_bandpass = self.high_bandpass
        high_stft = self.high_stft
        assert high_bandpass is not None and high_stft is not None
        high_filtered = high_bandpass(high_signal)
        high_rectified = rectify(high_filtered)
        high_norm, _, _ = normalize(high_rectified)
        high_specs = []
        for channel_idx in range(high_norm.shape[1]):
            high_spec = high_stft(high_norm[:, channel_idx])
            high_spec = self._resize_freq_axis(high_spec, self.high_freq_bins_out)
            high_specs.append(high_spec)
        high_stack = np.stack(high_specs, axis=0)

        if low_stack.shape[1:] != high_stack.shape[1:]:
            raise ValueError(
                f"Dual-branch shape mismatch low={low_stack.shape[1:]}, high={high_stack.shape[1:]}"
            )
        fused = np.concatenate([low_stack, high_stack], axis=0)
        return fused.astype(np.float32)

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self.dual_branch_enabled:
            return self._dual_branch(signal)
        return self._single_branch(signal)

    def process_window(self, window: np.ndarray) -> np.ndarray:
        return self.process(window)
