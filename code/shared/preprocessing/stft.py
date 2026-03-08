"""STFT transform and preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.config import PreprocessConfig

from .filters import BandpassFilter, normalize, rectify


class SignalWindower:
    def __init__(self, window_size: int, stride: int):
        self.window_size = int(window_size)
        self.stride = int(stride)

    def segment(self, signal: np.ndarray) -> List[np.ndarray]:
        total = signal.shape[0]
        segments: List[np.ndarray] = []
        start = 0
        while start + self.window_size <= total:
            segments.append(signal[start : start + self.window_size])
            start += self.stride
        return segments

    def split(self, signal: np.ndarray) -> List[np.ndarray]:
        return self.segment(signal)

    def count_segments(self, signal_length: int) -> int:
        if signal_length < self.window_size:
            return 0
        return (signal_length - self.window_size) // self.stride + 1


class STFTProcessor:
    def __init__(self, window_size: int = 24, hop_size: int = 12, n_fft: int = 46):
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.n_fft = int(n_fft)
        self.freq_bins = self.n_fft // 2 + 1

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


def _default_dual_branch_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "fuse_mode": "concat_channels",
        "low_rate": 200,
        "high_rate": 1000,
        "high_segment_length": 420,
        "high_segment_stride": 210,
        "high_stft_window": 120,
        "high_stft_hop": 60,
        "high_stft_n_fft": 230,
        "high_freq_bins_out": 24,
        "multi_phase_offsets": [0.0, 0.33, 0.66],
    }


class PreprocessPipeline:
    def __init__(self, config: Optional[Any] = None, **kwargs: Any):
        if isinstance(config, PreprocessConfig):
            dual_branch = config.dual_branch
            cfg = {
                "sampling_rate": config.sampling_rate,
                "num_channels": config.num_channels,
                "lowcut": config.lowcut,
                "highcut": config.highcut,
                "filter_order": config.filter_order,
                "device_sampling_rate": config.device_sampling_rate,
                "target_length": config.target_length,
                "segment_stride": config.segment_stride,
                "stft_window": config.stft_window,
                "stft_hop": config.stft_hop,
                "n_fft": config.n_fft,
                "freq_bins_out": config.freq_bins_out,
                "normalize": config.normalize,
                "clip_min": config.clip_min,
                "clip_max": config.clip_max,
                "dual_branch": dual_branch,
            }
        elif isinstance(config, dict):
            cfg = dict(config)
        elif config is None:
            cfg = {}
        else:
            cfg = {"sampling_rate": config}
        cfg.update(kwargs)

        self.sampling_rate = float(cfg.get("sampling_rate", 200.0))
        self.num_channels = int(cfg.get("num_channels", PreprocessConfig().num_channels))
        self.lowcut = float(cfg.get("lowcut", 20.0))
        self.highcut = float(cfg.get("highcut", 90.0))
        self.filter_order = int(cfg.get("filter_order", 4))
        self.device_sampling_rate = int(cfg.get("device_sampling_rate", 1000))
        self.target_length = int(cfg.get("target_length", cfg.get("segment_length", 84)))
        self.segment_length = int(cfg.get("segment_length", self.target_length))
        self.segment_stride = int(cfg.get("segment_stride", max(1, self.segment_length // 2)))
        self.freq_bins_out = int(cfg.get("freq_bins_out", 24))
        self.normalize_mode = str(cfg.get("normalize", "log"))
        self.clip_min = float(cfg.get("clip_min", 0.0))
        self.clip_max = float(cfg.get("clip_max", 10.0))

        dual_branch = cfg.get("dual_branch")
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
        if "high_stft_window_size" in merged and "high_stft_window" not in merged:
            merged["high_stft_window"] = merged["high_stft_window_size"]
        if "high_stft_hop_size" in merged and "high_stft_hop" not in merged:
            merged["high_stft_hop"] = merged["high_stft_hop_size"]
        self.dual_branch = merged
        self.dual_branch_enabled = bool(self.dual_branch.get("enabled", False))
        self.fuse_mode = str(self.dual_branch.get("fuse_mode", "concat_channels"))
        if self.fuse_mode != "concat_channels":
            raise ValueError(f"Unsupported dual-branch fuse_mode={self.fuse_mode!r}")

        self.bandpass = BandpassFilter(
            lowcut=self.lowcut,
            highcut=self.highcut,
            sampling_rate=self.sampling_rate,
            order=self.filter_order,
        )
        self.stft = STFTProcessor(
            window_size=int(cfg.get("stft_window", cfg.get("stft_window_size", 24))),
            hop_size=int(cfg.get("stft_hop", cfg.get("stft_hop_size", 12))),
            n_fft=int(cfg.get("n_fft", cfg.get("stft_n_fft", 46))),
        )

        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

        self.low_rate = int(self.dual_branch.get("low_rate", int(self.sampling_rate)))
        self.high_rate = int(self.dual_branch.get("high_rate", self.device_sampling_rate))
        self.high_segment_length = int(self.dual_branch.get("high_segment_length", self.segment_length))
        self.high_segment_stride = int(self.dual_branch.get("high_segment_stride", self.segment_stride))
        self.high_freq_bins_out = int(self.dual_branch.get("high_freq_bins_out", self.freq_bins_out))
        self.multi_phase_offsets = [float(v) for v in self.dual_branch.get("multi_phase_offsets", [0.0])]
        self._low_decimate_ratio = max(1, int(round(self.high_rate / max(self.low_rate, 1))))

        if self.dual_branch_enabled:
            self.high_bandpass = BandpassFilter(
                lowcut=self.lowcut,
                highcut=self.highcut,
                sampling_rate=self.high_rate,
                order=self.filter_order,
            )
            self.high_stft = STFTProcessor(
                window_size=int(self.dual_branch.get("high_stft_window", 120)),
                hop_size=int(self.dual_branch.get("high_stft_hop", 60)),
                n_fft=int(self.dual_branch.get("high_stft_n_fft", 230)),
            )
        else:
            self.high_bandpass = None
            self.high_stft = None
        self._target_time_frames = self._compute_target_time_frames()

    def _compute_target_time_frames(self) -> int:
        _, low_frames = self.stft.compute_output_shape(self.target_length)
        if not self.dual_branch_enabled or self.high_stft is None:
            return int(low_frames)
        _, high_frames = self.high_stft.compute_output_shape(self.high_segment_length)
        return int(max(low_frames, high_frames))

    def get_dual_branch_spec(self) -> Dict[str, Any]:
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
            return (self.num_channels * 2, self.high_freq_bins_out, self._target_time_frames)
        return (self.num_channels, self.freq_bins_out, self._target_time_frames)

    def set_normalization_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._norm_mean = mean
        self._norm_std = std

    def _to_2d_channels(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        if signal.shape[1] < self.num_channels:
            raise ValueError(
                f"Expected at least {self.num_channels} EMG channels, got {signal.shape[1]}"
            )
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

    @staticmethod
    def _resize_time_axis(spec: np.ndarray, target_frames: int) -> np.ndarray:
        src_bins, time_frames = spec.shape
        if time_frames == target_frames:
            return spec
        if time_frames <= 1:
            return np.repeat(spec, target_frames, axis=1)
        src_x = np.linspace(0.0, 1.0, time_frames)
        dst_x = np.linspace(0.0, 1.0, target_frames)
        resized = np.empty((src_bins, target_frames), dtype=np.float64)
        for bin_idx in range(src_bins):
            resized[bin_idx] = np.interp(dst_x, src_x, spec[bin_idx])
        return resized

    def _postprocess_spec(self, spec: np.ndarray, *, target_bins: int) -> np.ndarray:
        spec = self._resize_freq_axis(spec, target_bins)
        spec = self._resize_time_axis(spec, self._target_time_frames)
        if self.normalize_mode == "log":
            spec = np.log1p(np.maximum(spec, 0.0))
        return np.clip(spec, self.clip_min, self.clip_max)

    def _single_branch(self, signal: np.ndarray) -> np.ndarray:
        signal = self._to_2d_channels(signal)
        filtered = self.bandpass(signal)
        rectified = rectify(filtered)
        normalized, _, _ = normalize(rectified, mean=self._norm_mean, std=self._norm_std)
        spectrograms = [
            self._postprocess_spec(self.stft(normalized[:, ch]), target_bins=self.freq_bins_out)
            for ch in range(normalized.shape[1])
        ]
        return np.stack(spectrograms, axis=0).astype(np.float32)

    def _dual_branch(self, raw_signal: np.ndarray) -> np.ndarray:
        raw_signal = self._to_2d_channels(raw_signal)
        if raw_signal.shape[0] < self.high_segment_length:
            raise ValueError(
                f"Dual-branch requires at least {self.high_segment_length} samples, got {raw_signal.shape[0]}"
            )
        raw_signal = raw_signal[: self.high_segment_length]

        low_signal = raw_signal[:: self._low_decimate_ratio]
        if low_signal.shape[0] == 0:
            raise ValueError("Low branch became empty after decimation")

        low_filtered = self.bandpass(low_signal)
        low_rectified = rectify(low_filtered)
        low_norm, _, _ = normalize(low_rectified, mean=self._norm_mean, std=self._norm_std)
        low_specs = [
            self._postprocess_spec(self.stft(low_norm[:, ch]), target_bins=self.freq_bins_out)
            for ch in range(low_norm.shape[1])
        ]
        low_stack = np.stack(low_specs, axis=0)

        high_bandpass = self.high_bandpass
        high_stft = self.high_stft
        assert high_bandpass is not None and high_stft is not None
        high_filtered = high_bandpass(raw_signal)
        high_rectified = rectify(high_filtered)
        high_norm, _, _ = normalize(high_rectified)
        high_specs = [
            self._postprocess_spec(high_stft(high_norm[:, ch]), target_bins=self.high_freq_bins_out)
            for ch in range(high_norm.shape[1])
        ]
        high_stack = np.stack(high_specs, axis=0)
        return np.concatenate([low_stack, high_stack], axis=0).astype(np.float32)

    def extract_segments(self, signal: np.ndarray) -> List[np.ndarray]:
        signal = self._to_2d_channels(signal)
        window_size = self.get_required_window_size()
        stride = self.get_required_window_stride()
        if signal.shape[0] < window_size:
            return []

        starts: List[int] = []
        max_start = signal.shape[0] - window_size
        offsets = self.multi_phase_offsets if self.dual_branch_enabled else [0.0]
        for offset in offsets:
            start = int(round(max(0.0, offset) * stride))
            while start <= max_start:
                starts.append(start)
                start += stride
        starts = sorted(set(starts))
        return [signal[start : start + window_size].astype(np.float32) for start in starts]

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self.dual_branch_enabled:
            return self._dual_branch(signal)
        return self._single_branch(signal)

    def process_window(self, window: np.ndarray) -> np.ndarray:
        return self.process(window)
