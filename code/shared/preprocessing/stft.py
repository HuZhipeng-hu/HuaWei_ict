"""
STFT 变换与预处理流水线

将多通道 EMG 时域信号转换为时频谱图，供 CNN 模型输入。

处理步骤:
    原始信号 (num_samples, num_channels)
    → 带通滤波 → 整流 → 标准化
    → 滑动窗口切分 → 逐窗口 STFT
    → 时频谱图 (num_channels, freq_bins, time_frames)
"""

import numpy as np
from typing import Optional, Tuple, List

from .filters import BandpassFilter, rectify, normalize


class SignalWindower:
    """
    滑动窗口切分

    将连续信号切分为固定长度的重叠片段，供后续 STFT 或预处理使用。
    训练时用于从长 CSV 数据中生成训练样本，运行时用于从连续采集缓冲区中
    切出推理窗口。

    Args:
        window_size: 每个片段的长度（采样点数）
        stride: 窗口滑动步长（重叠量 = window_size - stride）
    """

    def __init__(self, window_size: int, stride: int):
        self.window_size = window_size
        self.stride = stride

    def segment(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        将连续信号切成片段列表

        Args:
            signal: (num_samples,) 或 (num_samples, num_channels)

        Returns:
            片段列表，每个形状为 (window_size,) 或 (window_size, num_channels)
        """
        total = signal.shape[0]
        segments = []
        start = 0
        while start + self.window_size <= total:
            segments.append(signal[start: start + self.window_size])
            start += self.stride
        return segments

    def count_segments(self, signal_length: int) -> int:
        """计算给定信号长度可以切出多少个片段"""
        if signal_length < self.window_size:
            return 0
        return (signal_length - self.window_size) // self.stride + 1


class STFTProcessor:
    """
    短时傅里叶变换 (STFT).

    将一维时域信号切成重叠小窗，逐窗做 FFT，得到时频表示。

    Args:
        window_size: STFT 窗口长度（采样点数）
        hop_size: 窗口移动步长
        n_fft: FFT 点数（≥ window_size，零填充到此长度）
    """

    def __init__(
        self,
        window_size: int = 24,
        hop_size: int = 12,
        n_fft: int = 46,
    ):
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft

        # 频率 bin 数（取实数 FFT 的正频率部分）
        self.freq_bins = n_fft // 2 + 1  # 24 for n_fft=46

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        对单通道信号执行 STFT

        Args:
            signal: (num_samples,) 一维时域信号

        Returns:
            spectrogram: (freq_bins, time_frames) 幅度谱
        """
        # 切帧
        num_frames = max(0, (len(signal) - self.window_size) // self.hop_size + 1)
        if num_frames == 0:
            # 信号太短，零填充
            padded = np.zeros(self.window_size)
            padded[:len(signal)] = signal
            frame = padded * np.hanning(self.window_size)
            spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))
            return spectrum.reshape(-1, 1)

        frames = []
        for i in range(num_frames):
            start = i * self.hop_size
            frame = signal[start: start + self.window_size]
            # 加汉宁窗减少频谱泄漏
            frame = frame * np.hanning(self.window_size)
            # FFT
            spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))
            frames.append(spectrum)

        # (freq_bins, time_frames)
        return np.array(frames).T

    def compute_output_shape(self, signal_length: int) -> Tuple[int, int]:
        """
        计算给定信号长度的 STFT 输出形状

        Returns:
            (freq_bins, time_frames)
        """
        num_frames = max(1, (signal_length - self.window_size) // self.hop_size + 1)
        return (self.freq_bins, num_frames)


class PreprocessPipeline:
    """
    完整 EMG 信号预处理流水线

    将原始多通道时域信号处理成模型可接受的时频谱图。
    训练和运行时必须使用相同的 PreprocessPipeline 实例参数，
    否则模型输入分布不匹配。

    处理流程:
        多通道信号 (num_samples, num_channels)
        → 带通滤波
        → 全波整流
        → Z-score 标准化
        → 逐通道 STFT
        → 堆叠为 (num_channels, freq_bins, time_frames)

    Args:
        sampling_rate: 采样率 (Hz)
        num_channels: EMG 通道数
        lowcut: 带通滤波下截止频率 (Hz)
        highcut: 带通滤波上截止频率 (Hz)
        filter_order: 滤波器阶数
        stft_window_size: STFT 窗口大小
        stft_hop_size: STFT 步长
        stft_n_fft: FFT 点数
    """

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
    ):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels

        # 子模块
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

        # 标准化统计量（可选外部注入）
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

    def set_normalization_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """
        设置标准化统计量（从训练集计算后注入）

        在推理时使用训练集的统计量，而非在线计算。

        Args:
            mean: (num_channels,) 各通道均值
            std: (num_channels,) 各通道标准差
        """
        self._norm_mean = mean
        self._norm_std = std

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        处理多通道 EMG 信号

        Args:
            signal: (num_samples, num_channels) 原始 EMG 信号
                    或 (num_samples,) 单通道信号

        Returns:
            spectrogram: (num_channels, freq_bins, time_frames)
                         float32 时频谱图
        """
        # 确保 2D
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        # 只取前 num_channels 个通道
        if signal.shape[1] > self.num_channels:
            signal = signal[:, :self.num_channels]

        signal = signal.astype(np.float64)

        # 1. 带通滤波
        filtered = self.bandpass(signal)

        # 2. 全波整流
        rectified = rectify(filtered)

        # 3. Z-score 标准化
        normalized, mean, std = normalize(
            rectified,
            mean=self._norm_mean,
            std=self._norm_std,
        )

        # 4. 逐通道 STFT
        num_channels = normalized.shape[1]
        spectrograms = []
        for ch in range(num_channels):
            spec = self.stft(normalized[:, ch])
            spectrograms.append(spec)

        # 堆叠: (num_channels, freq_bins, time_frames)
        result = np.stack(spectrograms, axis=0)
        return result.astype(np.float32)

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """
        处理一个滑动窗口的数据（实时推理用）

        与 process() 相同的处理流程，但适配较短的窗口数据。

        Args:
            window: (window_size, num_channels) 一个窗口的 EMG 数据

        Returns:
            spectrogram: (num_channels, freq_bins, time_frames) float32
        """
        return self.process(window)
