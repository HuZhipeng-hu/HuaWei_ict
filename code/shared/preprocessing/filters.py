"""
EMG 信号滤波与标准化

提供以下处理步骤（按典型使用顺序排列）：
1. BandpassFilter: 20-90Hz 带通滤波，去除直流偏移和高频噪声
2. rectify():      全波整流（取绝对值），提取肌电包络
3. normalize():    Z-score 标准化，消除通道间幅度差异
"""

import numpy as np
from typing import Optional, Tuple

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class BandpassFilter:
    """
    Butterworth 带通滤波器

    使用零相位滤波（filtfilt）避免引入时延。
    默认通带为 20-90Hz，适用于 sEMG 信号。

    Args:
        lowcut: 下截止频率 (Hz)
        highcut: 上截止频率 (Hz)
        sampling_rate: 采样率 (Hz)
        order: 滤波器阶数
    """

    def __init__(
        self,
        lowcut: float = 20.0,
        highcut: float = 90.0,
        sampling_rate: float = 200.0,
        order: int = 4,
    ):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy 未安装，请运行: pip install scipy")

        self.lowcut = lowcut
        self.highcut = highcut
        self.sampling_rate = sampling_rate
        self.order = order

        # 计算 Butterworth 滤波器系数
        nyquist = sampling_rate / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b, self.a = butter(order, [low, high], btype='band')

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号施加带通滤波

        Args:
            signal: 输入信号
                - 1D: (num_samples,) — 单通道
                - 2D: (num_samples, num_channels) — 多通道

        Returns:
            滤波后的信号，形状不变
        """
        if signal.ndim == 1:
            return self._filter_1d(signal)
        elif signal.ndim == 2:
            # 逐通道滤波
            filtered = np.zeros_like(signal)
            for ch in range(signal.shape[1]):
                filtered[:, ch] = self._filter_1d(signal[:, ch])
            return filtered
        else:
            raise ValueError(f"输入维度必须为 1D 或 2D，实际为 {signal.ndim}D")

    def _filter_1d(self, x: np.ndarray) -> np.ndarray:
        """单通道零相位滤波"""
        # filtfilt 要求数据长度至少为 3 * max(len(a), len(b))
        min_len = 3 * max(len(self.a), len(self.b))
        if len(x) < min_len:
            # 数据太短，用边界值填充后滤波再裁剪
            padded = np.pad(x, (min_len, min_len), mode='edge')
            result = filtfilt(self.b, self.a, padded)
            return result[min_len: min_len + len(x)]
        return filtfilt(self.b, self.a, x)


def rectify(signal: np.ndarray) -> np.ndarray:
    """
    全波整流（取绝对值）

    EMG 信号是双极性的（正负交替），整流后变为单极性，
    更易提取包络和时频特征。

    Args:
        signal: 输入信号，任意维度

    Returns:
        整流后的信号（绝对值），形状不变
    """
    return np.abs(signal)


def normalize(
    signal: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score 标准化

    将信号标准化为零均值、单位方差，消除通道间的幅度差异。
    如果未提供均值和标准差，则从输入数据中计算。

    Args:
        signal: 输入信号 (num_samples, num_channels)
        mean: 预计算的均值（用于推理时使用训练集统计量）
        std: 预计算的标准差
        eps: 防止除零的小常量

    Returns:
        (normalized_signal, mean, std) — 后两个用于保存/复用
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if mean is None:
        mean = np.mean(signal, axis=0)
    if std is None:
        std = np.std(signal, axis=0)

    # 避免除零
    std_safe = np.maximum(std, eps)
    normalized = (signal - mean) / std_safe

    return normalized, mean, std
