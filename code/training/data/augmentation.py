"""
EMG 数据增强

对原始 EMG 信号施加多种随机变换以扩充训练数据量，
提高模型的鲁棒性和泛化能力。

支持的增强方式:
- 时间扭曲 (Time Warping): 随机拉伸/压缩时间轴
- 幅度缩放 (Amplitude Scaling): 随机缩放信号幅度
- 高斯噪声 (Gaussian Noise): 添加随机噪声模拟传感器噪声
- 通道丢弃 (Channel Dropout): 随机将某个通道的信号置零
- Mixup: 线性混合两个样本，提高泛化能力
"""

import numpy as np
from typing import Optional, Tuple


class DataAugmentor:
    """
    EMG 数据增强器

    对时频谱图 (num_channels, freq_bins, time_frames) 施加随机增强。
    所有增强都是"惰性"的——每次调用 augment() 以给定概率随机决定
    是否应用某个增强，确保增强后的数据多样性。

    Args:
        time_warp_rate: 时间扭曲幅度（0.1 = ±10%）
        amplitude_scale: 幅度缩放范围（0.15 = ±15%）
        noise_std: 高斯噪声标准差
        channel_drop_prob: 通道丢弃概率
        seed: 随机种子（None = 不固定）
    """

    def __init__(
        self,
        time_warp_rate: float = 0.1,
        amplitude_scale: float = 0.15,
        noise_std: float = 0.05,
        channel_drop_prob: float = 0.1,
        mixup_alpha: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.time_warp_rate = time_warp_rate
        self.amplitude_scale = amplitude_scale
        self.noise_std = noise_std
        self.channel_drop_prob = channel_drop_prob
        self.mixup_alpha = mixup_alpha
        self.rng = np.random.RandomState(seed)

    def augment(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        对单个时频谱图施加随机增强组合

        每种增强以 50% 概率独立应用。

        Args:
            spectrogram: (num_channels, freq_bins, time_frames) float32

        Returns:
            增强后的谱图，形状不变
        """
        result = spectrogram.copy()

        # 时间扭曲（约50%概率）
        if self.rng.random() < 0.5:
            result = self._time_warp(result)

        # 幅度缩放（约50%概率）
        if self.rng.random() < 0.5:
            result = self._amplitude_scale(result)

        # 高斯噪声（约50%概率）
        if self.rng.random() < 0.5:
            result = self._add_noise(result)

        # 通道丢弃（较低概率）
        if self.rng.random() < self.channel_drop_prob:
            result = self._channel_dropout(result)

        return result

    def augment_batch(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        factor: int = 3,
        use_mixup: bool = False,
    ) -> tuple:
        """
        批量增强: 将数据集扩充为 factor 倍

        原始数据保留，增强数据追加在后面。
        可选启用 Mixup 增强（在常规增强后叠加）。

        Args:
            samples: (N, C, F, T) 原始样本
            labels: (N,) 原始标签
            factor: 增强倍数（总量 = N * factor）
            use_mixup: 是否在增强后额外添加 Mixup 样本

        Returns:
            (augmented_samples, augmented_labels)
        """
        all_samples = [samples]  # 保留原始数据
        all_labels = [labels]

        # 常规增强（时间扭曲 + 幅度缩放 + 噪声 + 通道丢弃）
        for _ in range(factor - 1):
            aug_samples = np.array([
                self.augment(s) for s in samples
            ])
            all_samples.append(aug_samples)
            all_labels.append(labels.copy())

        # Mixup 增强（可选，生成额外 N 个混合样本）
        if use_mixup and len(samples) > 1:
            mixup_samples, mixup_labels = self._mixup_batch(samples, labels)
            all_samples.append(mixup_samples)
            all_labels.append(mixup_labels)

        return (
            np.concatenate(all_samples, axis=0),
            np.concatenate(all_labels, axis=0),
        )

    def _time_warp(self, spec: np.ndarray) -> np.ndarray:
        """
        时间扭曲: 随机拉伸/压缩时间轴

        通过线性插值实现，保持输出长度不变。
        """
        num_channels, freq_bins, time_frames = spec.shape

        # 随机扭曲因子
        warp_factor = 1.0 + self.rng.uniform(
            -self.time_warp_rate, self.time_warp_rate
        )

        # 原始时间索引
        original_indices = np.arange(time_frames)
        # 扭曲后的时间索引
        warped_indices = np.linspace(
            0, time_frames - 1,
            int(time_frames * warp_factor),
        )

        # 需要 time_frames 个输出点
        output_indices = np.linspace(0, len(warped_indices) - 1, time_frames)

        result = np.zeros_like(spec)
        for ch in range(num_channels):
            for f in range(freq_bins):
                result[ch, f] = np.interp(
                    output_indices,
                    np.arange(len(warped_indices)),
                    np.interp(warped_indices, original_indices, spec[ch, f]),
                )

        return result

    def _amplitude_scale(self, spec: np.ndarray) -> np.ndarray:
        """
        幅度缩放: 随机全局缩放

        模拟佩戴松紧度差异导致的信号幅度变化。
        """
        scale = 1.0 + self.rng.uniform(
            -self.amplitude_scale, self.amplitude_scale
        )
        return spec * scale

    def _add_noise(self, spec: np.ndarray) -> np.ndarray:
        """
        高斯噪声: 添加随机噪声

        模拟传感器噪声和环境干扰。
        """
        noise = self.rng.normal(0, self.noise_std, spec.shape)
        return spec + noise.astype(spec.dtype)

    def _channel_dropout(self, spec: np.ndarray) -> np.ndarray:
        """
        通道丢弃: 随机将一个通道置零

        模拟某个电极接触不良的情况，提高模型鲁棒性。
        """
        result = spec.copy()
        ch = self.rng.randint(0, spec.shape[0])
        result[ch] = 0.0
        return result

    def _mixup_batch(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup 增强: 线性混合随机配对的样本

        对每个样本，从 batch 中随机选一个不同样本，按 Beta(α, α)
        生成混合比 λ，混合 x' = λx₁ + (1-λ)x₂。
        标签取 λ 较大的那个（硬标签 Mixup，兼容 sparse CE 损失）。

        Args:
            samples: (N, C, F, T) 样本
            labels: (N,) 标签

        Returns:
            (mixed_samples, mixed_labels) 各 N 个
        """
        n = len(samples)
        # 随机打乱索引作为配对
        shuffle_idx = self.rng.permutation(n)

        mixed_samples = np.empty_like(samples)
        mixed_labels = np.empty_like(labels)

        for i in range(n):
            j = shuffle_idx[i]
            lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha)

            # 线性混合
            mixed_samples[i] = lam * samples[i] + (1 - lam) * samples[j]

            # 硬标签: 取权重大的那个
            mixed_labels[i] = labels[i] if lam >= 0.5 else labels[j]

        return mixed_samples, mixed_labels
