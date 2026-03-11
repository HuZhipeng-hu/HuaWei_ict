"""Feature-space data augmentation for train split only."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class DataAugmentor:
    """Apply lightweight stochastic augmentation to spectrogram features."""

    def __init__(
        self,
        temporal_shift_max: int = 2,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        noise_std: float = 0.02,
        channel_drop_prob: float = 0.1,
        mixup_alpha: float = 0.2,
        seed: Optional[int] = None,
        amplitude_scale: Optional[float] = None,
    ):
        if amplitude_scale is not None:
            amp = abs(float(amplitude_scale))
            scale_min = 1.0 - amp
            scale_max = 1.0 + amp

        self.temporal_shift_max = max(0, int(temporal_shift_max))
        self.scale_min = float(min(scale_min, scale_max))
        self.scale_max = float(max(scale_min, scale_max))
        self.noise_std = max(0.0, float(noise_std))
        self.channel_drop_prob = min(1.0, max(0.0, float(channel_drop_prob)))
        self.mixup_alpha = max(1e-6, float(mixup_alpha))
        self.rng = np.random.RandomState(seed)

    def augment(self, spectrogram: np.ndarray) -> np.ndarray:
        result = spectrogram.astype(np.float32, copy=True)

        if self.temporal_shift_max > 0 and self.rng.random() < 0.5:
            result = self._temporal_shift(result)
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            if self.rng.random() < 0.5:
                result = self._amplitude_scale(result)
        if self.noise_std > 0.0 and self.rng.random() < 0.5:
            result = self._add_noise(result)
        if self.channel_drop_prob > 0.0 and self.rng.random() < self.channel_drop_prob:
            result = self._channel_dropout(result)

        return result

    def augment_batch(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        factor: int = 2,
        use_mixup: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        factor = max(1, int(factor))
        all_samples = [samples.astype(np.float32, copy=False)]
        all_labels = [labels.astype(np.int32, copy=False)]

        for _ in range(factor - 1):
            aug_samples = np.asarray([self.augment(sample) for sample in samples], dtype=np.float32)
            all_samples.append(aug_samples)
            all_labels.append(labels.astype(np.int32, copy=True))

        if use_mixup and len(samples) > 1:
            mixup_samples, mixup_labels = self._mixup_batch(samples, labels)
            all_samples.append(mixup_samples)
            all_labels.append(mixup_labels)

        return np.concatenate(all_samples, axis=0), np.concatenate(all_labels, axis=0)

    def _temporal_shift(self, spec: np.ndarray) -> np.ndarray:
        shift = int(self.rng.randint(-self.temporal_shift_max, self.temporal_shift_max + 1))
        if shift == 0:
            return spec

        result = np.zeros_like(spec)
        if shift > 0:
            result[..., shift:] = spec[..., :-shift]
        else:
            result[..., :shift] = spec[..., -shift:]
        return result

    def _amplitude_scale(self, spec: np.ndarray) -> np.ndarray:
        scale = float(self.rng.uniform(self.scale_min, self.scale_max))
        return spec * scale

    def _add_noise(self, spec: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.noise_std, size=spec.shape).astype(np.float32)
        return spec + noise

    def _channel_dropout(self, spec: np.ndarray) -> np.ndarray:
        result = spec.copy()
        channel = int(self.rng.randint(0, spec.shape[0]))
        result[channel] = 0.0
        return result

    def _mixup_batch(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(samples)
        perm = self.rng.permutation(n)
        mixed_samples = np.empty_like(samples, dtype=np.float32)
        mixed_labels = np.empty_like(labels, dtype=np.int32)

        for i in range(n):
            j = int(perm[i])
            lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
            mixed_samples[i] = lam * samples[i] + (1.0 - lam) * samples[j]
            mixed_labels[i] = labels[i] if lam >= 0.5 else labels[j]

        return mixed_samples, mixed_labels
