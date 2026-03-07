"""
CSV dataset loader.

Folder name is treated as class label. Each CSV file is segmented and preprocessed
into spectrogram samples.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from shared.gestures import FOLDER_TO_GESTURE, GestureType, NUM_CLASSES
from shared.preprocessing import PreprocessPipeline, SignalWindower

logger = logging.getLogger(__name__)


class CSVDatasetLoader:
    """Load armband CSV files and convert them into model-ready samples."""

    def __init__(
        self,
        data_dir: str,
        preprocess: PreprocessPipeline,
        num_emg_channels: int = 6,
        device_sampling_rate: int = 1000,
        target_sampling_rate: int = 200,
        segment_length: int = 84,
        segment_stride: int = 42,
        center_value: float = 128.0,
    ):
        self.data_dir = Path(data_dir)
        self.preprocess = preprocess
        self.num_emg_channels = num_emg_channels
        self.device_sampling_rate = device_sampling_rate
        self.target_sampling_rate = target_sampling_rate
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.center_value = center_value

        self.decimate_ratio = max(1, device_sampling_rate // target_sampling_rate)
        self.dual_branch_enabled = bool(getattr(self.preprocess, "dual_branch_enabled", False))

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.windower = SignalWindower(window_size=segment_length, stride=segment_stride)
        self.gesture_folders = self._scan_folders()
        if not self.gesture_folders:
            raise ValueError(f"No valid gesture folders found in {self.data_dir}")

    def _scan_folders(self) -> Dict[GestureType, List[Path]]:
        result: Dict[GestureType, List[Path]] = {}
        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue
            folder_name = folder.name.lower()
            if folder_name not in FOLDER_TO_GESTURE:
                logger.debug("Skip unknown folder: %s", folder.name)
                continue

            gesture = FOLDER_TO_GESTURE[folder_name]
            csv_files = sorted(folder.glob("*.csv"))
            if not csv_files:
                logger.warning("No CSV files under folder: %s", folder)
                continue
            result[gesture] = csv_files
            logger.info("[%s] %s files", gesture.name, len(csv_files))
        return result

    def _source_id(self, csv_path: Path) -> str:
        try:
            return csv_path.relative_to(self.data_dir).as_posix()
        except ValueError:
            return csv_path.as_posix()

    def _read_csv(self, csv_path: Path) -> np.ndarray:
        rows: List[List[float]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < self.num_emg_channels:
                    continue
                try:
                    emg_values = [float(row[i]) for i in range(self.num_emg_channels)]
                except (ValueError, IndexError):
                    continue
                rows.append(emg_values)

        if not rows:
            return np.empty((0, self.num_emg_channels), dtype=np.float32)

        data = np.asarray(rows, dtype=np.float32)
        data -= self.center_value
        return data

    def _decimate(self, signal: np.ndarray) -> np.ndarray:
        if self.decimate_ratio <= 1:
            return signal
        return signal[:: self.decimate_ratio]

    def _segment(self, signal: np.ndarray) -> List[np.ndarray]:
        return self.windower.segment(signal)

    def _segment_dual_branch(self, raw_signal: np.ndarray) -> List[np.ndarray]:
        raw_window = int(getattr(self.preprocess, "high_segment_length", self.segment_length * self.decimate_ratio))
        raw_stride = int(getattr(self.preprocess, "high_segment_stride", self.segment_stride * self.decimate_ratio))
        phase_offsets = list(getattr(self.preprocess, "multi_phase_offsets", [0.0, 0.33, 0.66]))

        segments: List[np.ndarray] = []
        total = raw_signal.shape[0]
        for phase in phase_offsets:
            phase_start = int(round(raw_stride * float(phase)))
            start = max(0, phase_start)
            while start + raw_window <= total:
                segments.append(raw_signal[start : start + raw_window])
                start += raw_stride
        return segments

    def load_file(self, csv_path: Path) -> List[np.ndarray]:
        emg_data = self._read_csv(csv_path)
        if emg_data.shape[0] == 0:
            return []

        if self.dual_branch_enabled:
            min_raw = int(getattr(self.preprocess, "high_segment_length", self.segment_length * self.decimate_ratio))
            if emg_data.shape[0] < min_raw:
                return []
            segments = self._segment_dual_branch(emg_data)
        else:
            decimated = self._decimate(emg_data)
            if decimated.shape[0] < self.segment_length:
                return []
            segments = self._segment(decimated)

        spectrograms: List[np.ndarray] = []
        for segment in segments:
            try:
                spec = self.preprocess.process(segment)
            except Exception as exc:
                logger.debug("Preprocess failed for %s: %s", csv_path.name, exc)
                continue
            spectrograms.append(spec)
        return spectrograms

    def load_all_with_sources(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_samples: List[np.ndarray] = []
        all_labels: List[int] = []
        all_sources: List[str] = []

        for gesture, csv_files in self.gesture_folders.items():
            gesture_samples = 0
            for csv_path in csv_files:
                source_id = self._source_id(csv_path)
                spectrograms = self.load_file(csv_path)
                for spec in spectrograms:
                    all_samples.append(spec)
                    all_labels.append(gesture.value)
                    all_sources.append(source_id)
                    gesture_samples += 1
            logger.info("[%s] loaded %s samples", gesture.name, gesture_samples)

        if not all_samples:
            raise ValueError("No valid samples loaded from dataset")

        samples = np.asarray(all_samples, dtype=np.float32)
        labels = np.asarray(all_labels, dtype=np.int32)
        sources = np.asarray(all_sources, dtype=object)

        logger.info("Loaded dataset: %s samples, shape=%s", len(samples), samples.shape)
        for gesture in GestureType:
            count = int(np.sum(labels == gesture.value))
            logger.info("  %s: %s", gesture.name, count)

        return samples, labels, sources

    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        samples, labels, _ = self.load_all_with_sources()
        return samples, labels

    @staticmethod
    def split(
        samples: np.ndarray,
        labels: np.ndarray,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Legacy sample-level stratified split."""
        rng = np.random.RandomState(seed)

        train_indices: List[int] = []
        val_indices: List[int] = []

        for class_id in range(NUM_CLASSES):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)
            n_val = max(1, int(len(class_indices) * val_ratio))
            val_indices.extend(class_indices[:n_val].tolist())
            train_indices.extend(class_indices[n_val:].tolist())

        train_indices_arr = np.asarray(train_indices, dtype=np.int64)
        val_indices_arr = np.asarray(val_indices, dtype=np.int64)
        rng.shuffle(train_indices_arr)
        rng.shuffle(val_indices_arr)

        return (
            (samples[train_indices_arr], labels[train_indices_arr]),
            (samples[val_indices_arr], labels[val_indices_arr]),
        )

    @staticmethod
    def kfold_split(
        samples: np.ndarray,
        labels: np.ndarray,
        k: int = 5,
        seed: int = 42,
    ):
        """Legacy sample-level stratified KFold."""
        rng = np.random.RandomState(seed)
        class_indices: Dict[int, np.ndarray] = {}
        for class_id in range(NUM_CLASSES):
            idx = np.where(labels == class_id)[0]
            rng.shuffle(idx)
            class_indices[class_id] = idx

        for fold in range(k):
            train_idx: List[int] = []
            val_idx: List[int] = []
            for class_id, idx in class_indices.items():
                n = len(idx)
                fold_size = n // k
                start = fold * fold_size
                end = start + fold_size if fold < k - 1 else n
                val_idx.extend(idx[start:end].tolist())
                train_idx.extend(idx[:start].tolist())
                train_idx.extend(idx[end:].tolist())

            train_idx_arr = np.asarray(train_idx, dtype=np.int64)
            val_idx_arr = np.asarray(val_idx, dtype=np.int64)
            rng.shuffle(train_idx_arr)
            rng.shuffle(val_idx_arr)

            yield (
                fold,
                (samples[train_idx_arr], labels[train_idx_arr]),
                (samples[val_idx_arr], labels[val_idx_arr]),
            )

    def get_stats(self) -> Dict[str, int]:
        stats = {"total_files": 0}
        for gesture, files in self.gesture_folders.items():
            stats[gesture.name] = len(files)
            stats["total_files"] += len(files)
        return stats
