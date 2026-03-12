"""Load NinaPro DB5 MAT archives and build full-coverage EMG pretraining samples."""

from __future__ import annotations

import io
import logging
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import scipy.io as sio

from ninapro_db5.config import DB5PretrainConfig
from shared.preprocessing import PreprocessPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DB5WindowRecord:
    feature: np.ndarray
    label: int
    source_id: str
    metadata: dict[str, Any]


class DB5PretrainDatasetLoader:
    """Build EMG-only pretraining windows from NinaPro DB5 zip archives."""

    def __init__(self, data_dir: str | Path, config: DB5PretrainConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self._stft_pipeline = PreprocessPipeline(
            {
                "sampling_rate": config.feature.target_sampling_rate_hz,
                "num_channels": config.feature.first_myo_channel_count,
                "lowcut": config.feature.lowcut_hz,
                "highcut": config.feature.highcut_hz,
                "target_length": config.feature.target_window_samples,
                "segment_length": config.feature.target_window_samples,
                "segment_stride": config.feature.target_window_samples,
                "stft_window": config.feature.emg_stft_window,
                "stft_hop": config.feature.emg_stft_hop,
                "n_fft": config.feature.emg_n_fft,
                "freq_bins_out": config.feature.emg_freq_bins,
                "normalize": "log",
                "clip_min": 0.0,
                "clip_max": 10.0,
                "dual_branch": {"enabled": False},
            }
        )
        self._class_name_by_label: dict[int, str] = {}
        self._gesture_label_map: dict[tuple[int, int], int] = {}
        self._window_diagnostics: dict[str, Any] = {
            "totals": {
                "segments": 0,
                "raw_candidates": 0,
                "passed_quality": 0,
                "selected": 0,
                "filtered_by_quality": 0,
            },
            "per_class": {},
        }

    def _zip_files(self) -> list[Path]:
        files = sorted(self.data_dir.glob(self.config.zip_glob))
        if not files:
            raise FileNotFoundError(f"No DB5 zip files matched {self.config.zip_glob!r} in {self.data_dir}")
        deduped: dict[str, Path] = {}
        for path in files:
            stem = path.stem.replace(" (1)", "")
            current = deduped.get(stem)
            if current is None:
                deduped[stem] = path
                continue
            if " (1)" in current.stem and " (1)" not in path.stem:
                deduped[stem] = path
        return [deduped[key] for key in sorted(deduped.keys())]

    @staticmethod
    def _iter_segments(labels: np.ndarray) -> Iterator[tuple[int, int, int]]:
        start = 0
        current = int(labels[0])
        for index in range(1, labels.shape[0]):
            value = int(labels[index])
            if value != current:
                yield current, start, index
                start = index
                current = value
        yield current, start, labels.shape[0]

    def _resample_window(self, signal: np.ndarray) -> np.ndarray:
        target_len = int(self.config.feature.target_window_samples)
        if signal.shape[0] == target_len:
            return signal.astype(np.float32)
        src = np.linspace(0.0, 1.0, signal.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        resampled = np.empty((target_len, signal.shape[1]), dtype=np.float32)
        for ch in range(signal.shape[1]):
            resampled[:, ch] = np.interp(dst, src, signal[:, ch]).astype(np.float32)
        return resampled

    @staticmethod
    def _window_energy(window: np.ndarray) -> float:
        return float(np.mean(np.abs(window)))

    @staticmethod
    def _window_std(window: np.ndarray) -> float:
        return float(np.mean(np.std(window, axis=0)))

    @staticmethod
    def _window_clip_ratio(window: np.ndarray, saturation_abs: float) -> float:
        return float(np.mean(np.abs(window) >= float(saturation_abs)))

    def _select_windows(
        self,
        segment_signal: np.ndarray,
        *,
        allow_many: int,
        local_label: int,
    ) -> tuple[list[tuple[np.ndarray, dict[str, float]]], dict[str, int]]:
        window = int(self.config.feature.source_window_samples)
        step = int(self.config.feature.source_step_samples)
        if segment_signal.shape[0] < window:
            return [], {
                "raw_candidates": 0,
                "passed_quality": 0,
                "selected": 0,
                "filtered_by_quality": 0,
            }
        starts = list(range(0, segment_signal.shape[0] - window + 1, step))
        if not starts:
            starts = [0]
        is_rest = int(local_label) == 0
        candidates: list[tuple[np.ndarray, dict[str, float], float]] = []
        for start in starts:
            raw_window = segment_signal[start : start + window]
            energy = self._window_energy(raw_window)
            mean_std = self._window_std(raw_window)
            clip_ratio = self._window_clip_ratio(raw_window, self.config.feature.saturation_abs)
            if clip_ratio > float(self.config.feature.clip_ratio_max):
                continue

            if is_rest:
                if energy > float(self.config.feature.energy_min):
                    continue
                if mean_std > float(self.config.feature.static_std_min):
                    continue
                score = -(energy + mean_std + clip_ratio * 10.0)
            else:
                if energy < float(self.config.feature.energy_min):
                    continue
                if mean_std < float(self.config.feature.static_std_min):
                    continue
                score = energy + mean_std - clip_ratio * 10.0

            candidates.append(
                (
                    self._resample_window(raw_window),
                    {
                        "energy": float(energy),
                        "mean_std": float(mean_std),
                        "clip_ratio": float(clip_ratio),
                    },
                    float(score),
                )
            )

        candidates.sort(key=lambda item: item[2], reverse=True)
        selected = candidates[: int(max(0, allow_many))]
        output = [(window_item, metrics) for window_item, metrics, _score in selected]
        diag = {
            "raw_candidates": int(len(starts)),
            "passed_quality": int(len(candidates)),
            "selected": int(len(output)),
            "filtered_by_quality": int(max(0, len(starts) - len(candidates))),
        }
        return output, diag

    @staticmethod
    def _gesture_key(exercise: int, local_label: int) -> str:
        return f"E{int(exercise)}_G{int(local_label):02d}"

    def _global_label(self, exercise: int, local_label: int) -> tuple[int, str]:
        if local_label == 0:
            return 0, "REST"
        key = (int(exercise), int(local_label))
        global_label = self._gesture_label_map.get(key)
        if global_label is None:
            base = 1 if self.config.include_rest_class else 0
            global_label = base + len(self._gesture_label_map)
            self._gesture_label_map[key] = global_label
        class_name = self._gesture_key(exercise, local_label)
        return global_label, class_name

    def _load_mat_from_zip(self, zip_path: Path, member: zipfile.ZipInfo) -> dict[str, Any]:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as handle:
                payload = handle.read()
        return sio.loadmat(io.BytesIO(payload))

    def _iter_records(self) -> Iterator[DB5WindowRecord]:
        for zip_path in self._zip_files():
            subject = zip_path.stem.replace(" (1)", "")
            with zipfile.ZipFile(zip_path) as zf:
                members = sorted(
                    (info for info in zf.infolist() if info.filename.lower().endswith(".mat")),
                    key=lambda info: info.filename,
                )
            for member in members:
                mat = self._load_mat_from_zip(zip_path, member)
                exercise = int(np.asarray(mat["exercise"]).reshape(-1)[0])
                labels_key = "restimulus" if self.config.use_restimulus else "stimulus"
                labels = np.asarray(mat[labels_key]).reshape(-1).astype(np.int32)
                repetitions = np.asarray(mat["rerepetition"]).reshape(-1).astype(np.int32)
                emg = np.asarray(mat["emg"], dtype=np.float32)
                if self.config.feature.use_first_myo_only:
                    emg = emg[:, : int(self.config.feature.first_myo_channel_count)]
                class_counter = Counter(labels.tolist())
                logger.info(
                    "DB5 file %s/%s exercise=%s labels=%s",
                    subject,
                    Path(member.filename).name,
                    exercise,
                    dict(class_counter),
                )

                segment_index = 0
                for local_label, start, end in self._iter_segments(labels):
                    segment = emg[start:end]
                    repetition = int(np.max(repetitions[start:end])) if end > start else 0
                    if local_label == 0 and not self.config.include_rest_class:
                        continue
                    global_label, class_name = self._global_label(exercise, int(local_label))
                    allow_many = int(
                        self.config.feature.max_rest_windows_per_segment
                        if local_label == 0
                        else self.config.feature.max_windows_per_segment
                    )
                    windows, selection_diag = self._select_windows(
                        segment,
                        allow_many=allow_many,
                        local_label=int(local_label),
                    )
                    self._class_name_by_label[int(global_label)] = class_name
                    totals = self._window_diagnostics["totals"]
                    totals["segments"] += 1
                    totals["raw_candidates"] += int(selection_diag["raw_candidates"])
                    totals["passed_quality"] += int(selection_diag["passed_quality"])
                    totals["selected"] += int(selection_diag["selected"])
                    totals["filtered_by_quality"] += int(selection_diag["filtered_by_quality"])
                    per_class = self._window_diagnostics["per_class"].setdefault(
                        class_name,
                        {
                            "segments": 0,
                            "raw_candidates": 0,
                            "passed_quality": 0,
                            "selected": 0,
                            "filtered_by_quality": 0,
                        },
                    )
                    per_class["segments"] += 1
                    per_class["raw_candidates"] += int(selection_diag["raw_candidates"])
                    per_class["passed_quality"] += int(selection_diag["passed_quality"])
                    per_class["selected"] += int(selection_diag["selected"])
                    per_class["filtered_by_quality"] += int(selection_diag["filtered_by_quality"])
                    if not windows:
                        segment_index += 1
                        continue

                    for win_idx, (window, quality_metrics) in enumerate(windows):
                        feature = self._stft_pipeline.process_window(window)
                        source_id = (
                            f"{subject}|{Path(member.filename).stem}|label={local_label}|rep={repetition}|"
                            f"seg={segment_index}|win={win_idx}"
                        )
                        metadata = {
                            "subject": subject,
                            "file": Path(member.filename).name,
                            "exercise": exercise,
                            "local_label": int(local_label),
                            "gesture_key": self._gesture_key(exercise, int(local_label)),
                            "class_name": class_name,
                            "global_label": int(global_label),
                            "repetition": repetition,
                            "window_energy": float(quality_metrics["energy"]),
                            "window_std": float(quality_metrics["mean_std"]),
                            "window_clip_ratio": float(quality_metrics["clip_ratio"]),
                        }
                        yield DB5WindowRecord(
                            feature=feature,
                            label=int(global_label),
                            source_id=source_id,
                            metadata=metadata,
                        )
                    segment_index += 1

    def load_all_with_sources(
        self,
        *,
        return_metadata: bool = False,
    ):
        self._class_name_by_label.clear()
        self._gesture_label_map.clear()
        self._window_diagnostics = {
            "totals": {
                "segments": 0,
                "raw_candidates": 0,
                "passed_quality": 0,
                "selected": 0,
                "filtered_by_quality": 0,
            },
            "per_class": {},
        }
        features: list[np.ndarray] = []
        labels: list[int] = []
        source_ids: list[str] = []
        metadata_rows: list[dict[str, Any]] = []
        for record in self._iter_records():
            features.append(record.feature)
            labels.append(int(record.label))
            source_ids.append(record.source_id)
            metadata_rows.append(dict(record.metadata))
        if not features:
            raise RuntimeError("No DB5 pretraining samples were selected from the archive set.")
        sample_array = np.stack(features, axis=0).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        source_array = np.asarray(source_ids, dtype=object)
        if return_metadata:
            return sample_array, label_array, source_array, metadata_rows
        return sample_array, label_array, source_array

    def get_class_names(self) -> list[str]:
        if not self._class_name_by_label:
            return []
        names: list[str] = []
        for label, name in sorted(self._class_name_by_label.items()):
            if str(name).upper() == "REST" and not self.config.include_rest_class:
                continue
            names.append(name)
        return names

    def get_window_diagnostics(self) -> dict[str, Any]:
        totals = dict(self._window_diagnostics.get("totals", {}))
        per_class = {
            str(name): dict(values)
            for name, values in (self._window_diagnostics.get("per_class", {}) or {}).items()
        }
        return {"totals": totals, "per_class": per_class}
