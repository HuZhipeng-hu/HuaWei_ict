"""CSV dataset loader with dual-branch preprocessing and quality filtering."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from shared.config import PreprocessConfig, QualityFilterConfig
from shared.preprocessing import PreprocessPipeline, SignalWindower

logger = logging.getLogger(__name__)


class CSVDatasetLoader:
    """Load EMG CSV files into spectrogram tensors for training/evaluation."""

    def __init__(
        self,
        data_dir: str | Path,
        gesture_to_idx: Dict[str, int],
        preprocess_config: PreprocessConfig,
        quality_filter: Optional[QualityFilterConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.gesture_to_idx = gesture_to_idx
        self.preprocess = PreprocessPipeline(preprocess_config)
        self._uses_dual_branch = bool(preprocess_config.dual_branch.enabled)

        if quality_filter is None:
            quality_filter = QualityFilterConfig(enabled=False)
        self.quality_filter = quality_filter

        self._recording_meta: Dict[str, Dict[str, Any]] = {}
        self._quality_stats: Dict[str, Any] = self._init_quality_stats()

    @staticmethod
    def _init_quality_stats() -> Dict[str, Any]:
        return {
            "total_windows": 0,
            "kept_windows": 0,
            "filtered_total": 0,
            "filtered_low_energy": 0,
            "filtered_clipped": 0,
            "filtered_static": 0,
            "per_class": defaultdict(
                lambda: {
                    "total_windows": 0,
                    "kept_windows": 0,
                    "filtered_total": 0,
                    "filtered_low_energy": 0,
                    "filtered_clipped": 0,
                    "filtered_static": 0,
                }
            ),
        }

    def _count_quality(self, class_name: str, bucket: str) -> None:
        self._quality_stats[bucket] += 1
        self._quality_stats["per_class"][class_name][bucket] += 1

    def scan_folders(self) -> Dict[str, List[Path]]:
        files_by_gesture: Dict[str, List[Path]] = {}
        actual_dirs = {p.name.lower(): p for p in self.data_dir.iterdir() if p.is_dir()} if self.data_dir.exists() else {}
        for gesture in self.gesture_to_idx:
            folder = actual_dirs.get(gesture.lower())
            if folder is None or not folder.exists():
                logger.warning("Gesture folder missing: %s", self.data_dir / gesture)
                files_by_gesture[gesture] = []
                continue
            csv_files = sorted(folder.glob("*.csv"))
            files_by_gesture[gesture] = csv_files
            logger.info("[%s] %d files", gesture, len(csv_files))
        return files_by_gesture

    def get_stats(self) -> Dict[str, int]:
        stats = {"total_files": 0}
        files_by_gesture = self.scan_folders()
        for gesture, files in files_by_gesture.items():
            stats[gesture] = len(files)
            stats["total_files"] += len(files)
        return stats

    def _resolve_channel_fields(self, fieldnames: Sequence[str]) -> List[str]:
        lowered = {name.lower(): name for name in fieldnames}
        emg_fields = [lowered.get(f"emg{i}") for i in range(1, 7)]
        if all(emg_fields):
            return emg_fields  # type: ignore[return-value]
        ch_fields = [lowered.get(f"ch{i}") for i in range(1, 7)]
        if all(ch_fields):
            return ch_fields  # type: ignore[return-value]
        raise ValueError(f"Unsupported CSV headers: {fieldnames}")

    def _read_csv(self, csv_path: Path) -> np.ndarray:
        rows: List[List[float]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV missing header: {csv_path}")
            channel_fields = self._resolve_channel_fields(reader.fieldnames)
            for row in reader:
                rows.append([float(row[c]) for c in channel_fields])
        if not rows:
            raise ValueError(f"CSV has no data rows: {csv_path}")
        signal = np.asarray(rows, dtype=np.float32)
        if signal.min() >= 0.0 and signal.max() > 64.0:
            signal = signal - 128.0
        return signal

    def _extract_recording_meta(self, csv_path: Path, gesture: str) -> Dict[str, str]:
        rel = csv_path.relative_to(self.data_dir).as_posix()
        stem = csv_path.stem

        session_match = re.search(r"(?:session|sess|s)[-_]?(\d+)", stem, re.IGNORECASE)
        date_match = re.search(r"(\d{8}(?:[_-]?\d{6})?)", stem)
        device_match = re.search(r"(?:dev|device)[-_]?([a-zA-Z0-9]+)", stem, re.IGNORECASE)
        user_match = re.search(r"(?:user|u)[-_]?([a-zA-Z0-9]+)", stem, re.IGNORECASE)

        session_id = f"s{session_match.group(1)}" if session_match else "s0"
        timestamp = date_match.group(1) if date_match else stem
        device_id = f"dev{device_match.group(1)}" if device_match else "unknown_device"
        user_id = f"user{user_match.group(1)}" if user_match else "unknown_user"

        return {
            "source_id": rel,
            "recording_id": stem,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "device_id": device_id,
            "gesture": gesture,
        }

    def _iter_raw_segments(self, signal: np.ndarray) -> List[np.ndarray]:
        if self._uses_dual_branch:
            return self.preprocess.extract_segments(signal)
        if signal.shape[0] > 0:
            signal = signal[::5]
        window_size = self.preprocess.get_required_window_size()
        stride = self.preprocess.get_required_window_stride()
        if signal.shape[0] < window_size:
            return []
        windower = SignalWindower(window_size=window_size, stride=stride)
        return windower.segment(signal)

    def _quality_reject_reasons(self, segment: np.ndarray) -> List[str]:
        qf = self.quality_filter
        if not qf.enabled:
            return []

        reasons: List[str] = []
        mean_abs = float(np.mean(np.abs(segment)))
        if mean_abs < qf.energy_min:
            reasons.append("low_energy")

        clip_ratio = float(np.mean(np.abs(segment) >= qf.saturation_abs))
        if clip_ratio > qf.clip_ratio_max:
            reasons.append("clipped")

        mean_std = float(np.mean(np.std(segment, axis=0)))
        if mean_std < qf.static_std_max:
            reasons.append("static")

        return reasons

    def load_all_with_sources(
        self,
        *,
        return_metadata: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        files_by_gesture = self.scan_folders()
        all_samples: List[np.ndarray] = []
        all_labels: List[int] = []
        all_sources: List[str] = []
        all_meta: List[Dict[str, Any]] = []

        self._quality_stats = self._init_quality_stats()
        self._recording_meta = {}

        for gesture, file_list in files_by_gesture.items():
            class_id = self.gesture_to_idx[gesture]
            class_samples = 0
            for csv_path in file_list:
                try:
                    signal = self._read_csv(csv_path)
                except Exception as exc:
                    logger.warning("Skip file %s: %s", csv_path.name, exc)
                    continue

                metadata = self._extract_recording_meta(csv_path, gesture)
                self._recording_meta[metadata["source_id"]] = metadata
                source_id = metadata["source_id"]

                segments = self._iter_raw_segments(signal)
                if not segments:
                    continue

                for segment in segments:
                    self._count_quality(gesture, "total_windows")
                    reasons = self._quality_reject_reasons(segment)
                    if reasons:
                        self._count_quality(gesture, "filtered_total")
                        for reason in reasons:
                            if reason == "low_energy":
                                self._count_quality(gesture, "filtered_low_energy")
                            elif reason == "clipped":
                                self._count_quality(gesture, "filtered_clipped")
                            elif reason == "static":
                                self._count_quality(gesture, "filtered_static")
                        continue

                    try:
                        feature = self.preprocess.process_window(segment)
                    except Exception as exc:
                        logger.warning("Skip segment from %s: %s", csv_path.name, exc)
                        continue

                    self._count_quality(gesture, "kept_windows")
                    all_samples.append(feature)
                    all_labels.append(class_id)
                    all_sources.append(source_id)
                    all_meta.append(dict(metadata))
                    class_samples += 1

            logger.info("[%s] loaded %d samples", gesture, class_samples)

        if not all_samples:
            raise RuntimeError("No samples loaded from dataset")

        samples = np.stack(all_samples, axis=0).astype(np.float32)
        labels = np.asarray(all_labels, dtype=np.int32)
        source_ids = np.asarray(all_sources, dtype=object)
        self._log_loaded_dataset(samples, labels)

        if return_metadata:
            return samples, labels, source_ids, all_meta
        return samples, labels, source_ids

    def _log_loaded_dataset(self, samples: np.ndarray, labels: np.ndarray) -> None:
        logger.info("Loaded dataset: %d samples, shape=%s", labels.shape[0], tuple(samples.shape))
        idx_to_gesture = {value: name for name, value in self.gesture_to_idx.items()}
        for cls, count in zip(*np.unique(labels, return_counts=True)):
            logger.info("  %s: %d", idx_to_gesture[int(cls)], int(count))

    def get_quality_report(self) -> Dict[str, Any]:
        per_class_stats = {
            cls: dict(stats)
            for cls, stats in self._quality_stats["per_class"].items()
        }

        recording_counts = defaultdict(set)
        sessions = set()
        users = set()
        devices = set()
        for meta in self._recording_meta.values():
            recording_counts[meta["gesture"]].add(meta["recording_id"])
            sessions.add(meta["session_id"])
            users.add(meta["user_id"])
            devices.add(meta["device_id"])

        quota_targets = {
            "FIST": 120,
            "PINCH": 120,
            "YE": 120,
            "SIDEGRIP": 120,
            "RELAX": 90,
            "OK": 90,
        }
        quota_status = {}
        for gesture, target in quota_targets.items():
            current = len(recording_counts.get(gesture, set()))
            quota_status[gesture] = {
                "target_recordings": target,
                "current_recordings": current,
                "missing_recordings": max(0, target - current),
            }

        return {
            "quality": {k: v for k, v in self._quality_stats.items() if k != "per_class"},
            "quality_per_class": per_class_stats,
            "recording_quota": quota_status,
            "metadata_coverage": {
                "users": len(users),
                "sessions": len(sessions),
                "devices": len(devices),
                "recordings": len(self._recording_meta),
            },
        }

    def save_quality_report(self, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = self.get_quality_report()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return out
