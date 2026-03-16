"""Dataset construction for event-onset training and replay."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from event_onset.config import EventDataConfig
from event_onset.manifest import load_event_manifest_rows, normalize_relative_path
from scripts.collection_utils import STANDARD_CSV_HEADERS
from shared.label_modes import get_label_mode_spec
from shared.preprocessing import PreprocessPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventWindowRecord:
    emg_feature: np.ndarray
    imu_feature: np.ndarray
    label: int
    source_id: str
    metadata: dict[str, Any]
    energy: float
    imu_motion: float


class EventClipDatasetLoader:
    """Load event clips and construct event-onset samples."""

    def __init__(
        self,
        data_dir: str | Path,
        data_config: EventDataConfig,
        recordings_manifest_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_config = data_config
        self.label_spec = get_label_mode_spec(data_config.label_mode, data_config.target_db5_keys)
        self.recordings_manifest_path = self._resolve_manifest_path(recordings_manifest_path)
        self._manifest_rows = load_event_manifest_rows(self.recordings_manifest_path) if self.recordings_manifest_path else {}
        self._quality_report: dict[str, Any] = {}
        self._stft_pipeline = PreprocessPipeline(
            {
                "sampling_rate": data_config.device_sampling_rate_hz,
                "num_channels": 8,
                "target_length": data_config.context_samples,
                "segment_length": data_config.context_samples,
                "segment_stride": data_config.window_step_samples,
                "stft_window": data_config.feature.emg_stft_window,
                "stft_hop": data_config.feature.emg_stft_hop,
                "n_fft": data_config.feature.emg_n_fft,
                "freq_bins_out": data_config.feature.emg_freq_bins,
                "normalize": "log",
                "clip_min": 0.0,
                "clip_max": 10.0,
                "dual_branch": {"enabled": False},
            }
        )

    def _resolve_manifest_path(self, recordings_manifest_path: str | Path | None) -> Path | None:
        if recordings_manifest_path is not None:
            raw = Path(recordings_manifest_path)
            candidates = [raw]
            if not raw.is_absolute():
                candidates.insert(0, self.data_dir / raw)
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
            preferred = candidates[0]
            return preferred.resolve() if preferred.is_absolute() else preferred

        default_path = self.data_dir / self.data_config.recordings_manifest_path
        if default_path.exists():
            return default_path.resolve()
        return default_path

    def _iter_clip_rows(self) -> Iterator[tuple[Path, dict[str, str]]]:
        if self.recordings_manifest_path and self.recordings_manifest_path.exists():
            self._manifest_rows = load_event_manifest_rows(self.recordings_manifest_path)
        if not self._manifest_rows:
            raise FileNotFoundError("Event-onset dataset requires a recordings manifest with event metadata.")
        allowed_targets = set(self.label_spec.gesture_to_idx.keys())
        for metadata in sorted(self._manifest_rows.values(), key=lambda row: row["relative_path"]):
            if metadata.get("capture_mode", "") != self.data_config.capture_mode_filter:
                continue
            target_state = str(metadata.get("target_state", "")).strip().upper()
            if target_state not in allowed_targets:
                logger.warning("Skip clip with target_state=%s not in configured label set.", target_state)
                continue
            csv_path = self.data_dir / metadata["relative_path"]
            if not csv_path.exists():
                logger.warning("Manifest file missing on disk: %s", csv_path)
                continue
            yield csv_path, dict(metadata)

    @staticmethod
    def _read_csv_matrix(csv_path: Path) -> np.ndarray:
        rows: list[list[float]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV missing header: {csv_path}")
            missing = [field for field in STANDARD_CSV_HEADERS if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing standardized columns {missing}: {csv_path}")
            for row in reader:
                rows.append([float(row[field]) for field in STANDARD_CSV_HEADERS])
        if not rows:
            raise ValueError(f"CSV has no rows: {csv_path}")
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix[:, :8].min(initial=0.0) >= 0.0 and matrix[:, :8].max(initial=0.0) > 64.0:
            matrix[:, :8] -= 128.0
        return matrix

    @staticmethod
    def _clip_ratio(emg_window: np.ndarray, saturation_abs: float) -> float:
        return float(np.mean(np.abs(emg_window) >= float(saturation_abs)))

    @staticmethod
    def _window_energy(emg_window: np.ndarray) -> float:
        return float(np.mean(np.abs(emg_window)))

    @staticmethod
    def _window_std(emg_window: np.ndarray) -> float:
        return float(np.mean(np.std(emg_window, axis=0)))

    @staticmethod
    def _imu_motion_score(imu_window: np.ndarray) -> float:
        if imu_window.shape[0] <= 1:
            return 0.0
        return float(np.mean(np.std(imu_window, axis=0)))

    def _iter_window_slices(self, matrix: np.ndarray) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        context = self.data_config.context_samples
        step = self.data_config.window_step_samples
        for end in range(context, matrix.shape[0] + 1, step):
            start = end - context
            window = matrix[start:end]
            yield end, window[:, :8], window[:, 8:14]

    def _ms_to_samples(self, value_ms: int) -> int:
        rate = float(self.data_config.device_sampling_rate_hz)
        return max(1, int(round(float(value_ms) * rate / 1000.0)))

    @staticmethod
    def _smooth_signal(values: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1 or values.size <= 1:
            return values.astype(np.float32)
        kernel_size = int(max(1, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
        return np.convolve(values.astype(np.float32), kernel, mode="same").astype(np.float32)

    def _detect_action_onsets(self, emg_matrix: np.ndarray) -> list[int]:
        if emg_matrix.size == 0:
            return []
        envelope = np.mean(np.abs(emg_matrix), axis=1).astype(np.float32)
        smoothed = self._smooth_signal(envelope, kernel_size=self._ms_to_samples(40))
        baseline = float(np.percentile(smoothed, 20))
        peak = float(np.max(smoothed))
        alpha = float(self.data_config.action_onset_threshold_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        threshold = baseline + alpha * (peak - baseline)
        min_gap = self._ms_to_samples(int(self.data_config.action_onset_min_gap_ms))
        candidates = np.where(smoothed >= threshold)[0].astype(np.int32).tolist()
        if not candidates:
            return []

        onsets: list[int] = []
        last = -10**9
        for idx in candidates:
            if idx - last < min_gap:
                continue
            lo = max(0, idx - min_gap // 2)
            hi = min(smoothed.shape[0], idx + min_gap // 2 + 1)
            local_peak = int(lo + int(np.argmax(smoothed[lo:hi])))
            if onsets and local_peak - onsets[-1] < min_gap:
                if smoothed[local_peak] > smoothed[onsets[-1]]:
                    onsets[-1] = local_peak
                continue
            onsets.append(local_peak)
            last = local_peak
        return sorted(set(int(v) for v in onsets))

    @staticmethod
    def _window_center(start: int, end: int) -> int:
        return int((int(start) + int(end)) // 2)

    def _resample_imu(self, imu_window: np.ndarray) -> np.ndarray:
        target_steps = int(self.data_config.feature.imu_resample_steps)
        if imu_window.shape[0] == target_steps:
            resampled = imu_window
        else:
            src = np.linspace(0.0, 1.0, imu_window.shape[0], dtype=np.float32)
            dst = np.linspace(0.0, 1.0, target_steps, dtype=np.float32)
            resampled = np.empty((target_steps, imu_window.shape[1]), dtype=np.float32)
            for channel_idx in range(imu_window.shape[1]):
                resampled[:, channel_idx] = np.interp(dst, src, imu_window[:, channel_idx]).astype(np.float32)
        centered = resampled - np.mean(resampled, axis=0, keepdims=True)
        std = np.std(centered, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (centered / std).T.astype(np.float32)

    def _build_event_windows(
        self,
        matrix: np.ndarray,
        metadata: dict[str, str],
    ) -> tuple[list[EventWindowRecord], dict[str, Any]]:
        qf = self.data_config.quality_filter
        target_state = str(metadata["target_state"]).strip().upper()
        if target_state not in self.label_spec.gesture_to_idx:
            raise ValueError(f"target_state={target_state!r} is not in configured label set {self.label_spec.class_names}")
        label = int(self.label_spec.gesture_to_idx[target_state])
        source_id = normalize_relative_path(metadata["relative_path"])
        is_relax_clip = target_state == "RELAX"
        relax_candidates: list[EventWindowRecord] = []
        scored_action_windows: list[EventWindowRecord] = []
        action_policy = str(self.data_config.action_window_policy).strip().lower()
        onset_indices: list[int] = []
        onset_pre = self._ms_to_samples(int(self.data_config.action_onset_pre_ms))
        onset_post = self._ms_to_samples(int(self.data_config.action_onset_post_ms))
        if not is_relax_clip and action_policy == "onset_peak":
            onset_indices = self._detect_action_onsets(matrix[:, :8])
            if onset_indices:
                # Single-action clips: keep strongest onset first for stable slicing.
                onset_indices = [int(onset_indices[0])]

        clip_diag: dict[str, Any] = {
            "relative_path": source_id,
            "target_state": target_state,
            "policy": action_policy if not is_relax_clip else "relax_idle",
            "onset_indices": list(onset_indices),
            "selected_window_ranges": [],
            "dropped_reasons": [],
            "selected_count": 0,
        }

        for end, emg_window, imu_window in self._iter_window_slices(matrix):
            start = int(end - self.data_config.context_samples)
            center = self._window_center(start, int(end))
            onset_distance = 10**9
            if not is_relax_clip and action_policy == "onset_peak":
                if not onset_indices:
                    continue
                in_region = any((onset - onset_pre) <= center <= (onset + onset_post) for onset in onset_indices)
                if not in_region:
                    continue
                onset_distance = min(abs(center - int(onset)) for onset in onset_indices)
            energy = self._window_energy(emg_window)
            clip_ratio = self._clip_ratio(emg_window, qf.saturation_abs)
            mean_std = self._window_std(emg_window)
            imu_motion = self._imu_motion_score(imu_window)
            if clip_ratio > float(qf.clip_ratio_max):
                continue

            emg_feature = self._stft_pipeline.process_window(emg_window)
            imu_feature = self._resample_imu(imu_window)
            merged_metadata = dict(metadata)
            merged_metadata["window_end_index"] = end
            merged_metadata["window_start_index"] = int(start)
            merged_metadata["window_energy"] = energy
            merged_metadata["imu_motion"] = imu_motion
            merged_metadata["window_center_index"] = int(center)
            merged_metadata["onset_distance"] = int(onset_distance)
            if onset_indices:
                merged_metadata["onset_idx"] = int(onset_indices[0])

            if is_relax_clip:
                if energy <= float(qf.energy_min) and imu_motion <= float(self.data_config.feature.imu_motion_std_max):
                    relax_candidates.append(
                        EventWindowRecord(
                            emg_feature=emg_feature,
                            imu_feature=imu_feature,
                            label=0,
                            source_id=source_id,
                            metadata=merged_metadata,
                            energy=energy,
                            imu_motion=imu_motion,
                        )
                    )
                continue

            if energy < float(qf.energy_min):
                continue
            if mean_std < float(qf.static_std_max):
                continue
            scored_action_windows.append(
                EventWindowRecord(
                    emg_feature=emg_feature,
                    imu_feature=imu_feature,
                    label=label,
                    source_id=source_id,
                    metadata=merged_metadata,
                    energy=energy,
                    imu_motion=imu_motion,
                )
            )

        if is_relax_clip:
            relax_candidates.sort(key=lambda item: (item.energy, item.imu_motion, int(item.metadata["window_end_index"])))
            chosen = relax_candidates[: int(self.data_config.idle_top_k_windows_per_clip)]
            for entry in chosen:
                entry.metadata["selection_mode"] = "idle_relax"
                clip_diag["selected_window_ranges"].append(
                    [int(entry.metadata["window_start_index"]), int(entry.metadata["window_end_index"])]
                )
            clip_diag["selected_count"] = int(len(chosen))
            if not chosen:
                clip_diag["dropped_reasons"].append("no_relax_window_after_filter")
            return chosen, clip_diag

        candidate_windows = list(scored_action_windows)
        if action_policy == "onset_peak" and candidate_windows:
            peak_energy = float(max(item.energy for item in candidate_windows))
            ratio = float(min(max(self.data_config.action_onset_energy_ratio_min, 0.0), 1.0))
            min_energy = max(float(qf.energy_min), peak_energy * ratio)
            filtered = [item for item in candidate_windows if float(item.energy) >= float(min_energy)]
            if filtered:
                candidate_windows = filtered
            else:
                clip_diag["dropped_reasons"].append("all_onset_windows_below_energy_ratio")
            candidate_windows.sort(
                key=lambda item: (
                    int(item.metadata.get("onset_distance", 10**9)),
                    -float(item.energy),
                    int(item.metadata.get("window_end_index", 0)),
                )
            )
        else:
            candidate_windows.sort(key=lambda item: item.energy, reverse=True)

        selected = candidate_windows[: int(self.data_config.top_k_windows_per_clip)]
        for entry in selected:
            entry.metadata["selection_mode"] = (
                "onset_peak_distance_energy" if action_policy == "onset_peak" else "top_k_energy"
            )
            clip_diag["selected_window_ranges"].append(
                [int(entry.metadata["window_start_index"]), int(entry.metadata["window_end_index"])]
            )
        clip_diag["selected_count"] = int(len(selected))
        if action_policy == "onset_peak" and not onset_indices:
            clip_diag["dropped_reasons"].append("no_onset_detected")
        if not selected and onset_indices:
            clip_diag["dropped_reasons"].append("no_action_window_after_filter")
        return selected, clip_diag

    def load_all_with_sources(
        self,
        *,
        return_metadata: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        emg_features: list[np.ndarray] = []
        imu_features: list[np.ndarray] = []
        labels: list[int] = []
        source_ids: list[str] = []
        metadata_rows: list[dict[str, Any]] = []
        clip_diagnostics: list[dict[str, Any]] = []

        clip_stats: dict[str, int] = Counter()
        window_stats = defaultdict(int)
        for csv_path, metadata in self._iter_clip_rows():
            matrix = self._read_csv_matrix(csv_path)
            selected, clip_diag = self._build_event_windows(matrix, metadata)
            clip_stats[metadata["target_state"]] += 1
            window_stats["clips_total"] += 1
            window_stats["selected_windows"] += len(selected)
            clip_diagnostics.append(dict(clip_diag))
            if not selected:
                window_stats["filtered_clips"] += 1
                continue
            for item in selected:
                emg_features.append(item.emg_feature)
                imu_features.append(item.imu_feature)
                labels.append(int(item.label))
                source_ids.append(item.source_id)
                metadata_rows.append(dict(item.metadata))
                window_stats[f"label_{self.label_spec.class_names[int(item.label)]}"] += 1

        if not emg_features:
            raise RuntimeError("No event-onset samples were selected from the dataset.")

        self._quality_report = {
            "clip_counts": dict(clip_stats),
            "window_counts": dict(window_stats),
            "action_window_policy": str(self.data_config.action_window_policy),
            "clip_diagnostics": clip_diagnostics,
            "manifest_path": str(self.recordings_manifest_path) if self.recordings_manifest_path else None,
        }
        emg_array = np.stack(emg_features, axis=0).astype(np.float32)
        imu_array = np.stack(imu_features, axis=0).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        source_array = np.asarray(source_ids, dtype=object)
        if return_metadata:
            return emg_array, imu_array, label_array, source_array, metadata_rows
        return emg_array, imu_array, label_array, source_array

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = Counter()
        for _, metadata in self._iter_clip_rows():
            stats[metadata["target_state"]] += 1
            stats["total_clips"] += 1
        return dict(stats)

    def get_quality_report(self) -> dict[str, Any]:
        return dict(self._quality_report)

    def save_quality_report(self, path: str | Path) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as handle:
            json.dump(self._quality_report, handle, ensure_ascii=False, indent=2)
        return resolved

    def iter_clips(self) -> Iterator[tuple[str, str, np.ndarray, dict[str, str]]]:
        for csv_path, metadata in self._iter_clip_rows():
            matrix = self._read_csv_matrix(csv_path)
            yield metadata["start_state"], metadata["target_state"], matrix, dict(metadata)
