"""Load NinaPro DB5 MAT archives and build EMG pretraining samples."""

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

from ninapro_db5.config import DB5MappingProfileConfig, DB5PretrainConfig
from shared.preprocessing import PreprocessPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DB5WindowRecord:
    feature: np.ndarray
    label: int
    source_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _AlignedWindowCandidate:
    feature: np.ndarray
    source_id: str
    metadata: dict[str, Any]
    is_rest: bool
    gesture_key: str


class DB5PretrainDatasetLoader:
    """Build EMG-only pretraining windows from NinaPro DB5 zip archives."""

    def __init__(
        self,
        data_dir: str | Path,
        config: DB5PretrainConfig,
        *,
        pretrain_mode: str = "legacy53",
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.pretrain_mode = str(pretrain_mode).strip().lower()
        if self.pretrain_mode not in {"legacy53", "aligned3"}:
            raise ValueError(f"Unsupported pretrain_mode={pretrain_mode!r}; expected legacy53 or aligned3")
        self._stft_pipeline = PreprocessPipeline(
            {
                "sampling_rate": config.feature.target_sampling_rate_hz,
                "num_channels": config.feature.first_myo_channel_count,
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
        self._mapping_profile_report: dict[str, Any] = {
            "mode": self.pretrain_mode,
            "selected_profile": None,
            "candidate_profiles": [],
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

    def _select_windows(self, segment_signal: np.ndarray, *, allow_many: int, local_label: int) -> list[np.ndarray]:
        window = int(self.config.feature.source_window_samples)
        step = int(self.config.feature.source_step_samples)
        if segment_signal.shape[0] < window:
            return []
        starts = list(range(0, segment_signal.shape[0] - window + 1, step))
        if not starts:
            starts = [0]
        windows = [self._resample_window(segment_signal[start : start + window]) for start in starts]
        if not windows:
            return []

        if self.pretrain_mode != "aligned3":
            if len(windows) > allow_many:
                positions = np.linspace(0, len(windows) - 1, allow_many, dtype=int).tolist()
                windows = [windows[pos] for pos in positions]
            return windows

        policy = self.config.aligned3.onset_window_policy
        if local_label == 0:
            top_k = min(int(policy.rest_top_k_per_segment), allow_many, len(windows))
            mode = str(policy.rest_selection).strip().lower()
        else:
            top_k = min(int(policy.action_top_k_per_segment), allow_many, len(windows))
            mode = str(policy.action_selection).strip().lower()
        if top_k <= 0:
            return []

        if mode in {"peak_energy", "high_energy"}:
            order = sorted(range(len(windows)), key=lambda idx: self._window_energy(windows[idx]), reverse=True)
        elif mode in {"low_energy", "idle_energy"}:
            order = sorted(range(len(windows)), key=lambda idx: self._window_energy(windows[idx]))
        else:
            order = list(range(len(windows)))
        selected = [windows[idx] for idx in order[:top_k]]
        return selected

    @staticmethod
    def _gesture_key(exercise: int, local_label: int) -> str:
        return f"E{int(exercise)}_G{int(local_label):02d}"

    def _global_label_legacy(self, exercise: int, local_label: int) -> tuple[int, str]:
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

    def _iter_raw_window_candidates(self) -> Iterator[_AlignedWindowCandidate]:
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
                    allow_many = int(
                        self.config.feature.max_rest_windows_per_segment
                        if local_label == 0
                        else self.config.feature.max_windows_per_segment
                    )
                    windows = self._select_windows(segment, allow_many=allow_many, local_label=int(local_label))
                    if not windows:
                        continue
                    gesture_key = self._gesture_key(exercise, local_label)
                    for win_idx, window in enumerate(windows):
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
                            "gesture_key": gesture_key,
                            "repetition": repetition,
                        }
                        yield _AlignedWindowCandidate(
                            feature=feature,
                            source_id=source_id,
                            metadata=metadata,
                            is_rest=int(local_label) == 0,
                            gesture_key=gesture_key,
                        )
                    segment_index += 1

    @staticmethod
    def _profile_counts(
        profile: DB5MappingProfileConfig,
        action_counts: Counter[str],
        rest_count: int,
        *,
        include_rest_class: bool,
    ) -> dict[str, int]:
        counts = {
            "FIST": int(sum(action_counts.get(key, 0) for key in profile.fist)),
            "PINCH": int(sum(action_counts.get(key, 0) for key in profile.pinch)),
        }
        if include_rest_class:
            counts["REST"] = int(rest_count)
        return counts

    def _select_aligned_profile(self, action_counts: Counter[str], rest_count: int) -> DB5MappingProfileConfig:
        aligned = self.config.aligned3
        min_samples = int(aligned.min_samples_per_class)
        required = ["FIST", "PINCH"] + (["REST"] if self.config.include_rest_class else [])
        profile_reports: list[dict[str, Any]] = []

        for profile in aligned.candidate_mapping_profiles:
            counts = self._profile_counts(
                profile,
                action_counts,
                rest_count,
                include_rest_class=self.config.include_rest_class,
            )
            min_required = min(counts.get(name, 0) for name in required)
            report = {
                "name": profile.name,
                "fist": list(profile.fist),
                "pinch": list(profile.pinch),
                "counts": counts,
                "min_required_count": int(min_required),
                "eligible": bool(min_required >= min_samples),
            }
            profile_reports.append(report)

        selected: DB5MappingProfileConfig | None = None
        if aligned.mapping_override is not None:
            selected = aligned.mapping_override
            selected_counts = self._profile_counts(
                selected,
                action_counts,
                rest_count,
                include_rest_class=self.config.include_rest_class,
            )
            min_required = min(selected_counts.get(name, 0) for name in required)
            if min_required < min_samples:
                raise RuntimeError(
                    f"mapping_override does not satisfy min_samples_per_class={min_samples}: "
                    f"counts={selected_counts}"
                )
            profile_reports.append(
                {
                    "name": selected.name or "mapping_override",
                    "fist": list(selected.fist),
                    "pinch": list(selected.pinch),
                    "counts": selected_counts,
                    "min_required_count": int(min_required),
                    "eligible": True,
                    "override": True,
                }
            )
        else:
            eligible = [item for item in profile_reports if item["eligible"]]
            if eligible:
                best = max(
                    eligible,
                    key=lambda item: (
                        int(item["min_required_count"]),
                        int(item["counts"].get("FIST", 0) + item["counts"].get("PINCH", 0)),
                    ),
                )
                selected = next(profile for profile in aligned.candidate_mapping_profiles if profile.name == best["name"])

        self._mapping_profile_report = {
            "mode": "aligned3",
            "min_samples_per_class": min_samples,
            "required_classes": required,
            "candidate_profiles": profile_reports,
            "available_action_keys": dict(sorted(action_counts.items())),
            "rest_count": int(rest_count),
            "selected_profile": selected.name if selected is not None else None,
            "override_used": aligned.mapping_override is not None,
        }

        if selected is None:
            raise RuntimeError(
                "No aligned3 mapping profile satisfies min sample coverage. "
                f"Required={required} min={min_samples} report={profile_reports}"
            )
        return selected

    def _iter_records_legacy(self) -> Iterator[DB5WindowRecord]:
        self._mapping_profile_report = {
            "mode": "legacy53",
            "selected_profile": None,
            "candidate_profiles": [],
        }
        for candidate in self._iter_raw_window_candidates():
            exercise = int(candidate.metadata["exercise"])
            local_label = int(candidate.metadata["local_label"])
            global_label, class_name = self._global_label_legacy(exercise, local_label)
            self._class_name_by_label[global_label] = class_name
            metadata = dict(candidate.metadata)
            metadata["global_label"] = int(global_label)
            metadata["class_name"] = class_name
            yield DB5WindowRecord(
                feature=candidate.feature,
                label=int(global_label),
                source_id=candidate.source_id,
                metadata=metadata,
            )

    def _iter_records_aligned(self) -> Iterator[DB5WindowRecord]:
        candidates = list(self._iter_raw_window_candidates())
        action_counts: Counter[str] = Counter()
        rest_count = 0
        for candidate in candidates:
            if candidate.is_rest:
                rest_count += 1
            else:
                action_counts[candidate.gesture_key] += 1
        selected_profile = self._select_aligned_profile(action_counts, rest_count)
        fist_keys = set(selected_profile.fist)
        pinch_keys = set(selected_profile.pinch)

        rest_label = 0
        fist_label = 1 if self.config.include_rest_class else 0
        pinch_label = fist_label + 1
        self._class_name_by_label = {
            fist_label: "FIST",
            pinch_label: "PINCH",
        }
        if self.config.include_rest_class:
            self._class_name_by_label[rest_label] = "REST"
        self._mapping_profile_report["selected_counts"] = self._profile_counts(
            selected_profile,
            action_counts,
            rest_count,
            include_rest_class=self.config.include_rest_class,
        )

        for candidate in candidates:
            metadata = dict(candidate.metadata)
            if candidate.is_rest:
                if not self.config.include_rest_class:
                    continue
                label = rest_label
                class_name = "REST"
            elif candidate.gesture_key in fist_keys:
                label = fist_label
                class_name = "FIST"
            elif candidate.gesture_key in pinch_keys:
                label = pinch_label
                class_name = "PINCH"
            else:
                continue
            metadata["global_label"] = int(label)
            metadata["class_name"] = class_name
            metadata["aligned_profile"] = str(self._mapping_profile_report.get("selected_profile") or "")
            yield DB5WindowRecord(
                feature=candidate.feature,
                label=int(label),
                source_id=candidate.source_id,
                metadata=metadata,
            )

    def _iter_records(self) -> Iterator[DB5WindowRecord]:
        if self.pretrain_mode == "aligned3" and bool(self.config.aligned3.enabled):
            yield from self._iter_records_aligned()
            return
        yield from self._iter_records_legacy()

    def load_all_with_sources(
        self,
        *,
        return_metadata: bool = False,
    ):
        self._class_name_by_label.clear()
        self._gesture_label_map.clear()
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
            if (
                str(name).upper() == "REST"
                and not self.config.include_rest_class
                and self.pretrain_mode != "legacy53"
            ):
                continue
            names.append(name)
        return names

    def get_mapping_profile_report(self) -> dict[str, Any]:
        return dict(self._mapping_profile_report)
