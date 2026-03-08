"""Dataset split utilities with portable manifest persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

SPLIT_MODES = ("legacy", "grouped_file")
MANIFEST_STRATEGIES = ("v1", "v2")


@dataclass
class SplitManifest:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    train_sources: list[str]
    val_sources: list[str]
    test_sources: list[str]
    seed: int
    split_mode: str
    manifest_strategy: str = "v1"
    num_samples: int = 0
    val_ratio: float = 0.0
    test_ratio: float = 0.0
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    group_keys_train: list[str] = field(default_factory=list)
    group_keys_val: list[str] = field(default_factory=list)
    group_keys_test: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SplitManifest":
        defaults = {
            "manifest_strategy": "v1",
            "group_keys_train": [],
            "group_keys_val": [],
            "group_keys_test": [],
        }
        payload = {**defaults, **data}
        return cls(**payload)


def _distribute_groups_per_class(
    labels: np.ndarray,
    groups: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_id in np.unique(labels):
        class_mask = labels == class_id
        class_groups = np.unique(groups[class_mask])
        rng.shuffle(class_groups)

        n_groups = len(class_groups)
        n_val = int(round(n_groups * val_ratio))
        n_test = int(round(n_groups * test_ratio))
        n_val = min(n_val, max(0, n_groups - 1))
        n_test = min(n_test, max(0, n_groups - n_val - 1))

        val_groups = set(class_groups[:n_val])
        test_groups = set(class_groups[n_val : n_val + n_test])
        train_groups = set(class_groups[n_val + n_test :])

        if not train_groups and class_groups.size > 0:
            fallback_group = class_groups[-1]
            train_groups.add(fallback_group)
            val_groups.discard(fallback_group)
            test_groups.discard(fallback_group)

        class_indices = np.where(class_mask)[0]
        for idx in class_indices:
            g = groups[idx]
            if g in val_groups:
                val_idx.append(int(idx))
            elif g in test_groups:
                test_idx.append(int(idx))
            else:
                train_idx.append(int(idx))

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


def _extract_v2_group_key(source_id: str, metadata: Optional[Dict]) -> str:
    if metadata:
        recording = str(metadata.get("recording_id") or metadata.get("source_id") or source_id)
        session = str(metadata.get("session_id") or "default_session")
        user = str(metadata.get("user_id") or "unknown_user")
        return f"{user}::{session}::{recording}"

    # Fallback parser for legacy source strings carrying key=value pieces.
    parts = source_id.split("|")
    data: Dict[str, str] = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            data[k.strip()] = v.strip()
    recording = data.get("recording_id", source_id)
    session = data.get("session_id", "default_session")
    user = data.get("user_id", "unknown_user")
    return f"{user}::{session}::{recording}"


def build_manifest(
    labels: Sequence[int],
    source_ids: Sequence[str],
    *,
    seed: int,
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    manifest_strategy: str = "v1",
    source_metadata: Optional[Sequence[Optional[Dict]]] = None,
    data_dir: Optional[str] = None,
) -> SplitManifest:
    del data_dir
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unsupported split_mode: {split_mode}, expected one of {SPLIT_MODES}")
    if manifest_strategy not in MANIFEST_STRATEGIES:
        raise ValueError(
            f"Unsupported manifest_strategy: {manifest_strategy}, expected one of {MANIFEST_STRATEGIES}"
        )

    labels_arr = np.asarray(labels, dtype=int)
    sources_arr = np.asarray(source_ids, dtype=object)
    if labels_arr.shape[0] != sources_arr.shape[0]:
        raise ValueError("labels/source_ids length mismatch")
    if num_classes is None:
        num_classes = int(labels_arr.max()) + 1 if labels_arr.size else 0
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    class_names = list(class_names)

    if source_metadata is not None and len(source_metadata) != len(labels_arr):
        raise ValueError("source_metadata length mismatch")

    if split_mode == "legacy":
        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(labels_arr))
        train_idx, val_idx, test_idx = _legacy_split(all_indices, val_ratio, test_ratio, rng)
        group_keys = np.array(sources_arr, dtype=object)
    else:
        if manifest_strategy == "v2":
            if source_metadata is None:
                source_metadata = [None] * len(sources_arr)
            group_keys = np.array(
                [_extract_v2_group_key(str(sid), meta) for sid, meta in zip(sources_arr, source_metadata)],
                dtype=object,
            )
        else:
            group_keys = np.array(sources_arr, dtype=object)

        train_idx, val_idx, test_idx = _distribute_groups_per_class(
            labels_arr, group_keys, val_ratio, test_ratio, seed
        )

    manifest = SplitManifest(
        train_indices=train_idx.tolist(),
        val_indices=val_idx.tolist(),
        test_indices=test_idx.tolist(),
        train_sources=sorted(set(sources_arr[train_idx].tolist())),
        val_sources=sorted(set(sources_arr[val_idx].tolist())),
        test_sources=sorted(set(sources_arr[test_idx].tolist())),
        group_keys_train=sorted(set(group_keys[train_idx].tolist())),
        group_keys_val=sorted(set(group_keys[val_idx].tolist())),
        group_keys_test=sorted(set(group_keys[test_idx].tolist())),
        seed=seed,
        split_mode=split_mode,
        manifest_strategy=manifest_strategy,
        num_samples=int(len(labels_arr)),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        class_distribution=_calc_class_distribution(
            labels_arr, train_idx, val_idx, test_idx, num_classes, class_names
        ),
    )
    validate_manifest(manifest)
    return manifest


def _legacy_split(
    all_indices: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = all_indices.copy()
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    test_idx = shuffled[:n_test]
    val_idx = shuffled[n_test : n_test + n_val]
    train_idx = shuffled[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _calc_class_distribution(
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    dist: Dict[str, Dict[str, int]] = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        split_labels = labels[idx]
        counts = np.bincount(split_labels, minlength=num_classes)
        dist[split_name] = {class_names[i]: int(counts[i]) for i in range(num_classes)}
    return dist


def validate_manifest(manifest: SplitManifest) -> None:
    train_set = set(manifest.train_indices)
    val_set = set(manifest.val_indices)
    test_set = set(manifest.test_indices)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Split indices overlap between train/val/test")

    # In manifest v2, the authoritative anti-leakage identity is the metadata-derived
    # group key (`user::session::recording`). Raw source ids remain for file lookup and
    # backwards compatibility, but may legitimately collide in synthetic or migrated data.
    if manifest.manifest_strategy != "v2":
        src_train, src_val, src_test = set(manifest.train_sources), set(manifest.val_sources), set(manifest.test_sources)
        if src_train & src_val or src_train & src_test or src_val & src_test:
            raise ValueError("Source leakage detected between train/val/test")

    grp_train = set(manifest.group_keys_train)
    grp_val = set(manifest.group_keys_val)
    grp_test = set(manifest.group_keys_test)
    if grp_train & grp_val or grp_train & grp_test or grp_val & grp_test:
        raise ValueError("Group leakage detected between train/val/test")


def save_manifest(manifest: SplitManifest, out_path: str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def load_manifest(in_path: str) -> SplitManifest:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = SplitManifest.from_dict(data)
    validate_manifest(manifest)
    return manifest


def split_arrays_from_manifest(
    samples: np.ndarray,
    labels: np.ndarray,
    manifest: SplitManifest,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    train_idx = np.asarray(manifest.train_indices, dtype=np.int32)
    val_idx = np.asarray(manifest.val_indices, dtype=np.int32)
    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    return (
        (samples[train_idx], labels[train_idx]),
        (samples[val_idx], labels[val_idx]),
        (samples[test_idx], labels[test_idx]),
    )


def split_and_optionally_augment(
    *,
    samples: np.ndarray,
    labels: np.ndarray,
    manifest: SplitManifest,
    augmentor=None,
    augment_factor: int = 1,
    use_mixup: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (train_samples, train_labels), (val_samples, val_labels), (test_samples, test_labels) = split_arrays_from_manifest(
        samples,
        labels,
        manifest,
    )
    if augmentor is not None and augment_factor > 1:
        train_samples, train_labels = augmentor.augment_batch(
            train_samples,
            train_labels,
            factor=augment_factor,
            use_mixup=use_mixup,
        )
    return (
        (train_samples, train_labels),
        (val_samples, val_labels),
        (test_samples, test_labels),
    )


def legacy_kfold_indices(
    *,
    labels: np.ndarray,
    base_indices: np.ndarray,
    k: int,
    seed: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    del labels
    if k <= 1:
        raise ValueError("k must be > 1")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(base_indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)
    for fold_idx in range(k):
        val_idx = np.asarray(folds[fold_idx], dtype=np.int64)
        train_parts = [np.asarray(folds[i], dtype=np.int64) for i in range(k) if i != fold_idx]
        train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
        yield fold_idx, train_idx, val_idx


def grouped_kfold_indices(
    *,
    labels: np.ndarray,
    source_ids: np.ndarray,
    base_indices: np.ndarray,
    k: int,
    seed: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    del labels
    if k <= 1:
        raise ValueError("k must be > 1")
    rng = np.random.default_rng(seed)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    base_sources = np.asarray(source_ids, dtype=object)[base_indices]
    unique_groups = np.unique(base_sources)
    rng.shuffle(unique_groups)
    group_folds = np.array_split(unique_groups, k)
    for fold_idx in range(k):
        val_groups = set(group_folds[fold_idx].tolist())
        mask = np.array([src in val_groups for src in base_sources], dtype=bool)
        yield fold_idx, base_indices[~mask], base_indices[mask]
