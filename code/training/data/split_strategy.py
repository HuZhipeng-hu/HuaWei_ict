"""
Dataset split strategy layer.

This module isolates split logic from training orchestration so split policies
can evolve without rewriting training loops.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from shared.gestures import NUM_CLASSES

logger = logging.getLogger(__name__)

SPLIT_MODES = ("legacy", "grouped_file")
MANIFEST_VERSION = 1


@dataclass
class SplitManifest:
    """Serializable split manifest for reproducible train/val/test evaluation."""

    version: int = MANIFEST_VERSION
    split_mode: str = "grouped_file"
    seed: int = 42
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    data_dir: str = ""
    num_samples: int = 0
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)
    train_sources: List[str] = field(default_factory=list)
    val_sources: List[str] = field(default_factory=list)
    test_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict) -> "SplitManifest":
        return cls(
            version=int(raw.get("version", MANIFEST_VERSION)),
            split_mode=str(raw.get("split_mode", "grouped_file")),
            seed=int(raw.get("seed", 42)),
            val_ratio=float(raw.get("val_ratio", 0.2)),
            test_ratio=float(raw.get("test_ratio", 0.2)),
            data_dir=str(raw.get("data_dir", "")),
            num_samples=int(raw.get("num_samples", 0)),
            train_indices=[int(x) for x in raw.get("train_indices", [])],
            val_indices=[int(x) for x in raw.get("val_indices", [])],
            test_indices=[int(x) for x in raw.get("test_indices", [])],
            train_sources=[str(x) for x in raw.get("train_sources", [])],
            val_sources=[str(x) for x in raw.get("val_sources", [])],
            test_sources=[str(x) for x in raw.get("test_sources", [])],
        )


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Saved split manifest: %s", out_path)


def load_manifest(path: str | Path) -> SplitManifest:
    in_path = Path(path)
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    manifest = SplitManifest.from_dict(raw)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: SplitManifest) -> None:
    if manifest.split_mode not in SPLIT_MODES:
        raise ValueError(
            f"Unsupported split_mode={manifest.split_mode!r}, expected one of {SPLIT_MODES}"
        )

    train_set = set(manifest.train_indices)
    val_set = set(manifest.val_indices)
    test_set = set(manifest.test_indices)

    if train_set & val_set:
        raise ValueError("train_indices and val_indices overlap")
    if train_set & test_set:
        raise ValueError("train_indices and test_indices overlap")
    if val_set & test_set:
        raise ValueError("val_indices and test_indices overlap")

    if manifest.num_samples > 0:
        all_idx = train_set | val_set | test_set
        if all_idx and max(all_idx) >= manifest.num_samples:
            raise ValueError("Manifest index out of range")


def _safe_split_counts(n_total: int, val_ratio: float, test_ratio: float) -> Tuple[int, int]:
    """
    Return (n_val, n_test) while keeping at least one training unit when possible.
    """
    if n_total <= 1:
        return 0, 0

    n_test = int(round(n_total * test_ratio))
    n_test = max(0, min(n_test, n_total - 1))

    remaining = n_total - n_test
    if remaining <= 1:
        return 0, n_test

    n_val = int(round(n_total * val_ratio))
    n_val = max(0, min(n_val, remaining - 1))
    return n_val, n_test


def _indices_from_sources(
    source_ids: np.ndarray, chosen_sources: Sequence[str]
) -> np.ndarray:
    if not chosen_sources:
        return np.array([], dtype=np.int64)
    mask = np.isin(source_ids, np.array(list(chosen_sources), dtype=object))
    return np.where(mask)[0].astype(np.int64)


def _build_grouped_file_indices(
    labels: np.ndarray,
    source_ids: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    rng = np.random.RandomState(seed)
    train_sources: List[str] = []
    val_sources: List[str] = []
    test_sources: List[str] = []

    for class_id in range(NUM_CLASSES):
        class_mask = labels == class_id
        class_sources = np.unique(source_ids[class_mask]).astype(object)
        class_sources = class_sources.tolist()
        rng.shuffle(class_sources)

        n_val, n_test = _safe_split_counts(len(class_sources), val_ratio, test_ratio)
        test_part = class_sources[:n_test]
        val_part = class_sources[n_test : n_test + n_val]
        train_part = class_sources[n_test + n_val :]

        # Ensure every class keeps at least one train source if possible.
        if not train_part and class_sources:
            if val_part:
                train_part.append(val_part.pop())
            elif test_part:
                train_part.append(test_part.pop())

        train_sources.extend(train_part)
        val_sources.extend(val_part)
        test_sources.extend(test_part)

    train_idx = _indices_from_sources(source_ids, train_sources)
    val_idx = _indices_from_sources(source_ids, val_sources)
    test_idx = _indices_from_sources(source_ids, test_sources)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx, sorted(set(train_sources)), sorted(set(val_sources)), sorted(set(test_sources))


def _build_legacy_indices(
    labels: np.ndarray,
    source_ids: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)

        n_val, n_test = _safe_split_counts(len(class_indices), val_ratio, test_ratio)
        test_part = class_indices[:n_test]
        val_part = class_indices[n_test : n_test + n_val]
        train_part = class_indices[n_test + n_val :]

        if len(train_part) == 0 and len(class_indices) > 0:
            if len(val_part) > 0:
                train_part = np.array([val_part[-1]], dtype=np.int64)
                val_part = val_part[:-1]
            elif len(test_part) > 0:
                train_part = np.array([test_part[-1]], dtype=np.int64)
                test_part = test_part[:-1]

        train_idx.extend(train_part.tolist())
        val_idx.extend(val_part.tolist())
        test_idx.extend(test_part.tolist())

    train_array = np.array(train_idx, dtype=np.int64)
    val_array = np.array(val_idx, dtype=np.int64)
    test_array = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_array)
    rng.shuffle(val_array)
    rng.shuffle(test_array)

    train_sources = sorted(set(source_ids[train_array].tolist())) if len(train_array) > 0 else []
    val_sources = sorted(set(source_ids[val_array].tolist())) if len(val_array) > 0 else []
    test_sources = sorted(set(source_ids[test_array].tolist())) if len(test_array) > 0 else []

    return train_array, val_array, test_array, train_sources, val_sources, test_sources


def build_manifest(
    labels: np.ndarray,
    source_ids: np.ndarray,
    split_mode: str = "grouped_file",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    data_dir: str = "",
) -> SplitManifest:
    """
    Build a reproducible split manifest.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unsupported split_mode={split_mode!r}")
    if len(labels) != len(source_ids):
        raise ValueError("labels and source_ids must have the same length")

    labels = np.asarray(labels, dtype=np.int32)
    source_ids = np.asarray(source_ids, dtype=object)

    if split_mode == "grouped_file":
        train_idx, val_idx, test_idx, train_sources, val_sources, test_sources = _build_grouped_file_indices(
            labels=labels,
            source_ids=source_ids,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        train_idx, val_idx, test_idx, train_sources, val_sources, test_sources = _build_legacy_indices(
            labels=labels,
            source_ids=source_ids,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    manifest = SplitManifest(
        split_mode=split_mode,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        data_dir=str(data_dir),
        num_samples=int(len(labels)),
        train_indices=train_idx.tolist(),
        val_indices=val_idx.tolist(),
        test_indices=test_idx.tolist(),
        train_sources=train_sources,
        val_sources=val_sources,
        test_sources=test_sources,
    )
    validate_manifest(manifest)
    return manifest


def split_arrays_from_manifest(
    samples: np.ndarray,
    labels: np.ndarray,
    manifest: SplitManifest,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    validate_manifest(manifest)

    train_idx = np.asarray(manifest.train_indices, dtype=np.int64)
    val_idx = np.asarray(manifest.val_indices, dtype=np.int64)
    test_idx = np.asarray(manifest.test_indices, dtype=np.int64)

    return (
        (samples[train_idx], labels[train_idx]),
        (samples[val_idx], labels[val_idx]),
        (samples[test_idx], labels[test_idx]),
    )


def split_and_optionally_augment(
    samples: np.ndarray,
    labels: np.ndarray,
    manifest: SplitManifest,
    augmentor=None,
    augment_factor: int = 1,
    use_mixup: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Materialize train/val/test splits and apply augmentation only on the train split.
    """
    train_data, val_data, test_data = split_arrays_from_manifest(samples, labels, manifest)
    train_samples, train_labels = train_data

    if augmentor is not None and (augment_factor > 1 or use_mixup):
        train_samples, train_labels = augmentor.augment_batch(
            train_samples,
            train_labels,
            factor=augment_factor,
            use_mixup=use_mixup,
        )

    return (train_samples, train_labels), val_data, test_data


def grouped_kfold_indices(
    labels: np.ndarray,
    source_ids: np.ndarray,
    base_indices: np.ndarray,
    k: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Grouped KFold by source file IDs over a subset of indices.
    """
    if k <= 1:
        raise ValueError("k must be > 1 for KFold")

    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int32)
    source_ids = np.asarray(source_ids, dtype=object)
    base_indices = np.asarray(base_indices, dtype=np.int64)

    class_source_folds: Dict[int, List[np.ndarray]] = {}
    for class_id in range(NUM_CLASSES):
        class_mask = labels[base_indices] == class_id
        class_sources = np.unique(source_ids[base_indices[class_mask]]).astype(object)
        class_sources = class_sources.tolist()
        rng.shuffle(class_sources)
        class_source_folds[class_id] = [np.asarray(chunk, dtype=object) for chunk in np.array_split(class_sources, k)]

    for fold_idx in range(k):
        val_sources: List[str] = []
        train_sources: List[str] = []
        for class_id in range(NUM_CLASSES):
            folds = class_source_folds[class_id]
            for idx, chunk in enumerate(folds):
                if idx == fold_idx:
                    val_sources.extend(chunk.tolist())
                else:
                    train_sources.extend(chunk.tolist())

        train_mask = np.isin(source_ids[base_indices], np.asarray(train_sources, dtype=object))
        val_mask = np.isin(source_ids[base_indices], np.asarray(val_sources, dtype=object))

        fold_train_idx = base_indices[np.where(train_mask)[0]]
        fold_val_idx = base_indices[np.where(val_mask)[0]]
        rng.shuffle(fold_train_idx)
        rng.shuffle(fold_val_idx)

        yield fold_idx, fold_train_idx.astype(np.int64), fold_val_idx.astype(np.int64)


def legacy_kfold_indices(
    labels: np.ndarray,
    base_indices: np.ndarray,
    k: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Legacy sample-level stratified KFold over a subset of indices.
    """
    if k <= 1:
        raise ValueError("k must be > 1 for KFold")

    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int32)
    base_indices = np.asarray(base_indices, dtype=np.int64)

    class_indices: Dict[int, np.ndarray] = {}
    for class_id in range(NUM_CLASSES):
        idx = base_indices[labels[base_indices] == class_id].astype(np.int64)
        rng.shuffle(idx)
        class_indices[class_id] = idx

    for fold_idx in range(k):
        fold_train: List[int] = []
        fold_val: List[int] = []
        for class_id in range(NUM_CLASSES):
            idx = class_indices[class_id]
            chunks = np.array_split(idx, k)
            for chunk_idx, chunk in enumerate(chunks):
                if chunk_idx == fold_idx:
                    fold_val.extend(chunk.tolist())
                else:
                    fold_train.extend(chunk.tolist())

        train_idx = np.asarray(fold_train, dtype=np.int64)
        val_idx = np.asarray(fold_val, dtype=np.int64)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        yield fold_idx, train_idx, val_idx
