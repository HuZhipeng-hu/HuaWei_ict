"""Utilities for incremental event-onset class expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def normalize_action_keys(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [part.strip().upper() for part in raw.split(",")]
    else:
        items = [str(part).strip().upper() for part in raw]
    seen: set[str] = set()
    normalized: list[str] = []
    for key in items:
        if not key or key == "RELAX":
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def build_event_class_names(action_keys: Sequence[str]) -> list[str]:
    return ["RELAX", *normalize_action_keys(action_keys)]


def build_row_mapping(old_class_names: Sequence[str], new_class_names: Sequence[str]) -> dict[int, int]:
    old_map = {str(name).strip().upper(): idx for idx, name in enumerate(old_class_names)}
    mapping: dict[int, int] = {}
    for new_idx, class_name in enumerate(new_class_names):
        key = str(class_name).strip().upper()
        if key in old_map:
            mapping[int(new_idx)] = int(old_map[key])
    return mapping


@dataclass(frozen=True)
class ExpansionStats:
    reused_class_count: int
    new_class_count: int
    reused_classes: list[str]
    new_classes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def expand_classifier_rows(
    *,
    old_weight: np.ndarray,
    old_bias: np.ndarray | None,
    target_weight: np.ndarray,
    target_bias: np.ndarray | None,
    old_class_names: Sequence[str],
    new_class_names: Sequence[str],
    init_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, ExpansionStats]:
    old_w = np.asarray(old_weight, dtype=np.float32)
    new_w = np.asarray(target_weight, dtype=np.float32).copy()
    if old_w.ndim != 2 or new_w.ndim != 2:
        raise ValueError("Classifier weights must be rank-2 matrices.")
    if int(old_w.shape[1]) != int(new_w.shape[1]):
        raise ValueError(
            "Classifier hidden dimensions mismatch for expansion: "
            f"old={tuple(old_w.shape)} new={tuple(new_w.shape)}"
        )

    old_b = None if old_bias is None else np.asarray(old_bias, dtype=np.float32)
    new_b = None if target_bias is None else np.asarray(target_bias, dtype=np.float32).copy()
    if old_b is not None and old_b.ndim != 1:
        raise ValueError("Old classifier bias must be rank-1.")
    if new_b is not None and new_b.ndim != 1:
        raise ValueError("Target classifier bias must be rank-1.")

    mapping = build_row_mapping(old_class_names, new_class_names)
    reused: list[str] = []
    reused_rows: set[int] = set()
    for new_idx, old_idx in mapping.items():
        if old_idx >= old_w.shape[0] or new_idx >= new_w.shape[0]:
            continue
        new_w[new_idx] = old_w[old_idx]
        reused_rows.add(int(new_idx))
        if new_b is not None and old_b is not None and old_idx < old_b.shape[0] and new_idx < new_b.shape[0]:
            new_b[new_idx] = old_b[old_idx]
        reused.append(str(new_class_names[new_idx]))

    reused_set = {item.upper() for item in reused}
    new_classes = [str(name) for name in new_class_names if str(name).strip().upper() not in reused_set]
    if init_seed is not None:
        pending_rows = [idx for idx in range(new_w.shape[0]) if idx not in reused_rows]
        if pending_rows:
            rng = np.random.default_rng(int(init_seed))
            std = float(np.std(old_w)) if old_w.size else 0.02
            std = max(std, 1e-3)
            new_w[np.asarray(pending_rows, dtype=np.int32)] = rng.normal(
                loc=0.0,
                scale=std,
                size=(len(pending_rows), int(new_w.shape[1])),
            ).astype(np.float32)
            if new_b is not None:
                new_b[np.asarray(pending_rows, dtype=np.int32)] = 0.0
    stats = ExpansionStats(
        reused_class_count=len(reused),
        new_class_count=len(new_classes),
        reused_classes=reused,
        new_classes=new_classes,
    )
    return new_w, new_b, stats
