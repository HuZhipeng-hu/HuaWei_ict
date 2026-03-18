"""Utility helpers for DB5 representation pretraining."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from training.trainer import build_balanced_sample_indices


AUGMENT_PROFILES: dict[str, dict[str, float]] = {
    "mild": {
        "scale_min": 0.92,
        "scale_max": 1.08,
        "noise_std": 0.01,
        "channel_drop_ratio": 0.08,
        "time_mask_ratio": 0.00,
        "freq_mask_ratio": 0.00,
    },
    "strong": {
        "scale_min": 0.88,
        "scale_max": 1.12,
        "noise_std": 0.015,
        "channel_drop_ratio": 0.10,
        "time_mask_ratio": 0.10,
        "freq_mask_ratio": 0.08,
    },
    # Curriculum target profile (starts from mild and anneals toward this target).
    "curriculum": {
        "scale_min": 0.88,
        "scale_max": 1.12,
        "noise_std": 0.015,
        "channel_drop_ratio": 0.10,
        "time_mask_ratio": 0.10,
        "freq_mask_ratio": 0.08,
    },
}


def parse_bool_arg(raw: str | None) -> bool | None:
    if raw is None:
        return None
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def resolve_repr_objective(raw: str) -> str:
    token = str(raw).strip().lower().replace("+", "_")
    token = token.replace("-", "_")
    aliases = {
        "supcon": "supcon",
        "supcon_ce": "supcon_ce",
        "supconce": "supcon_ce",
        "supcon_ce_loss": "supcon_ce",
        "multitask_repr": "multitask_repr",
        "multitask": "multitask_repr",
        "multi_task_repr": "multitask_repr",
    }
    resolved = aliases.get(token)
    if resolved is None:
        raise ValueError(f"Unsupported repr objective: {raw!r}")
    return resolved


def resolve_sampler_mode(raw: str) -> str:
    token = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "balanced": "class_balanced",
        "class_balanced": "class_balanced",
        "class_source_balanced": "class_source_balanced",
        "source_balanced": "class_source_balanced",
    }
    resolved = aliases.get(token)
    if resolved is None:
        raise ValueError(f"Unsupported sampler_mode: {raw!r}")
    return resolved


def _validate_ratio(name: str, value: float) -> float:
    val = float(value)
    if val < 0.0 or val > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return val


def resolve_augmentation_params(
    *,
    profile: str,
    scale_min: float | None = None,
    scale_max: float | None = None,
    noise_std: float | None = None,
    channel_drop_ratio: float | None = None,
    time_mask_ratio: float | None = None,
    freq_mask_ratio: float | None = None,
) -> dict[str, float | str]:
    key = str(profile).strip().lower()
    if key not in AUGMENT_PROFILES:
        raise ValueError(f"Unsupported augmentation profile: {profile!r}")
    base = dict(AUGMENT_PROFILES[key])
    if scale_min is not None:
        base["scale_min"] = float(scale_min)
    if scale_max is not None:
        base["scale_max"] = float(scale_max)
    if base["scale_min"] <= 0.0 or base["scale_max"] <= 0.0 or base["scale_min"] > base["scale_max"]:
        raise ValueError("Invalid scale range for augmentation.")
    if noise_std is not None:
        base["noise_std"] = float(noise_std)
    if base["noise_std"] < 0.0:
        raise ValueError("noise_std must be >= 0.")
    if channel_drop_ratio is not None:
        base["channel_drop_ratio"] = float(channel_drop_ratio)
    if time_mask_ratio is not None:
        base["time_mask_ratio"] = float(time_mask_ratio)
    if freq_mask_ratio is not None:
        base["freq_mask_ratio"] = float(freq_mask_ratio)
    base["channel_drop_ratio"] = _validate_ratio("channel_drop_ratio", base["channel_drop_ratio"])
    base["time_mask_ratio"] = _validate_ratio("time_mask_ratio", base["time_mask_ratio"])
    base["freq_mask_ratio"] = _validate_ratio("freq_mask_ratio", base["freq_mask_ratio"])
    base["profile"] = key
    return base


def resolve_epoch_augmentation_params(
    *,
    params: dict[str, float | str],
    epoch: int,
    total_epochs: int,
) -> dict[str, float | str]:
    """Resolve per-epoch augmentation parameters with optional curriculum schedule."""

    profile = str(params.get("profile", "mild")).strip().lower()
    if profile != "curriculum":
        return dict(params)

    total = max(1, int(total_epochs))
    progress = float(max(0, int(epoch) - 1)) / float(max(1, total - 1))
    # Keep early phase stable, then linearly ramp perturbation strength.
    alpha = 0.0 if progress <= 0.30 else float(min(1.0, (progress - 0.30) / 0.70))
    mild = AUGMENT_PROFILES["mild"]
    target = AUGMENT_PROFILES["strong"]
    out: dict[str, float | str] = dict(params)
    for key in (
        "scale_min",
        "scale_max",
        "noise_std",
        "channel_drop_ratio",
        "time_mask_ratio",
        "freq_mask_ratio",
    ):
        base_val = float(mild[key])
        target_val = float(params.get(key, target[key]))
        out[key] = float(base_val + (target_val - base_val) * alpha)
    out["profile"] = "curriculum"
    out["curriculum_alpha"] = float(alpha)
    out["curriculum_progress"] = float(progress)
    return out


def apply_random_block_mask(
    batch: np.ndarray,
    *,
    rng: np.random.Generator,
    axis: int,
    ratio: float,
) -> None:
    if ratio <= 0.0:
        return
    length = int(batch.shape[axis])
    if length <= 1:
        return
    mask_len = max(1, int(round(length * float(ratio))))
    mask_len = min(mask_len, length)
    max_start = max(1, length - mask_len + 1)
    for sample_idx in range(int(batch.shape[0])):
        start = int(rng.integers(0, max_start))
        slices = [slice(None)] * batch.ndim
        slices[0] = sample_idx
        slices[axis] = slice(start, start + mask_len)
        batch[tuple(slices)] = 0.0


def augment_batch(batch_x: np.ndarray, rng: np.random.Generator, *, params: dict[str, float | str]) -> np.ndarray:
    x = batch_x.astype(np.float32, copy=True)
    scales = rng.uniform(
        float(params["scale_min"]),
        float(params["scale_max"]),
        size=(x.shape[0], 1, 1, 1),
    ).astype(np.float32)
    x *= scales
    noise_std = float(params["noise_std"])
    if noise_std > 0.0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=x.shape).astype(np.float32)
        x += noise
    drop_mask = rng.random((x.shape[0], x.shape[1], 1, 1), dtype=np.float32) < float(params["channel_drop_ratio"])
    x = np.where(drop_mask, 0.0, x)
    apply_random_block_mask(x, rng=rng, axis=3, ratio=float(params["time_mask_ratio"]))
    apply_random_block_mask(x, rng=rng, axis=2, ratio=float(params["freq_mask_ratio"]))
    return x.astype(np.float32)


def build_class_source_balanced_indices(
    *,
    labels: np.ndarray,
    source_ids: np.ndarray,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    if int(batch_size) <= 0 or int(steps) <= 0:
        return np.asarray([], dtype=np.int32)
    labels_i32 = labels.astype(np.int32)
    if labels_i32.size == 0:
        return np.asarray([], dtype=np.int32)
    sources = np.asarray(source_ids)
    if sources.shape[0] != labels_i32.shape[0]:
        raise ValueError("source_ids length mismatch for class-source balanced sampler.")

    class_ids = np.unique(labels_i32)
    if class_ids.size == 0:
        return np.asarray([], dtype=np.int32)
    class_to_source_pools: dict[int, dict[str, np.ndarray]] = {}
    for cls in class_ids.tolist():
        class_mask = labels_i32 == int(cls)
        class_sources = np.unique(sources[class_mask])
        pools: dict[str, np.ndarray] = {}
        for src in class_sources.tolist():
            idx = np.where(class_mask & (sources == src))[0].astype(np.int32)
            if idx.size > 0:
                pools[str(src)] = idx
        if pools:
            class_to_source_pools[int(cls)] = pools

    if not class_to_source_pools:
        raise RuntimeError("No valid class/source pools for class_source_balanced sampler.")

    rng = np.random.default_rng(int(seed))
    all_indices: list[int] = []
    for _step in range(int(steps)):
        if int(class_ids.size) >= int(batch_size):
            chosen_classes = rng.choice(class_ids, size=int(batch_size), replace=False)
        else:
            chosen_classes = np.resize(class_ids, int(batch_size))
            rng.shuffle(chosen_classes)
        batch: list[int] = []
        for cls in chosen_classes.tolist():
            pools = class_to_source_pools[int(cls)]
            source_keys = np.asarray(sorted(pools.keys()), dtype=object)
            src_choice = str(rng.choice(source_keys))
            pool = pools[src_choice]
            batch.append(int(rng.choice(pool)))
        rng.shuffle(batch)
        all_indices.extend(batch[: int(batch_size)])
    return np.asarray(all_indices, dtype=np.int32)


def build_training_indices(
    *,
    labels: np.ndarray,
    source_ids: np.ndarray,
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
    sampler_mode: str,
    sampler_cfg: Any,
    class_names: Sequence[str],
) -> np.ndarray:
    mode = resolve_sampler_mode(sampler_mode)
    if mode == "class_source_balanced":
        return build_class_source_balanced_indices(
            labels=labels,
            source_ids=source_ids,
            batch_size=int(batch_size),
            steps=int(steps_per_epoch),
            seed=int(seed),
        )
    return build_balanced_sample_indices(
        labels.astype(np.int32),
        int(batch_size),
        sampler_cfg,
        steps=int(steps_per_epoch),
        seed=int(seed),
        class_names=list(class_names),
    )


def build_quality_aware_positive_pairs(
    *,
    anchor_indices: np.ndarray,
    labels: np.ndarray,
    metadata_rows: Sequence[dict[str, Any]],
    seed: int,
    cross_source_ratio: float = 0.6,
    top_quality_ratio: float = 0.4,
) -> np.ndarray:
    """Build positive pair indices for contrastive training.

    Priority:
    1) same action, different recording/source
    2) same action, same recording, neighboring windows
    3) fallback: any same-action sample
    """

    anchors = np.asarray(anchor_indices, dtype=np.int32)
    if anchors.size == 0:
        return anchors.copy()
    labels_i32 = np.asarray(labels, dtype=np.int32)
    if labels_i32.shape[0] != len(metadata_rows):
        raise ValueError("metadata length mismatch in quality-aware positive pair builder")

    rng = np.random.default_rng(int(seed))
    by_class: dict[int, np.ndarray] = {}
    for cls in np.unique(labels_i32).tolist():
        by_class[int(cls)] = np.where(labels_i32 == int(cls))[0].astype(np.int32)

    def _recording_id(meta: dict[str, Any]) -> str:
        return str(meta.get("recording_id", "")).strip()

    def _segment(meta: dict[str, Any]) -> int:
        try:
            return int(meta.get("segment_index", -1))
        except Exception:
            return -1

    def _win(meta: dict[str, Any]) -> int:
        try:
            return int(meta.get("window_index", -1))
        except Exception:
            return -1

    def _quality(meta: dict[str, Any]) -> float:
        try:
            return float(meta.get("window_quality_score", 0.0))
        except Exception:
            return 0.0

    def _pick_from(candidates: list[int]) -> int:
        if not candidates:
            return -1
        uniq = np.asarray(sorted(set(int(i) for i in candidates)), dtype=np.int32)
        if uniq.size == 0:
            return -1
        ranked = sorted(uniq.tolist(), key=lambda idx: _quality(metadata_rows[int(idx)]), reverse=True)
        keep = max(1, int(round(len(ranked) * float(np.clip(top_quality_ratio, 0.1, 1.0)))))
        pool = ranked[:keep]
        return int(rng.choice(np.asarray(pool, dtype=np.int32)))

    positives: list[int] = []
    for anchor in anchors.tolist():
        anchor = int(anchor)
        cls = int(labels_i32[anchor])
        class_pool = by_class.get(cls, np.asarray([], dtype=np.int32))
        if class_pool.size <= 1:
            positives.append(anchor)
            continue
        meta = metadata_rows[anchor]
        rec = _recording_id(meta)
        seg = _segment(meta)
        win = _win(meta)

        same_class = [int(i) for i in class_pool.tolist() if int(i) != anchor]
        diff_source = [i for i in same_class if _recording_id(metadata_rows[i]) and _recording_id(metadata_rows[i]) != rec]
        near_same_source = [
            i
            for i in same_class
            if _recording_id(metadata_rows[i]) == rec
            and _segment(metadata_rows[i]) == seg
            and abs(_win(metadata_rows[i]) - win) <= 2
        ]

        use_diff = bool(diff_source) and (not near_same_source or float(rng.random()) < float(cross_source_ratio))
        chosen = _pick_from(diff_source if use_diff else near_same_source)
        if chosen < 0:
            chosen = _pick_from(same_class)
        positives.append(int(chosen if chosen >= 0 else anchor))

    return np.asarray(positives, dtype=np.int32)
