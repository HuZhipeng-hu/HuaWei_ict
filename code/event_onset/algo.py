"""Lightweight algorithmic recognizer for event-onset runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ALGO_MODE_V1 = "v1_single"
ALGO_MODE_V2 = "v2_two_stage"


def _normalize_algo_mode(raw: str | None) -> str:
    mode = str(raw or "").strip().lower()
    if not mode:
        return ALGO_MODE_V1
    if mode not in {ALGO_MODE_V1, ALGO_MODE_V2}:
        raise ValueError(f"Unsupported algo_mode={raw!r}. Expected one of: {ALGO_MODE_V1}, {ALGO_MODE_V2}")
    return mode


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return (exp_logits / np.sum(exp_logits)).astype(np.float32)


def _one_hot_confidence(num_classes: int, index: int, confidence: float) -> np.ndarray:
    conf = min(max(float(confidence), 0.0), 1.0)
    if num_classes <= 1:
        return np.asarray([1.0], dtype=np.float32)
    base = float((1.0 - conf) / float(max(1, num_classes - 1)))
    probs = np.full((num_classes,), base, dtype=np.float32)
    probs[int(index)] = conf
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    return (probs / float(np.sum(probs))).astype(np.float32)


def _ensure_emg_shape(emg_feature: np.ndarray) -> np.ndarray:
    emg = np.asarray(emg_feature, dtype=np.float32)
    if emg.ndim == 4:
        if emg.shape[0] != 1:
            raise ValueError(f"Expected EMG batch size 1, got shape={tuple(emg.shape)}")
        emg = emg[0]
    if emg.ndim != 3:
        raise ValueError(f"Expected EMG feature ndim=3 or 4, got ndim={emg.ndim}")
    return emg


def _ensure_imu_shape(imu_feature: np.ndarray) -> np.ndarray:
    imu = np.asarray(imu_feature, dtype=np.float32)
    if imu.ndim == 3:
        if imu.shape[0] != 1:
            raise ValueError(f"Expected IMU batch size 1, got shape={tuple(imu.shape)}")
        imu = imu[0]
    if imu.ndim != 2:
        raise ValueError(f"Expected IMU feature ndim=2 or 3, got ndim={imu.ndim}")
    return imu


def compute_rule_signal_scores(emg_feature: np.ndarray, imu_feature: np.ndarray) -> dict[str, float]:
    emg = _ensure_emg_shape(emg_feature)
    imu = _ensure_imu_shape(imu_feature)

    emg_energy = float(np.mean(np.abs(emg)))
    imu_motion = float(np.mean(np.abs(np.diff(imu, axis=1)))) if imu.shape[1] > 1 else 0.0

    gyro_z_idx = 5 if imu.shape[0] > 5 else max(0, imu.shape[0] - 1)
    gyro_z = imu[int(gyro_z_idx)]
    cw_score = float(np.mean(np.maximum(gyro_z, 0.0)))
    ccw_score = float(np.mean(np.maximum(-gyro_z, 0.0)))
    wrist_peak = float(max(cw_score, ccw_score))
    wrist_margin = float(abs(cw_score - ccw_score))

    return {
        "emg_energy": emg_energy,
        "imu_motion": imu_motion,
        "cw_score": cw_score,
        "ccw_score": ccw_score,
        "wrist_peak": wrist_peak,
        "wrist_margin": wrist_margin,
    }


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, float(percentile)))


def suggest_rule_thresholds_from_features(
    emg_samples: np.ndarray,
    imu_samples: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
    mask: np.ndarray | None = None,
    fallback: dict[str, float] | None = None,
) -> dict[str, Any]:
    fallback = dict(fallback or {})
    names = [str(name).strip().upper() for name in class_names]
    if emg_samples.shape[0] != imu_samples.shape[0] or emg_samples.shape[0] != labels.shape[0]:
        raise ValueError("emg_samples/imu_samples/labels size mismatch")

    use_mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones((labels.shape[0],), dtype=bool)
    if use_mask.shape[0] != labels.shape[0]:
        raise ValueError("mask size mismatch")

    emg_used = np.asarray(emg_samples[use_mask], dtype=np.float32)
    imu_used = np.asarray(imu_samples[use_mask], dtype=np.float32)
    label_used = np.asarray(labels[use_mask], dtype=np.int32)

    if emg_used.shape[0] == 0:
        return {
            "status": "fallback_no_samples",
            "thresholds": {
                "wrist_rule_min": float(fallback.get("wrist_rule_min", 0.55)),
                "wrist_rule_margin": float(fallback.get("wrist_rule_margin", 0.10)),
                "release_emg_min": float(fallback.get("release_emg_min", 0.45)),
                "release_imu_max": float(fallback.get("release_imu_max", 1.50)),
            },
            "stats": {},
            "sample_count": 0,
        }

    score_rows = [compute_rule_signal_scores(emg_used[idx], imu_used[idx]) for idx in range(emg_used.shape[0])]
    wrist_peak = np.asarray([row["wrist_peak"] for row in score_rows], dtype=np.float32)
    wrist_margin = np.asarray([row["wrist_margin"] for row in score_rows], dtype=np.float32)
    emg_energy = np.asarray([row["emg_energy"] for row in score_rows], dtype=np.float32)
    imu_motion = np.asarray([row["imu_motion"] for row in score_rows], dtype=np.float32)

    wrist_indices = {idx for idx, name in enumerate(names) if name in {"WRIST_CW", "WRIST_CCW"}}
    tense_index = next((idx for idx, name in enumerate(names) if name == "TENSE_OPEN"), None)

    wrist_pos = np.asarray([label in wrist_indices for label in label_used], dtype=bool)
    wrist_neg = ~wrist_pos
    tense_pos = np.asarray([int(label) == int(tense_index) for label in label_used], dtype=bool) if tense_index is not None else None
    tense_neg = ~tense_pos if tense_pos is not None else None

    def _blend_cutoff(pos_values: np.ndarray, neg_values: np.ndarray, *, pos_pct: float, neg_pct: float, default: float) -> float:
        pos_anchor = _safe_percentile(pos_values, pos_pct)
        neg_anchor = _safe_percentile(neg_values, neg_pct)
        if pos_anchor is not None and neg_anchor is not None:
            return float((pos_anchor + neg_anchor) / 2.0)
        if pos_anchor is not None:
            return float(pos_anchor)
        if neg_anchor is not None:
            return float(neg_anchor)
        return float(default)

    wrist_rule_min = _blend_cutoff(
        wrist_peak[wrist_pos],
        wrist_peak[wrist_neg],
        pos_pct=20.0,
        neg_pct=90.0,
        default=float(fallback.get("wrist_rule_min", 0.55)),
    )
    wrist_rule_margin = _blend_cutoff(
        wrist_margin[wrist_pos],
        wrist_margin[wrist_neg],
        pos_pct=20.0,
        neg_pct=90.0,
        default=float(fallback.get("wrist_rule_margin", 0.10)),
    )

    if tense_pos is None:
        release_emg_min = float(fallback.get("release_emg_min", 0.45))
        release_imu_max = float(fallback.get("release_imu_max", 1.50))
    else:
        release_emg_min = _blend_cutoff(
            emg_energy[tense_pos],
            emg_energy[tense_neg],
            pos_pct=20.0,
            neg_pct=90.0,
            default=float(fallback.get("release_emg_min", 0.45)),
        )
        pos_imu_hi = _safe_percentile(imu_motion[tense_pos], 80.0)
        neg_imu_lo = _safe_percentile(imu_motion[tense_neg], 10.0)
        if pos_imu_hi is not None and neg_imu_lo is not None:
            release_imu_max = float((pos_imu_hi + neg_imu_lo) / 2.0)
        elif pos_imu_hi is not None:
            release_imu_max = float(pos_imu_hi)
        else:
            release_imu_max = float(fallback.get("release_imu_max", 1.50))

    thresholds = {
        "wrist_rule_min": float(np.clip(wrist_rule_min, 0.20, 4.00)),
        "wrist_rule_margin": float(np.clip(wrist_rule_margin, 0.02, 1.20)),
        "release_emg_min": float(np.clip(release_emg_min, 0.10, 6.00)),
        "release_imu_max": float(np.clip(release_imu_max, 0.20, 6.00)),
    }

    stats = {
        "wrist_pos_count": int(np.sum(wrist_pos)),
        "wrist_neg_count": int(np.sum(wrist_neg)),
        "tense_pos_count": int(np.sum(tense_pos)) if tense_pos is not None else 0,
        "tense_neg_count": int(np.sum(tense_neg)) if tense_neg is not None else 0,
        "wrist_peak_p20_pos": _safe_percentile(wrist_peak[wrist_pos], 20.0),
        "wrist_peak_p90_neg": _safe_percentile(wrist_peak[wrist_neg], 90.0),
        "wrist_margin_p20_pos": _safe_percentile(wrist_margin[wrist_pos], 20.0),
        "wrist_margin_p90_neg": _safe_percentile(wrist_margin[wrist_neg], 90.0),
        "release_emg_p20_pos": _safe_percentile(emg_energy[tense_pos], 20.0) if tense_pos is not None else None,
        "release_emg_p90_neg": _safe_percentile(emg_energy[tense_neg], 90.0) if tense_pos is not None else None,
        "release_imu_p80_pos": _safe_percentile(imu_motion[tense_pos], 80.0) if tense_pos is not None else None,
        "release_imu_p10_neg": _safe_percentile(imu_motion[tense_neg], 10.0) if tense_pos is not None else None,
    }

    return {
        "status": "ok",
        "thresholds": thresholds,
        "stats": stats,
        "sample_count": int(emg_used.shape[0]),
    }


def build_event_algo_feature_vector(emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
    """Build a compact deterministic feature vector from EMG/IMU runtime features."""

    emg = _ensure_emg_shape(emg_feature)
    imu = _ensure_imu_shape(imu_feature)

    emg_abs = np.abs(emg)
    imu_abs = np.abs(imu)
    emg_diff = np.abs(np.diff(emg, axis=2)) if emg.shape[2] > 1 else np.zeros_like(emg)
    imu_diff = np.abs(np.diff(imu, axis=1)) if imu.shape[1] > 1 else np.zeros_like(imu)

    emg_stats = np.concatenate(
        [
            np.mean(emg_abs, axis=(1, 2)),
            np.std(emg, axis=(1, 2)),
            np.mean(emg_diff, axis=(1, 2)),
            np.max(emg_abs, axis=(1, 2)),
        ],
        axis=0,
    ).astype(np.float32)
    imu_stats = np.concatenate(
        [
            np.mean(imu_abs, axis=1),
            np.std(imu, axis=1),
            np.mean(imu_diff, axis=1),
            np.max(imu_abs, axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    flat = np.concatenate(
        [
            emg.reshape(-1).astype(np.float32),
            imu.reshape(-1).astype(np.float32),
            emg_stats,
            imu_stats,
        ],
        axis=0,
    )
    return flat.astype(np.float32)


def _fit_centroid_bank(
    feature_vectors: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(feature_vectors, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int32)
    names = [str(name).strip().upper() for name in class_names]

    if samples.ndim != 2:
        raise ValueError(f"feature_vectors must be rank-2, got shape={tuple(samples.shape)}")
    if target.ndim != 1 or target.shape[0] != samples.shape[0]:
        raise ValueError("labels shape mismatch with feature_vectors")
    if len(names) == 0:
        raise ValueError("class_names must not be empty")

    mean = np.mean(samples, axis=0, dtype=np.float32)
    std = np.std(samples, axis=0, dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    scaled = ((samples - mean) / std).astype(np.float32)
    scaled = _l2_normalize_rows(scaled)

    centroids: list[np.ndarray] = []
    for class_idx in range(len(names)):
        mask = target == int(class_idx)
        if not np.any(mask):
            raise ValueError(f"No train samples for class index {class_idx} ({names[class_idx]})")
        centroid = np.mean(scaled[mask], axis=0, dtype=np.float32)
        centroids.append(centroid.astype(np.float32))
    centroid_matrix = _l2_normalize_rows(np.stack(centroids, axis=0).astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32), centroid_matrix


def _predict_with_bank(
    feature_vector: np.ndarray,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    centroids: np.ndarray,
    temperature: float,
) -> np.ndarray:
    vector = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != feature_mean.shape[0]:
        raise ValueError(
            f"feature_dim mismatch: got={vector.shape[0]}, expected={feature_mean.shape[0]}"
        )
    scaled = ((vector - feature_mean) / feature_std).astype(np.float32)
    norm = float(np.linalg.norm(scaled))
    if norm > 1e-8:
        scaled = scaled / norm
    logits = np.matmul(centroids, scaled) / float(max(1e-3, temperature))
    return _softmax(np.asarray(logits, dtype=np.float32))


@dataclass(frozen=True)
class EventAlgoModel:
    class_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    centroids: np.ndarray
    temperature: float = 0.15
    rule_config: dict[str, Any] | None = None
    algo_mode: str = ALGO_MODE_V1
    gate_feature_mean: np.ndarray | None = None
    gate_feature_std: np.ndarray | None = None
    gate_centroids: np.ndarray | None = None
    gate_action_threshold: float = 0.55
    gate_margin_threshold: float = 0.05
    action_class_names: tuple[str, ...] | None = None
    action_feature_mean: np.ndarray | None = None
    action_feature_std: np.ndarray | None = None
    action_centroids: np.ndarray | None = None

    def to_json_dict(self) -> dict:
        mode = _normalize_algo_mode(self.algo_mode)
        payload = {
            "version": "event_algo_v2" if mode == ALGO_MODE_V2 else "event_algo_v1",
            "algo_mode": mode,
            "classifier": "nearest_centroid_cosine",
            "class_names": list(self.class_names),
            "feature_dim": int(self.feature_mean.shape[0]),
            "temperature": float(self.temperature),
            "feature_mean": self.feature_mean.astype(np.float32).tolist(),
            "feature_std": self.feature_std.astype(np.float32).tolist(),
            "centroids": self.centroids.astype(np.float32).tolist(),
            "rule_config": dict(self.rule_config or {}),
        }
        if mode == ALGO_MODE_V2:
            payload.update(
                {
                    "gate_action_threshold": float(self.gate_action_threshold),
                    "gate_margin_threshold": float(self.gate_margin_threshold),
                    "gate_classifier": {
                        "feature_mean": np.asarray(self.gate_feature_mean, dtype=np.float32).tolist(),
                        "feature_std": np.asarray(self.gate_feature_std, dtype=np.float32).tolist(),
                        "centroids": np.asarray(self.gate_centroids, dtype=np.float32).tolist(),
                    },
                    "action_classifier": {
                        "class_names": list(self.action_class_names or ()),
                        "feature_mean": np.asarray(self.action_feature_mean, dtype=np.float32).tolist(),
                        "feature_std": np.asarray(self.action_feature_std, dtype=np.float32).tolist(),
                        "centroids": np.asarray(self.action_centroids, dtype=np.float32).tolist(),
                    },
                }
            )
        return payload

    @staticmethod
    def from_json_dict(payload: dict) -> "EventAlgoModel":
        class_names = tuple(str(item).strip().upper() for item in payload.get("class_names", []))
        feature_mean = np.asarray(payload.get("feature_mean", []), dtype=np.float32)
        feature_std = np.asarray(payload.get("feature_std", []), dtype=np.float32)
        centroids = np.asarray(payload.get("centroids", []), dtype=np.float32)
        temperature = float(payload.get("temperature", 0.15))
        version = str(payload.get("version", "event_algo_v1")).strip().lower()
        implied_mode = ALGO_MODE_V2 if version == "event_algo_v2" else ALGO_MODE_V1
        algo_mode = _normalize_algo_mode(payload.get("algo_mode", implied_mode))

        if not class_names:
            raise ValueError("algo_model missing class_names")
        if feature_mean.ndim != 1 or feature_std.ndim != 1:
            raise ValueError("algo_model feature_mean/feature_std must be rank-1")
        if centroids.ndim != 2:
            raise ValueError("algo_model centroids must be rank-2")
        if centroids.shape[0] != len(class_names):
            raise ValueError(
                f"algo_model centroids rows ({centroids.shape[0]}) mismatch class_names ({len(class_names)})"
            )
        if centroids.shape[1] != feature_mean.shape[0] or feature_std.shape[0] != feature_mean.shape[0]:
            raise ValueError("algo_model feature dimension mismatch among scaler/centroids")
        safe_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)

        gate_feature_mean = None
        gate_feature_std = None
        gate_centroids = None
        action_class_names: tuple[str, ...] | None = None
        action_feature_mean = None
        action_feature_std = None
        action_centroids = None
        gate_action_threshold = float(payload.get("gate_action_threshold", 0.55))
        gate_margin_threshold = float(payload.get("gate_margin_threshold", 0.05))

        if algo_mode == ALGO_MODE_V2:
            gate_payload = dict(payload.get("gate_classifier") or {})
            action_payload = dict(payload.get("action_classifier") or {})
            gate_feature_mean = np.asarray(gate_payload.get("feature_mean", []), dtype=np.float32)
            gate_feature_std = np.asarray(gate_payload.get("feature_std", []), dtype=np.float32)
            gate_centroids = np.asarray(gate_payload.get("centroids", []), dtype=np.float32)
            action_class_names = tuple(str(item).strip().upper() for item in action_payload.get("class_names", []))
            action_feature_mean = np.asarray(action_payload.get("feature_mean", []), dtype=np.float32)
            action_feature_std = np.asarray(action_payload.get("feature_std", []), dtype=np.float32)
            action_centroids = np.asarray(action_payload.get("centroids", []), dtype=np.float32)

            if gate_feature_mean.ndim != 1 or gate_feature_std.ndim != 1:
                raise ValueError("algo_model gate_classifier feature_mean/feature_std must be rank-1")
            if gate_centroids.ndim != 2 or gate_centroids.shape[0] != 2:
                raise ValueError("algo_model gate_classifier centroids must have shape [2, feature_dim]")
            if gate_centroids.shape[1] != feature_mean.shape[0] or gate_feature_mean.shape[0] != feature_mean.shape[0]:
                raise ValueError("algo_model gate_classifier feature dimension mismatch")
            if gate_feature_std.shape[0] != feature_mean.shape[0]:
                raise ValueError("algo_model gate_classifier std dimension mismatch")

            if not action_class_names:
                raise ValueError("algo_model action_classifier missing class_names")
            if action_feature_mean.ndim != 1 or action_feature_std.ndim != 1:
                raise ValueError("algo_model action_classifier feature_mean/feature_std must be rank-1")
            if action_centroids.ndim != 2 or action_centroids.shape[0] != len(action_class_names):
                raise ValueError("algo_model action_classifier centroids rows mismatch class_names")
            if action_centroids.shape[1] != feature_mean.shape[0] or action_feature_mean.shape[0] != feature_mean.shape[0]:
                raise ValueError("algo_model action_classifier feature dimension mismatch")
            if action_feature_std.shape[0] != feature_mean.shape[0]:
                raise ValueError("algo_model action_classifier std dimension mismatch")
            if "RELAX" in action_class_names:
                raise ValueError("algo_model action_classifier class_names must not include RELAX")

            gate_feature_std = np.where(gate_feature_std < 1e-8, 1.0, gate_feature_std).astype(np.float32)
            action_feature_std = np.where(action_feature_std < 1e-8, 1.0, action_feature_std).astype(np.float32)
            gate_centroids = _l2_normalize_rows(gate_centroids.astype(np.float32))
            action_centroids = _l2_normalize_rows(action_centroids.astype(np.float32))

        return EventAlgoModel(
            class_names=class_names,
            feature_mean=feature_mean.astype(np.float32),
            feature_std=safe_std,
            centroids=_l2_normalize_rows(centroids.astype(np.float32)),
            temperature=max(1e-3, temperature),
            rule_config=dict(payload.get("rule_config") or {}),
            algo_mode=algo_mode,
            gate_feature_mean=gate_feature_mean,
            gate_feature_std=gate_feature_std,
            gate_centroids=gate_centroids,
            gate_action_threshold=float(gate_action_threshold),
            gate_margin_threshold=float(gate_margin_threshold),
            action_class_names=action_class_names,
            action_feature_mean=action_feature_mean,
            action_feature_std=action_feature_std,
            action_centroids=action_centroids,
        )


def fit_event_algo_model(
    feature_vectors: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
    temperature: float = 0.15,
    algo_mode: str = ALGO_MODE_V1,
    gate_action_threshold: float = 0.55,
    gate_margin_threshold: float = 0.05,
) -> EventAlgoModel:
    mode = _normalize_algo_mode(algo_mode)
    names = [str(name).strip().upper() for name in class_names]
    samples = np.asarray(feature_vectors, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int32)

    mean, std, centroids = _fit_centroid_bank(samples, target, class_names=names)
    model = EventAlgoModel(
        class_names=tuple(names),
        feature_mean=mean,
        feature_std=std,
        centroids=centroids,
        temperature=max(1e-3, float(temperature)),
        rule_config={},
        algo_mode=mode,
    )
    if mode == ALGO_MODE_V1:
        return model

    try:
        relax_idx = names.index("RELAX")
    except ValueError as exc:
        raise ValueError("Two-stage algo requires RELAX in class_names.") from exc

    gate_labels = (target != int(relax_idx)).astype(np.int32)
    gate_mean, gate_std, gate_centroids = _fit_centroid_bank(
        samples,
        gate_labels,
        class_names=["RELAX_GATE", "ACTION_GATE"],
    )

    action_indices = [idx for idx, name in enumerate(names) if name != "RELAX"]
    if not action_indices:
        raise ValueError("Two-stage algo requires at least one non-RELAX class.")
    action_mask = gate_labels == 1
    action_samples = samples[action_mask]
    if action_samples.shape[0] == 0:
        raise ValueError("Two-stage algo requires non-RELAX train samples.")
    action_class_names = [names[idx] for idx in action_indices]
    raw_to_action = {raw_idx: action_idx for action_idx, raw_idx in enumerate(action_indices)}
    action_labels = np.asarray([raw_to_action[int(item)] for item in target[action_mask]], dtype=np.int32)
    action_mean, action_std, action_centroids = _fit_centroid_bank(
        action_samples,
        action_labels,
        class_names=action_class_names,
    )

    return EventAlgoModel(
        class_names=tuple(names),
        feature_mean=mean,
        feature_std=std,
        centroids=centroids,
        temperature=max(1e-3, float(temperature)),
        rule_config={},
        algo_mode=mode,
        gate_feature_mean=gate_mean,
        gate_feature_std=gate_std,
        gate_centroids=gate_centroids,
        gate_action_threshold=float(gate_action_threshold),
        gate_margin_threshold=float(gate_margin_threshold),
        action_class_names=tuple(action_class_names),
        action_feature_mean=action_mean,
        action_feature_std=action_std,
        action_centroids=action_centroids,
    )


def predict_algo_proba_with_meta_from_vector(
    model: EventAlgoModel,
    feature_vector: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = _normalize_algo_mode(model.algo_mode)
    class_names = [str(name).strip().upper() for name in model.class_names]
    num_classes = len(class_names)
    relax_idx = class_names.index("RELAX") if "RELAX" in class_names else None

    if mode != ALGO_MODE_V2:
        probs = _predict_with_bank(
            feature_vector,
            feature_mean=model.feature_mean,
            feature_std=model.feature_std,
            centroids=model.centroids,
            temperature=model.temperature,
        )
        gate_relax = float(probs[int(relax_idx)]) if relax_idx is not None else 0.0
        gate_action = float(max(0.0, 1.0 - gate_relax))
        meta = {
            "algo_mode": ALGO_MODE_V1,
            "rule_hit": False,
            "rule_name": None,
            "gate_action_prob": gate_action,
            "gate_relax_prob": gate_relax,
            "gate_accepted": bool(int(np.argmax(probs)) != int(relax_idx) if relax_idx is not None else True),
            "stage2_used": False,
            "stage2_pred_class": None,
        }
        return probs.astype(np.float32), meta

    if (
        model.gate_feature_mean is None
        or model.gate_feature_std is None
        or model.gate_centroids is None
        or model.action_feature_mean is None
        or model.action_feature_std is None
        or model.action_centroids is None
        or not model.action_class_names
    ):
        raise ValueError("Two-stage algo model is incomplete. Missing gate/action classifier fields.")

    gate_probs = _predict_with_bank(
        feature_vector,
        feature_mean=np.asarray(model.gate_feature_mean, dtype=np.float32),
        feature_std=np.asarray(model.gate_feature_std, dtype=np.float32),
        centroids=np.asarray(model.gate_centroids, dtype=np.float32),
        temperature=model.temperature,
    )
    gate_relax = float(gate_probs[0])
    gate_action = float(gate_probs[1])
    gate_accepted = bool(
        gate_action >= float(model.gate_action_threshold)
        and (gate_action - gate_relax) >= float(model.gate_margin_threshold)
    )

    if not gate_accepted:
        if relax_idx is None:
            probs = _predict_with_bank(
                feature_vector,
                feature_mean=model.feature_mean,
                feature_std=model.feature_std,
                centroids=model.centroids,
                temperature=model.temperature,
            )
        else:
            relax_conf = float(np.clip(max(gate_relax, 1.0 - gate_action), 0.50, 0.98))
            probs = _one_hot_confidence(num_classes, int(relax_idx), relax_conf)
        meta = {
            "algo_mode": ALGO_MODE_V2,
            "rule_hit": False,
            "rule_name": None,
            "gate_action_prob": gate_action,
            "gate_relax_prob": gate_relax,
            "gate_accepted": False,
            "stage2_used": False,
            "stage2_pred_class": None,
        }
        return probs.astype(np.float32), meta

    action_probs = _predict_with_bank(
        feature_vector,
        feature_mean=np.asarray(model.action_feature_mean, dtype=np.float32),
        feature_std=np.asarray(model.action_feature_std, dtype=np.float32),
        centroids=np.asarray(model.action_centroids, dtype=np.float32),
        temperature=model.temperature,
    )

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    full_probs = np.zeros((num_classes,), dtype=np.float32)
    action_mass = float(np.clip(gate_action, 0.0, 1.0))
    for action_idx, action_name in enumerate(model.action_class_names or ()):  # type: ignore[arg-type]
        mapped_idx = class_to_idx.get(str(action_name).strip().upper())
        if mapped_idx is None:
            continue
        full_probs[int(mapped_idx)] = float(action_probs[action_idx]) * action_mass
    if relax_idx is not None:
        full_probs[int(relax_idx)] = float(max(0.0, 1.0 - action_mass))

    total = float(np.sum(full_probs))
    if total <= 1e-8:
        full_probs = _predict_with_bank(
            feature_vector,
            feature_mean=model.feature_mean,
            feature_std=model.feature_std,
            centroids=model.centroids,
            temperature=model.temperature,
        )
    else:
        full_probs = (full_probs / total).astype(np.float32)

    stage2_pred_idx = int(np.argmax(action_probs))
    stage2_pred_class = str((model.action_class_names or ("",))[stage2_pred_idx]).strip().upper()
    meta = {
        "algo_mode": ALGO_MODE_V2,
        "rule_hit": False,
        "rule_name": None,
        "gate_action_prob": gate_action,
        "gate_relax_prob": gate_relax,
        "gate_accepted": True,
        "stage2_used": True,
        "stage2_pred_class": stage2_pred_class,
    }
    return full_probs.astype(np.float32), meta


def predict_algo_proba_from_vector(model: EventAlgoModel, feature_vector: np.ndarray) -> np.ndarray:
    probs, _ = predict_algo_proba_with_meta_from_vector(model, feature_vector)
    return probs.astype(np.float32)


def compute_algo_stage_metrics(
    model: EventAlgoModel,
    feature_vectors: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
) -> dict[str, float]:
    samples = np.asarray(feature_vectors, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int32)
    names = [str(name).strip().upper() for name in class_names]
    relax_idx = names.index("RELAX") if "RELAX" in names else None

    if samples.shape[0] == 0:
        return {
            "gate_accept_rate": 0.0,
            "gate_action_recall": 0.0,
            "stage2_action_acc": 0.0,
            "rule_hit_rate": 0.0,
        }

    gate_accept_count = 0
    action_total = 0
    action_accept_count = 0
    stage2_total = 0
    stage2_correct = 0
    rule_hit_count = 0

    for idx in range(samples.shape[0]):
        probs, meta = predict_algo_proba_with_meta_from_vector(model, samples[idx])
        pred = int(np.argmax(probs))
        is_action_true = bool(relax_idx is not None and int(target[idx]) != int(relax_idx))

        gate_accepted = bool(meta.get("gate_accepted", False))
        stage2_used = bool(meta.get("stage2_used", False))
        rule_hit = bool(meta.get("rule_hit", False))

        if gate_accepted:
            gate_accept_count += 1
        if rule_hit:
            rule_hit_count += 1

        if is_action_true:
            action_total += 1
            if gate_accepted:
                action_accept_count += 1
            if stage2_used:
                stage2_total += 1
                if pred == int(target[idx]):
                    stage2_correct += 1

    return {
        "gate_accept_rate": float(gate_accept_count / samples.shape[0]),
        "gate_action_recall": float(action_accept_count / action_total) if action_total else 0.0,
        "stage2_action_acc": float(stage2_correct / stage2_total) if stage2_total else 0.0,
        "rule_hit_rate": float(rule_hit_count / samples.shape[0]),
    }


def save_event_algo_model(model: EventAlgoModel, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_event_algo_model(path: str | Path) -> EventAlgoModel:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Algo model not found: {model_path}")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("algo model payload must be an object")
    return EventAlgoModel.from_json_dict(payload)


class EventAlgoPredictor:
    """Algorithm backend predictor that matches EventPredictor output contract."""

    def __init__(self, *, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = load_event_algo_model(self.model_path)
        cfg = dict(self.model.rule_config or {})
        self.rules_enabled = bool(cfg.get("enabled", False))
        self.wrist_rule_min = float(cfg.get("wrist_rule_min", 0.55))
        self.wrist_rule_margin = float(cfg.get("wrist_rule_margin", 0.10))
        self.release_emg_min = float(cfg.get("release_emg_min", 0.45))
        self.release_imu_max = float(cfg.get("release_imu_max", 1.50))
        self.rule_confidence = float(cfg.get("rule_confidence", 0.94))
        self.tuple_class_names = tuple(str(name).strip().upper() for name in self.model.class_names)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.tuple_class_names

    def _find_class_idx(self, name: str) -> int | None:
        query = str(name).strip().upper()
        try:
            return self.tuple_class_names.index(query)
        except ValueError:
            return None

    def _rule_predict(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        if not self.rules_enabled:
            return None, None
        scores = compute_rule_signal_scores(emg_feature, imu_feature)
        num_classes = len(self.tuple_class_names)
        cw_score = float(scores["cw_score"])
        ccw_score = float(scores["ccw_score"])
        if max(cw_score, ccw_score) >= self.wrist_rule_min and abs(cw_score - ccw_score) >= self.wrist_rule_margin:
            name = "WRIST_CW" if cw_score > ccw_score else "WRIST_CCW"
            idx = self._find_class_idx(name)
            if idx is not None:
                return _one_hot_confidence(num_classes, idx, self.rule_confidence), name

        release_idx = self._find_class_idx("TENSE_OPEN")
        if (
            release_idx is not None
            and float(scores["emg_energy"]) >= self.release_emg_min
            and float(scores["imu_motion"]) <= self.release_imu_max
        ):
            return _one_hot_confidence(num_classes, release_idx, self.rule_confidence), "TENSE_OPEN"
        return None, None

    def _predict_internal(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        rule_output, rule_name = self._rule_predict(emg_feature, imu_feature)
        if rule_output is not None:
            meta = {
                "algo_mode": _normalize_algo_mode(self.model.algo_mode),
                "rule_hit": True,
                "rule_name": str(rule_name or "").strip().upper() or None,
                "gate_action_prob": 1.0,
                "gate_relax_prob": 0.0,
                "gate_accepted": True,
                "stage2_used": False,
                "stage2_pred_class": None,
            }
            return rule_output.astype(np.float32), meta

        vector = build_event_algo_feature_vector(emg_feature, imu_feature)
        probs, meta = predict_algo_proba_with_meta_from_vector(self.model, vector)
        meta = dict(meta)
        meta["rule_hit"] = False
        meta["rule_name"] = None
        return probs.astype(np.float32), meta

    def predict_proba(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
        probs, _ = self._predict_internal(emg_feature, imu_feature)
        return probs.astype(np.float32)

    def predict_proba_with_meta(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return self._predict_internal(emg_feature, imu_feature)
