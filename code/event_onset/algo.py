"""Lightweight algorithmic recognizer for event-onset runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


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


@dataclass(frozen=True)
class EventAlgoModel:
    class_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    centroids: np.ndarray
    temperature: float = 0.15
    rule_config: dict | None = None

    def to_json_dict(self) -> dict:
        return {
            "version": "event_algo_v1",
            "classifier": "nearest_centroid_cosine",
            "class_names": list(self.class_names),
            "feature_dim": int(self.feature_mean.shape[0]),
            "temperature": float(self.temperature),
            "feature_mean": self.feature_mean.astype(np.float32).tolist(),
            "feature_std": self.feature_std.astype(np.float32).tolist(),
            "centroids": self.centroids.astype(np.float32).tolist(),
            "rule_config": dict(self.rule_config or {}),
        }

    @staticmethod
    def from_json_dict(payload: dict) -> "EventAlgoModel":
        class_names = tuple(str(item).strip().upper() for item in payload.get("class_names", []))
        feature_mean = np.asarray(payload.get("feature_mean", []), dtype=np.float32)
        feature_std = np.asarray(payload.get("feature_std", []), dtype=np.float32)
        centroids = np.asarray(payload.get("centroids", []), dtype=np.float32)
        temperature = float(payload.get("temperature", 0.15))
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
        return EventAlgoModel(
            class_names=class_names,
            feature_mean=feature_mean.astype(np.float32),
            feature_std=safe_std,
            centroids=_l2_normalize_rows(centroids.astype(np.float32)),
            temperature=max(1e-3, temperature),
            rule_config=dict(payload.get("rule_config") or {}),
        )


def fit_event_algo_model(
    feature_vectors: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
    temperature: float = 0.15,
) -> EventAlgoModel:
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

    return EventAlgoModel(
        class_names=tuple(names),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        centroids=centroid_matrix,
        temperature=max(1e-3, float(temperature)),
        rule_config={},
    )


def predict_algo_proba_from_vector(model: EventAlgoModel, feature_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != model.feature_mean.shape[0]:
        raise ValueError(
            f"feature_dim mismatch: got={vector.shape[0]}, expected={model.feature_mean.shape[0]}"
        )
    scaled = ((vector - model.feature_mean) / model.feature_std).astype(np.float32)
    norm = float(np.linalg.norm(scaled))
    if norm > 1e-8:
        scaled = scaled / norm
    logits = np.matmul(model.centroids, scaled) / float(model.temperature)
    return _softmax(np.asarray(logits, dtype=np.float32))


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

    def _rule_predict(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray | None:
        if not self.rules_enabled:
            return None
        emg = _ensure_emg_shape(emg_feature)
        imu = _ensure_imu_shape(imu_feature)
        num_classes = len(self.tuple_class_names)

        emg_energy = float(np.mean(np.abs(emg)))
        imu_motion = float(np.mean(np.abs(np.diff(imu, axis=1)))) if imu.shape[1] > 1 else 0.0

        gyro_z_idx = 5 if imu.shape[0] > 5 else (imu.shape[0] - 1)
        gyro_z = imu[int(max(0, gyro_z_idx))]
        cw_score = float(np.mean(np.maximum(gyro_z, 0.0)))
        ccw_score = float(np.mean(np.maximum(-gyro_z, 0.0)))
        if max(cw_score, ccw_score) >= self.wrist_rule_min and abs(cw_score - ccw_score) >= self.wrist_rule_margin:
            name = "WRIST_CW" if cw_score > ccw_score else "WRIST_CCW"
            idx = self._find_class_idx(name)
            if idx is not None:
                return _one_hot_confidence(num_classes, idx, self.rule_confidence)

        release_idx = self._find_class_idx("TENSE_OPEN")
        if release_idx is not None and emg_energy >= self.release_emg_min and imu_motion <= self.release_imu_max:
            return _one_hot_confidence(num_classes, release_idx, self.rule_confidence)
        return None

    def predict_proba(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
        rule_output = self._rule_predict(emg_feature, imu_feature)
        if rule_output is not None:
            return rule_output.astype(np.float32)
        vector = build_event_algo_feature_vector(emg_feature, imu_feature)
        return predict_algo_proba_from_vector(self.model, vector)
