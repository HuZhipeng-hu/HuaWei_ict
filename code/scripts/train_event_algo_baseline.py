"""Train and export an algorithmic event-onset recognizer (EMG+IMU features)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.algo import (
    build_event_algo_feature_vector,
    fit_event_algo_model,
    predict_algo_proba_from_vector,
    save_event_algo_model,
)
from event_onset.config import load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from training.data.split_strategy import load_manifest
from training.reporting import compute_classification_report, save_classification_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lightweight algo recognizer for event-onset runtime")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--target_db5_keys", default=None)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_id", default="event_algo_baseline")
    parser.add_argument("--algo_model_out", default=None)
    return parser


def _parse_target_keys(raw: str | None) -> list[str] | None:
    if not str(raw or "").strip():
        return None
    keys = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not keys:
        raise ValueError("--target_db5_keys provided but no valid tokens were parsed.")
    return keys


def _resolve_manifest_path(base_dir: Path, raw: str | None, *, desc: str) -> Path:
    if not str(raw or "").strip():
        raise FileNotFoundError(f"{desc} is required but missing.")
    path = Path(str(raw).strip())
    candidates = [path]
    if not path.is_absolute():
        candidates.insert(0, base_dir / path)
        candidates.insert(0, CODE_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{desc} not found: {candidates[0]}")


def _build_feature_matrix(emg_samples: np.ndarray, imu_samples: np.ndarray) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for idx in range(emg_samples.shape[0]):
        vectors.append(
            build_event_algo_feature_vector(
                emg_samples[idx],
                imu_samples[idx],
            )
        )
    return np.stack(vectors, axis=0).astype(np.float32)


def _mask_by_sources(source_ids: np.ndarray, allowed_sources: set[str]) -> np.ndarray:
    return np.asarray([str(source) in allowed_sources for source in source_ids], dtype=bool)


def _predict_labels(model, features: np.ndarray) -> np.ndarray:
    predictions = np.zeros((features.shape[0],), dtype=np.int32)
    for idx in range(features.shape[0]):
        probs = predict_algo_proba_from_vector(model, features[idx])
        predictions[idx] = int(np.argmax(probs))
    return predictions


def _evaluate_split(
    *,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    class_names: list[str],
) -> dict:
    if int(np.sum(mask)) == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "per_class_rows": [],
            "confusion_matrix": [[0 for _ in class_names] for _ in class_names],
            "top_confusion_pairs": [],
        }
    subset_features = features[mask]
    subset_labels = labels[mask]
    preds = _predict_labels(model, subset_features)
    return compute_classification_report(subset_labels, preds, class_names=class_names)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_algo.train")
    args = build_parser().parse_args()

    run_dir = Path(args.run_root) / str(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    _, data_cfg, _, _ = load_event_training_config(args.config)
    target_keys = _parse_target_keys(args.target_db5_keys)
    if target_keys is not None:
        data_cfg.target_db5_keys = list(target_keys)

    recordings_manifest = args.recordings_manifest or data_cfg.recordings_manifest_path
    resolved_recordings_manifest = _resolve_manifest_path(
        base_dir=Path(args.data_dir),
        raw=recordings_manifest,
        desc="recordings_manifest",
    )
    split_manifest_raw = args.split_manifest or data_cfg.split_manifest_path
    resolved_split_manifest = _resolve_manifest_path(
        base_dir=CODE_ROOT,
        raw=split_manifest_raw,
        desc="split_manifest",
    )

    loader = EventClipDatasetLoader(
        args.data_dir,
        data_cfg,
        recordings_manifest_path=resolved_recordings_manifest,
    )
    emg_samples, imu_samples, labels, source_ids = loader.load_all_with_sources()
    features = _build_feature_matrix(emg_samples, imu_samples)

    manifest = load_manifest(str(resolved_split_manifest))
    train_mask = _mask_by_sources(source_ids, set(manifest.train_sources))
    val_mask = _mask_by_sources(source_ids, set(manifest.val_sources))
    test_mask = _mask_by_sources(source_ids, set(manifest.test_sources))

    if int(np.sum(train_mask)) == 0:
        raise RuntimeError("No train windows matched split_manifest train_sources.")
    if int(np.sum(test_mask)) == 0:
        raise RuntimeError("No test windows matched split_manifest test_sources.")

    class_names = list(loader.label_spec.class_names)
    model = fit_event_algo_model(
        features[train_mask],
        labels[train_mask],
        class_names=class_names,
        temperature=float(args.temperature),
    )

    model_out = Path(args.algo_model_out) if str(args.algo_model_out or "").strip() else (run_dir / "models" / "algo_model.json")
    model_out = save_event_algo_model(model, model_out)

    val_report = _evaluate_split(
        model=model,
        features=features,
        labels=labels,
        mask=val_mask,
        class_names=class_names,
    )
    test_report = _evaluate_split(
        model=model,
        features=features,
        labels=labels,
        mask=test_mask,
        class_names=class_names,
    )

    eval_dir = run_dir / "evaluation"
    save_classification_report(test_report, eval_dir, prefix="test")
    save_classification_report(val_report, eval_dir, prefix="val")

    offline_summary = {
        "run_id": str(args.run_id),
        "model_type": "event_algo_nearest_centroid",
        "algo_model_path": str(model_out),
        "config_path": str(args.config),
        "recordings_manifest": str(resolved_recordings_manifest),
        "split_manifest": str(resolved_split_manifest),
        "target_db5_keys": ",".join(data_cfg.target_db5_keys),
        "temperature": float(args.temperature),
        "train_window_count": int(np.sum(train_mask)),
        "val_window_count": int(np.sum(val_mask)),
        "test_window_count": int(np.sum(test_mask)),
        "val_accuracy": float(val_report.get("accuracy", 0.0)),
        "val_macro_f1": float(val_report.get("macro_f1", 0.0)),
        "test_accuracy": float(test_report.get("accuracy", 0.0)),
        "test_macro_f1": float(test_report.get("macro_f1", 0.0)),
    }
    summary_path = run_dir / "offline_summary.json"
    summary_path.write_text(json.dumps(offline_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("algo_model=%s", model_out)
    logger.info("offline_summary=%s", summary_path)
    print(json.dumps(offline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

