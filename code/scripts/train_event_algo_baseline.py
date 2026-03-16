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

from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.algo import (
    ALGO_MODE_V1,
    ALGO_MODE_V2,
    EventAlgoModel,
    build_event_algo_feature_vector,
    compute_algo_stage_metrics,
    fit_event_algo_model,
    predict_algo_proba_with_meta_from_vector,
    save_event_algo_model,
)
from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.runtime import EventOnsetController
from training.data.split_strategy import load_manifest
from training.reporting import compute_classification_report, save_classification_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lightweight algo recognizer for event-onset runtime")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--target_db5_keys", default=None)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--algo_mode", default=ALGO_MODE_V2, choices=[ALGO_MODE_V1, ALGO_MODE_V2])
    parser.add_argument("--gate_action_threshold", type=float, default=0.55)
    parser.add_argument("--gate_margin_threshold", type=float, default=0.05)
    parser.add_argument("--wrist_rule_min", type=float, default=0.55)
    parser.add_argument("--wrist_rule_margin", type=float, default=0.10)
    parser.add_argument("--release_emg_min", type=float, default=0.45)
    parser.add_argument("--release_imu_max", type=float, default=1.50)
    parser.add_argument("--rule_confidence", type=float, default=0.94)
    parser.add_argument("--actuation_mapping", default=None)
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


def _predict_labels(model: EventAlgoModel, features: np.ndarray) -> np.ndarray:
    predictions = np.zeros((features.shape[0],), dtype=np.int32)
    for idx in range(features.shape[0]):
        probs, _ = predict_algo_proba_with_meta_from_vector(model, features[idx])
        predictions[idx] = int(np.argmax(probs))
    return predictions


def _evaluate_split(
    *,
    model: EventAlgoModel,
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
            "gate_accept_rate": 0.0,
            "gate_action_recall": 0.0,
            "stage2_action_acc": 0.0,
            "rule_hit_rate": 0.0,
        }
    subset_features = features[mask]
    subset_labels = labels[mask]
    preds = _predict_labels(model, subset_features)
    report = compute_classification_report(subset_labels, preds, class_names=class_names)
    report.update(
        compute_algo_stage_metrics(
            model,
            subset_features,
            subset_labels,
            class_names=class_names,
        )
    )
    return report


def _evaluate_control_metrics(
    *,
    loader: EventClipDatasetLoader,
    allowed_sources: set[str],
    class_names: list[str],
    runtime_cfg,
    label_to_state: dict[int, object],
    predict_proba,
) -> dict:
    class_to_idx = {str(name).strip().upper(): int(idx) for idx, name in enumerate(class_names)}
    relax_state = label_to_state[0]
    action_states = {label_to_state[idx] for idx in range(1, len(class_names))}

    total = 0
    action_total = 0
    relax_total = 0
    command_success = 0
    false_release = 0
    false_trigger = 0

    for start_state, target_state, matrix, metadata in loader.iter_clips():
        source_id = str(metadata.get("relative_path", ""))
        if source_id not in allowed_sources:
            continue

        start_name = str(start_state).strip().upper()
        target_name = str(target_state).strip().upper()
        if start_name not in class_to_idx or target_name not in class_to_idx:
            continue

        controller = EventOnsetController(
            data_config=runtime_cfg.data,
            inference_config=runtime_cfg.inference,
            runtime_config=runtime_cfg.runtime,
            class_names=class_names,
            label_to_state=label_to_state,
            predict_proba=predict_proba,
            actuator=None,
        )

        start_label = class_to_idx[start_name]
        target_label = class_to_idx[target_name]
        expected_state = label_to_state[target_label]

        controller.state_machine.current_label = int(start_label)
        controller.state_machine.current_state = label_to_state[int(start_label)]
        steps = controller.ingest_rows(np.asarray(matrix[:, :14], dtype=np.float32))
        transitions = [step for step in steps if bool(step.decision.changed)]

        total += 1
        if target_label == 0:
            relax_total += 1
            triggered_action = any(step.decision.state in action_states for step in transitions)
            success = (not triggered_action) and (controller.current_state == relax_state)
            if triggered_action:
                false_trigger += 1
        else:
            action_total += 1
            reached_idx = next(
                (
                    idx
                    for idx, step in enumerate(transitions)
                    if step.decision.state == expected_state
                ),
                None,
            )
            reached_target = reached_idx is not None or controller.current_state == expected_state
            wrong_action = any(
                (step.decision.state in action_states) and (step.decision.state != expected_state)
                for step in transitions
            )
            released_after_target = False
            if reached_idx is not None:
                released_after_target = any(step.decision.state == relax_state for step in transitions[reached_idx + 1 :])

            if released_after_target:
                false_release += 1
            if wrong_action:
                false_trigger += 1

            success = reached_target and (not wrong_action) and (not released_after_target)

        if success:
            command_success += 1

    return {
        "total_clip_count": int(total),
        "action_clip_count": int(action_total),
        "relax_clip_count": int(relax_total),
        "command_success_rate": float(command_success / total) if total else 0.0,
        "false_release_rate": float(false_release / action_total) if action_total else 0.0,
        "false_trigger_rate": float(false_trigger / total) if total else 0.0,
    }


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
    runtime_cfg = load_event_runtime_config(args.runtime_config)

    target_keys = _parse_target_keys(args.target_db5_keys)
    if target_keys is not None:
        data_cfg.target_db5_keys = list(target_keys)
    runtime_cfg.data.target_db5_keys = list(data_cfg.target_db5_keys)

    recordings_manifest = args.recordings_manifest or data_cfg.recordings_manifest_path
    resolved_recordings_manifest = _resolve_manifest_path(
        base_dir=Path(args.data_dir),
        raw=recordings_manifest,
        desc="recordings_manifest",
    )
    data_cfg.recordings_manifest_path = str(resolved_recordings_manifest)
    runtime_cfg.data.recordings_manifest_path = str(resolved_recordings_manifest)

    split_manifest_raw = args.split_manifest or data_cfg.split_manifest_path
    resolved_split_manifest = _resolve_manifest_path(
        base_dir=CODE_ROOT,
        raw=split_manifest_raw,
        desc="split_manifest",
    )

    if str(args.actuation_mapping or "").strip():
        runtime_cfg.actuation_mapping_path = str(args.actuation_mapping)

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
        algo_mode=str(args.algo_mode),
        gate_action_threshold=float(args.gate_action_threshold),
        gate_margin_threshold=float(args.gate_margin_threshold),
    )
    model = EventAlgoModel(
        class_names=model.class_names,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        centroids=model.centroids,
        temperature=model.temperature,
        rule_config={
            "enabled": True,
            "wrist_rule_min": float(args.wrist_rule_min),
            "wrist_rule_margin": float(args.wrist_rule_margin),
            "release_emg_min": float(args.release_emg_min),
            "release_imu_max": float(args.release_imu_max),
            "rule_confidence": float(args.rule_confidence),
        },
        algo_mode=model.algo_mode,
        gate_feature_mean=model.gate_feature_mean,
        gate_feature_std=model.gate_feature_std,
        gate_centroids=model.gate_centroids,
        gate_action_threshold=model.gate_action_threshold,
        gate_margin_threshold=model.gate_margin_threshold,
        action_class_names=model.action_class_names,
        action_feature_mean=model.action_feature_mean,
        action_feature_std=model.action_feature_std,
        action_centroids=model.action_centroids,
    )

    model_out = (
        Path(args.algo_model_out)
        if str(args.algo_model_out or "").strip()
        else (run_dir / "models" / "algo_model.json")
    )
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

    label_to_state, _ = load_and_validate_actuation_map(
        runtime_cfg.actuation_mapping_path,
        class_names=class_names,
    )

    def _predict_proba(emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
        vector = build_event_algo_feature_vector(emg_feature, imu_feature)
        probs, _ = predict_algo_proba_with_meta_from_vector(model, vector)
        return probs.astype(np.float32)

    val_control = _evaluate_control_metrics(
        loader=loader,
        allowed_sources=set(manifest.val_sources),
        class_names=class_names,
        runtime_cfg=runtime_cfg,
        label_to_state=label_to_state,
        predict_proba=_predict_proba,
    )
    test_control = _evaluate_control_metrics(
        loader=loader,
        allowed_sources=set(manifest.test_sources),
        class_names=class_names,
        runtime_cfg=runtime_cfg,
        label_to_state=label_to_state,
        predict_proba=_predict_proba,
    )

    eval_dir = run_dir / "evaluation"
    save_classification_report(test_report, eval_dir, prefix="test")
    save_classification_report(val_report, eval_dir, prefix="val")

    offline_summary = {
        "run_id": str(args.run_id),
        "model_type": "event_algo_nearest_centroid",
        "algo_mode": str(model.algo_mode),
        "algo_model_path": str(model_out),
        "config_path": str(args.config),
        "runtime_config_path": str(args.runtime_config),
        "recordings_manifest": str(resolved_recordings_manifest),
        "split_manifest": str(resolved_split_manifest),
        "target_db5_keys": ",".join(data_cfg.target_db5_keys),
        "temperature": float(args.temperature),
        "gate_action_threshold": float(model.gate_action_threshold),
        "gate_margin_threshold": float(model.gate_margin_threshold),
        "rule_config": dict(model.rule_config or {}),
        "train_window_count": int(np.sum(train_mask)),
        "val_window_count": int(np.sum(val_mask)),
        "test_window_count": int(np.sum(test_mask)),
        "val_accuracy": float(val_report.get("accuracy", 0.0)),
        "val_macro_f1": float(val_report.get("macro_f1", 0.0)),
        "val_gate_accept_rate": float(val_report.get("gate_accept_rate", 0.0)),
        "val_gate_action_recall": float(val_report.get("gate_action_recall", 0.0)),
        "val_stage2_action_acc": float(val_report.get("stage2_action_acc", 0.0)),
        "val_rule_hit_rate": float(val_report.get("rule_hit_rate", 0.0)),
        "val_command_success_rate": float(val_control.get("command_success_rate", 0.0)),
        "val_false_trigger_rate": float(val_control.get("false_trigger_rate", 0.0)),
        "val_false_release_rate": float(val_control.get("false_release_rate", 0.0)),
        "test_accuracy": float(test_report.get("accuracy", 0.0)),
        "test_macro_f1": float(test_report.get("macro_f1", 0.0)),
        "test_gate_accept_rate": float(test_report.get("gate_accept_rate", 0.0)),
        "test_gate_action_recall": float(test_report.get("gate_action_recall", 0.0)),
        "test_stage2_action_acc": float(test_report.get("stage2_action_acc", 0.0)),
        "test_rule_hit_rate": float(test_report.get("rule_hit_rate", 0.0)),
        "test_command_success_rate": float(test_control.get("command_success_rate", 0.0)),
        "test_false_trigger_rate": float(test_control.get("false_trigger_rate", 0.0)),
        "test_false_release_rate": float(test_control.get("false_release_rate", 0.0)),
    }
    summary_path = run_dir / "offline_summary.json"
    summary_path.write_text(json.dumps(offline_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("algo_model=%s", model_out)
    logger.info("offline_summary=%s", summary_path)
    print(json.dumps(offline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
