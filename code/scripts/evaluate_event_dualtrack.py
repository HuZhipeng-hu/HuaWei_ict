"""Offline A/B evaluation for model vs algorithm recognizer backends."""

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
from event_onset.algo import EventAlgoPredictor
from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.inference import EventPredictor
from event_onset.runtime import EventOnsetController
from shared.config import load_config
from shared.label_modes import get_label_mode_spec
from training.data.split_strategy import load_manifest
from training.reporting import compute_classification_report


DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model/algo backends on same split and control policy")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--model_run_id", required=True)
    parser.add_argument("--eval_mode", default="dualtrack", choices=["dualtrack", "model_only"])
    parser.add_argument("--algo_model_path", default=None)
    parser.add_argument("--training_config", default="configs/training_event_onset_demo_p0.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--model_backend", default="ckpt", choices=["ckpt", "lite"])
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_metadata", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--target_event_action_accuracy_model", type=float, default=0.8)
    parser.add_argument("--target_event_action_accuracy_algo", type=float, default=0.9)
    parser.add_argument("--target_command_success_rate", type=float, default=0.9)
    parser.add_argument("--max_false_trigger_rate", type=float, default=0.05)
    parser.add_argument("--max_false_release_rate", type=float, default=0.05)
    return parser


def _parse_keys(raw: str) -> list[str]:
    keys = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not keys:
        raise ValueError("target_db5_keys is empty")
    return keys


def _apply_run_model_overrides(*, model_cfg, run_dir: Path, logger: logging.Logger) -> None:
    override_path = run_dir / "config_snapshots" / "effective_overrides.yaml"
    if not override_path.exists():
        return
    payload = load_config(override_path)
    model_section = dict(payload.get("model", {}) or {})
    if not model_section:
        return
    changes: list[str] = []
    if "base_channels" in model_section:
        model_cfg.base_channels = int(model_section["base_channels"])
        changes.append(f"base_channels={model_cfg.base_channels}")
    if "use_se" in model_section:
        model_cfg.use_se = bool(model_section["use_se"])
        changes.append(f"use_se={model_cfg.use_se}")
    if "dropout_rate" in model_section:
        model_cfg.dropout_rate = float(model_section["dropout_rate"])
        changes.append(f"dropout_rate={model_cfg.dropout_rate}")
    if "num_classes" in model_section:
        model_cfg.num_classes = int(model_section["num_classes"])
        changes.append(f"num_classes={model_cfg.num_classes}")
    if changes:
        logger.info("Applied run model overrides from %s: %s", override_path, ", ".join(changes))


def _resolve_split_manifest(
    *,
    split_manifest_arg: str | None,
    run_root: Path,
    model_run_id: str,
    training_data_cfg,
) -> Path:
    if str(split_manifest_arg or "").strip():
        return Path(str(split_manifest_arg)).resolve()

    offline_summary_path = run_root / str(model_run_id) / "offline_summary.json"
    if offline_summary_path.exists():
        payload = json.loads(offline_summary_path.read_text(encoding="utf-8"))
        candidate = str(payload.get("manifest_path", "")).strip()
        if candidate:
            return (CODE_ROOT / candidate).resolve()
    return (CODE_ROOT / str(training_data_cfg.split_manifest_path)).resolve()


def _validate_backend_class_contract(
    *,
    backend_name: str,
    expected_class_names: list[str],
    recognized_class_names: list[str],
) -> None:
    expected = [str(item).strip().upper() for item in expected_class_names]
    got = [str(item).strip().upper() for item in recognized_class_names]
    if got != expected:
        raise ValueError(
            f"{backend_name} class order mismatch: expected={expected}, got={got}"
        )


def _evaluate_window_metrics(
    *,
    predictor,
    emg_samples: np.ndarray,
    imu_samples: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    test_sources: set[str],
    class_names: list[str],
) -> dict:
    mask = np.asarray([str(source) in test_sources for source in source_ids], dtype=bool)
    if int(np.sum(mask)) == 0:
        raise RuntimeError("No test windows matched split_manifest test_sources.")

    test_emg = emg_samples[mask]
    test_imu = imu_samples[mask]
    y_true = labels[mask].astype(np.int32)
    y_pred = np.zeros_like(y_true)
    relax_idx = next((idx for idx, name in enumerate(class_names) if str(name).strip().upper() == "RELAX"), None)
    gate_accept_count = 0
    action_total = 0
    action_accept_count = 0
    stage2_total = 0
    stage2_correct = 0
    rule_hit_count = 0
    has_algo_meta = False
    for idx in range(test_emg.shape[0]):
        if hasattr(predictor, "predict_proba_with_meta"):
            probs, meta = predictor.predict_proba_with_meta(test_emg[idx], test_imu[idx])
            has_algo_meta = True
        else:
            probs = predictor.predict_proba(test_emg[idx], test_imu[idx])
            meta = {}
        probs = np.asarray(probs, dtype=np.float32)
        y_pred[idx] = int(np.argmax(probs))
        if has_algo_meta:
            gate_accepted = bool(meta.get("gate_accepted", False))
            stage2_used = bool(meta.get("stage2_used", False))
            rule_hit = bool(meta.get("rule_hit", False))
            if gate_accepted:
                gate_accept_count += 1
            if rule_hit:
                rule_hit_count += 1
            is_action = bool(relax_idx is not None and int(y_true[idx]) != int(relax_idx))
            if is_action:
                action_total += 1
                if gate_accepted:
                    action_accept_count += 1
                if stage2_used:
                    stage2_total += 1
                    if int(y_pred[idx]) == int(y_true[idx]):
                        stage2_correct += 1
    report = compute_classification_report(y_true, y_pred, class_names=class_names)
    if has_algo_meta:
        report.update(
            {
                "gate_accept_rate": float(gate_accept_count / test_emg.shape[0]) if test_emg.shape[0] else 0.0,
                "gate_action_recall": float(action_accept_count / action_total) if action_total else 0.0,
                "stage2_action_acc": float(stage2_correct / stage2_total) if stage2_total else 0.0,
                "rule_hit_rate": float(rule_hit_count / test_emg.shape[0]) if test_emg.shape[0] else 0.0,
            }
        )
    else:
        report.update(
            {
                "gate_accept_rate": 0.0,
                "gate_action_recall": 0.0,
                "stage2_action_acc": 0.0,
                "rule_hit_rate": 0.0,
            }
        )
    return report


def _evaluate_control_metrics(
    *,
    controller_factory,
    loader: EventClipDatasetLoader,
    test_sources: set[str],
    class_names: list[str],
    label_to_state: dict[int, object],
) -> dict:
    class_to_idx = {str(name).strip().upper(): int(idx) for idx, name in enumerate(class_names)}
    relax_state = label_to_state[0]
    action_states = {
        state
        for idx, state in label_to_state.items()
        if int(idx) != 0 and state != relax_state
    }
    release_command_labels = {
        int(label)
        for label, state in label_to_state.items()
        if int(label) != 0 and state == relax_state
    }

    total = 0
    action_total = 0
    release_command_total = 0
    command_success = 0
    false_release = 0
    false_trigger = 0

    for start_state, target_state, matrix, metadata in loader.iter_clips():
        source_id = str(metadata.get("relative_path", ""))
        if source_id not in test_sources:
            continue
        start_name = str(start_state).strip().upper()
        target_name = str(target_state).strip().upper()
        if start_name not in class_to_idx or target_name not in class_to_idx:
            continue

        controller = controller_factory()
        start_label = class_to_idx[start_name]
        target_label = class_to_idx[target_name]
        expected_state = label_to_state[target_label]

        controller.state_machine.current_label = int(start_label)
        controller.state_machine.current_state = label_to_state[int(start_label)]
        steps = controller.ingest_rows(np.asarray(matrix[:, :14], dtype=np.float32))
        transitions = [step for step in steps if bool(step.decision.changed)]
        total += 1

        if target_label == 0:
            triggered_action = any(step.decision.state in action_states for step in transitions)
            success = (not triggered_action) and (controller.current_state == relax_state)
            if triggered_action:
                false_trigger += 1
        elif target_label in release_command_labels:
            # command_only release classes (e.g. TENSE_OPEN -> RELAX) use a
            # dedicated success criterion and are excluded from false_release denominator.
            release_command_total += 1
            reached_target = any(step.decision.state == relax_state for step in transitions) or (
                controller.current_state == relax_state
            )
            wrong_action = any(step.decision.state in action_states for step in transitions)
            if wrong_action:
                false_trigger += 1
            success = reached_target and (not wrong_action)
        else:
            action_total += 1
            reached_idx = next(
                (idx for idx, step in enumerate(transitions) if step.decision.state == expected_state),
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
        "release_command_clip_count": int(release_command_total),
        "command_success_rate": float(command_success / total) if total else 0.0,
        "false_release_rate": float(false_release / action_total) if action_total else 0.0,
        "false_trigger_rate": float(false_trigger / total) if total else 0.0,
    }


def _rank_backend(metrics: dict) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("command_success_rate", 0.0)),
        -float(metrics.get("false_trigger_rate", 1.0)),
        -float(metrics.get("false_release_rate", 1.0)),
        float(metrics.get("event_action_accuracy", 0.0)),
        float(metrics.get("event_action_macro_f1", 0.0)),
    )


def _attach_acceptance_flags(
    metrics: dict,
    *,
    target_event_action_accuracy: float,
    target_command_success_rate: float,
    max_false_trigger_rate: float,
    max_false_release_rate: float,
) -> dict:
    payload = dict(metrics)
    payload["event_action_accuracy_target"] = float(target_event_action_accuracy)
    payload["event_action_accuracy_gap"] = float(target_event_action_accuracy) - float(
        payload.get("event_action_accuracy", 0.0)
    )
    payload["command_success_rate_target"] = float(target_command_success_rate)
    payload["false_trigger_rate_limit"] = float(max_false_trigger_rate)
    payload["false_release_rate_limit"] = float(max_false_release_rate)
    payload["pass_event_action_accuracy"] = bool(
        float(payload.get("event_action_accuracy", 0.0)) >= float(target_event_action_accuracy)
    )
    payload["pass_command_success_rate"] = bool(
        float(payload.get("command_success_rate", 0.0)) >= float(target_command_success_rate)
    )
    payload["pass_false_trigger_rate"] = bool(
        float(payload.get("false_trigger_rate", 1.0)) <= float(max_false_trigger_rate)
    )
    payload["pass_false_release_rate"] = bool(
        float(payload.get("false_release_rate", 1.0)) <= float(max_false_release_rate)
    )
    payload["pass_strict_online_gate"] = bool(
        payload["pass_command_success_rate"]
        and payload["pass_false_trigger_rate"]
        and payload["pass_false_release_rate"]
    )
    return payload


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_dualtrack_eval")
    args = build_parser().parse_args()

    run_root = Path(args.run_root)
    model_run_dir = run_root / str(args.model_run_id)
    if not model_run_dir.exists():
        raise FileNotFoundError(f"model run dir not found: {model_run_dir}")

    model_cfg, data_cfg, _, _ = load_event_training_config(args.training_config)
    runtime_cfg = load_event_runtime_config(args.runtime_config)
    _apply_run_model_overrides(model_cfg=model_cfg, run_dir=model_run_dir, logger=logger)
    target_keys = _parse_keys(args.target_db5_keys)
    data_cfg.target_db5_keys = list(target_keys)
    runtime_cfg.data.target_db5_keys = list(target_keys)

    if str(args.recordings_manifest or "").strip():
        data_cfg.recordings_manifest_path = str(args.recordings_manifest)
        runtime_cfg.data.recordings_manifest_path = str(args.recordings_manifest)

    split_manifest = _resolve_split_manifest(
        split_manifest_arg=args.split_manifest,
        run_root=run_root,
        model_run_id=str(args.model_run_id),
        training_data_cfg=data_cfg,
    )
    if not split_manifest.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest}")

    if str(args.checkpoint or "").strip():
        checkpoint_path = str(args.checkpoint)
    else:
        checkpoint_path = str(model_run_dir / "checkpoints" / "event_onset_best.ckpt")
    model_path = str(args.model_path) if str(args.model_path or "").strip() else str(runtime_cfg.model_path)
    model_metadata_path = (
        str(args.model_metadata)
        if str(args.model_metadata or "").strip()
        else str(runtime_cfg.model_metadata_path)
    )
    eval_mode = str(args.eval_mode).strip().lower()
    if eval_mode not in {"dualtrack", "model_only"}:
        raise ValueError(f"Unsupported eval_mode={args.eval_mode!r}.")

    algo_model_path: Path | None = None
    if eval_mode == "dualtrack":
        if not str(args.algo_model_path or "").strip():
            raise ValueError("dualtrack mode requires --algo_model_path.")
        algo_model_path = Path(str(args.algo_model_path)).resolve()
        if not algo_model_path.exists():
            raise FileNotFoundError(f"algo model not found: {algo_model_path}")

    label_spec = get_label_mode_spec(data_cfg.label_mode, data_cfg.target_db5_keys)
    model_cfg.num_classes = int(len(label_spec.class_names))
    label_to_state, mapping_by_name = load_and_validate_actuation_map(
        runtime_cfg.actuation_mapping_path,
        class_names=label_spec.class_names,
    )

    model_predictor = EventPredictor(
        backend=str(args.model_backend),
        model_config=model_cfg,
        device_target=str(args.device_target),
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        model_metadata_path=model_metadata_path,
    )
    model_class_names = (
        [str(item).strip().upper() for item in model_predictor.metadata.class_names]
        if model_predictor.metadata is not None and model_predictor.metadata.class_names
        else list(label_spec.class_names)
    )
    _validate_backend_class_contract(
        backend_name="model",
        expected_class_names=list(label_spec.class_names),
        recognized_class_names=model_class_names,
    )

    algo_predictor = None
    if eval_mode == "dualtrack":
        assert algo_model_path is not None
        algo_predictor = EventAlgoPredictor(model_path=algo_model_path)
        _validate_backend_class_contract(
            backend_name="algo",
            expected_class_names=list(label_spec.class_names),
            recognized_class_names=list(algo_predictor.class_names),
        )

    manifest = load_manifest(str(split_manifest))
    test_sources = set(manifest.test_sources)

    loader = EventClipDatasetLoader(
        str(args.data_dir),
        data_cfg,
        recordings_manifest_path=data_cfg.recordings_manifest_path,
    )
    emg_samples, imu_samples, labels, source_ids = loader.load_all_with_sources()

    class_names = list(label_spec.class_names)

    model_window = _evaluate_window_metrics(
        predictor=model_predictor,
        emg_samples=emg_samples,
        imu_samples=imu_samples,
        labels=labels,
        source_ids=source_ids,
        test_sources=test_sources,
        class_names=class_names,
    )
    algo_window = None
    if eval_mode == "dualtrack":
        assert algo_predictor is not None
        algo_window = _evaluate_window_metrics(
            predictor=algo_predictor,
            emg_samples=emg_samples,
            imu_samples=imu_samples,
            labels=labels,
            source_ids=source_ids,
            test_sources=test_sources,
            class_names=class_names,
        )

    def _build_controller_factory(predictor):
        return lambda: EventOnsetController(
            data_config=runtime_cfg.data,
            inference_config=runtime_cfg.inference,
            runtime_config=runtime_cfg.runtime,
            class_names=class_names,
            label_to_state=label_to_state,
            predict_proba=predictor.predict_proba,
            actuator=None,
        )

    model_control = _evaluate_control_metrics(
        controller_factory=_build_controller_factory(model_predictor),
        loader=loader,
        test_sources=test_sources,
        class_names=class_names,
        label_to_state=label_to_state,
    )
    algo_control = None
    if eval_mode == "dualtrack":
        assert algo_predictor is not None
        algo_control = _evaluate_control_metrics(
            controller_factory=_build_controller_factory(algo_predictor),
            loader=loader,
            test_sources=test_sources,
            class_names=class_names,
            label_to_state=label_to_state,
        )

    model_metrics = _attach_acceptance_flags(
        {
        "backend": "model",
        "model_backend": str(args.model_backend),
        "checkpoint_path": checkpoint_path,
        "model_path": model_path,
        "model_metadata_path": model_metadata_path,
        "window_test_accuracy": float(model_window.get("accuracy", 0.0)),
        "window_macro_f1": float(model_window.get("macro_f1", 0.0)),
        "event_action_accuracy": float(model_window.get("event_action_accuracy", 0.0)),
        "event_action_macro_f1": float(model_window.get("event_action_macro_f1", 0.0)),
        "gate_accept_rate": float(model_window.get("gate_accept_rate", 0.0)),
        "gate_action_recall": float(model_window.get("gate_action_recall", 0.0)),
        "stage2_action_acc": float(model_window.get("stage2_action_acc", 0.0)),
        "rule_hit_rate": float(model_window.get("rule_hit_rate", 0.0)),
        "command_success_rate": float(model_control.get("command_success_rate", 0.0)),
        "false_release_rate": float(model_control.get("false_release_rate", 0.0)),
        "false_trigger_rate": float(model_control.get("false_trigger_rate", 0.0)),
        "top_confusion_pairs": list(model_window.get("top_confusion_pairs", []))[:10],
        },
        target_event_action_accuracy=float(args.target_event_action_accuracy_model),
        target_command_success_rate=float(args.target_command_success_rate),
        max_false_trigger_rate=float(args.max_false_trigger_rate),
        max_false_release_rate=float(args.max_false_release_rate),
    )

    tracks = {"model": model_metrics}
    candidates = [model_metrics]

    if eval_mode == "dualtrack":
        assert algo_model_path is not None
        assert algo_window is not None
        assert algo_control is not None
        algo_metrics = _attach_acceptance_flags(
            {
                "backend": "algo",
                "algo_model_path": str(algo_model_path),
                "window_test_accuracy": float(algo_window.get("accuracy", 0.0)),
                "window_macro_f1": float(algo_window.get("macro_f1", 0.0)),
                "event_action_accuracy": float(algo_window.get("event_action_accuracy", 0.0)),
                "event_action_macro_f1": float(algo_window.get("event_action_macro_f1", 0.0)),
                "gate_accept_rate": float(algo_window.get("gate_accept_rate", 0.0)),
                "gate_action_recall": float(algo_window.get("gate_action_recall", 0.0)),
                "stage2_action_acc": float(algo_window.get("stage2_action_acc", 0.0)),
                "rule_hit_rate": float(algo_window.get("rule_hit_rate", 0.0)),
                "command_success_rate": float(algo_control.get("command_success_rate", 0.0)),
                "false_release_rate": float(algo_control.get("false_release_rate", 0.0)),
                "false_trigger_rate": float(algo_control.get("false_trigger_rate", 0.0)),
                "top_confusion_pairs": list(algo_window.get("top_confusion_pairs", []))[:10],
            },
            target_event_action_accuracy=float(args.target_event_action_accuracy_algo),
            target_command_success_rate=float(args.target_command_success_rate),
            max_false_trigger_rate=float(args.max_false_trigger_rate),
            max_false_release_rate=float(args.max_false_release_rate),
        )
        tracks["algo"] = algo_metrics
        candidates.append(algo_metrics)

    candidates_sorted = sorted(candidates, key=_rank_backend, reverse=True)
    recommended = dict(candidates_sorted[0])
    recommended_backend = str(recommended.get("backend"))
    alternative = dict(candidates_sorted[1]) if len(candidates_sorted) > 1 else None
    recommended_reason = {
        "primary": (
            "higher command_success_rate under same safety constraints"
            if eval_mode == "dualtrack"
            else "model_only mode"
        ),
        "recommended_command_success_rate": float(recommended.get("command_success_rate", 0.0)),
        "recommended_false_trigger_rate": float(recommended.get("false_trigger_rate", 1.0)),
        "recommended_false_release_rate": float(recommended.get("false_release_rate", 1.0)),
        "recommended_event_action_accuracy": float(recommended.get("event_action_accuracy", 0.0)),
    }
    if alternative is not None:
        recommended_reason["runner_up_backend"] = str(alternative.get("backend"))
        recommended_reason["runner_up_command_success_rate"] = float(alternative.get("command_success_rate", 0.0))
        recommended_reason["runner_up_event_action_accuracy"] = float(alternative.get("event_action_accuracy", 0.0))

    summary = {
        "status": "ok",
        "model_run_id": str(args.model_run_id),
        "eval_mode": eval_mode,
        "target_db5_keys": list(target_keys),
        "split_manifest": str(split_manifest),
        "recordings_manifest": str(data_cfg.recordings_manifest_path),
        "mapping": mapping_by_name,
        "rank_rule": (
            "command_success_rate desc, false_trigger_rate asc, false_release_rate asc, "
            "event_action_accuracy desc, event_action_macro_f1 desc"
        ),
        "tracks": tracks,
        "recommended_backend": recommended_backend,
        "recommended_metrics": recommended,
        "recommended_reason": recommended_reason,
        "targets": {
            "model_event_action_accuracy": float(args.target_event_action_accuracy_model),
            "algo_event_action_accuracy": (
                float(args.target_event_action_accuracy_algo) if eval_mode == "dualtrack" else None
            ),
            "command_success_rate": float(args.target_command_success_rate),
            "max_false_trigger_rate": float(args.max_false_trigger_rate),
            "max_false_release_rate": float(args.max_false_release_rate),
        },
    }

    output_path = (
        Path(str(args.output_json)).resolve()
        if str(args.output_json or "").strip()
        else (model_run_dir / "evaluation" / "dualtrack_summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("dualtrack_summary=%s", output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
