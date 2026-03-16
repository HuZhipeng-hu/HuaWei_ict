"""Evaluate demo runtime control behavior with latch + TENSE_OPEN release semantics."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.inference import EventPredictor
from event_onset.runtime import EventOnsetController
from shared.label_modes import get_label_mode_spec
from training.data.split_strategy import load_manifest


DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate event demo control metrics")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--training_config", default="configs/training_event_onset_demo_p0.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--backend", default="ckpt", choices=["ckpt", "lite"])
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_metadata", default=None)
    parser.add_argument("--output_json", default=None)
    return parser


def _parse_keys(raw: str) -> list[str]:
    keys = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not keys:
        raise ValueError("target_db5_keys is empty")
    return keys


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be object: {path}")
    return payload


def _validate_runtime_class_contract(
    *,
    backend: str,
    expected_class_names: list[str],
    model_num_classes: int,
    mapping_by_name: dict[str, str],
    metadata,
) -> None:
    normalized_expected = [str(name).strip().upper() for name in expected_class_names]
    if int(model_num_classes) != len(normalized_expected):
        raise ValueError(
            f"model.num_classes={model_num_classes} mismatches expected labels={len(normalized_expected)} "
            f"({normalized_expected})"
        )

    mapping_keys = sorted(str(key).strip().upper() for key in mapping_by_name.keys())
    if mapping_keys != sorted(normalized_expected):
        raise ValueError(
            f"Actuation mapping keys mismatch expected classes. mapping_keys={mapping_keys}, "
            f"expected={sorted(normalized_expected)}"
        )

    if metadata is None:
        if backend == "lite":
            raise ValueError("Lite backend requires model metadata with class_names for strict runtime validation.")
        return

    metadata_class_names = [str(name).strip().upper() for name in metadata.class_names]
    if not metadata_class_names:
        if backend == "lite":
            raise ValueError("Lite backend metadata must include non-empty class_names.")
        return
    if metadata_class_names != normalized_expected:
        raise ValueError(
            "Runtime class order mismatch between config and model metadata: "
            f"config={normalized_expected}, metadata={metadata_class_names}"
        )


def _resolve_split_manifest(*, args: argparse.Namespace, run_dir: Path, training_data_cfg) -> Path:
    if str(args.split_manifest or "").strip():
        return Path(args.split_manifest).resolve()

    offline_summary_path = run_dir / "offline_summary.json"
    if offline_summary_path.exists():
        offline = _load_json(offline_summary_path)
        candidate = str(offline.get("manifest_path", "")).strip()
        if candidate:
            return (CODE_ROOT / candidate).resolve()

    return (CODE_ROOT / str(training_data_cfg.split_manifest_path)).resolve()


def _load_window_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "evaluation" / "test_metrics.json"
    if not metrics_path.exists():
        return {
            "window_test_accuracy": 0.0,
            "window_macro_f1": 0.0,
            "event_action_accuracy": 0.0,
            "event_action_macro_f1": 0.0,
        }
    payload = _load_json(metrics_path)
    return {
        "window_test_accuracy": float(payload.get("accuracy", 0.0) or 0.0),
        "window_macro_f1": float(payload.get("macro_f1", 0.0) or 0.0),
        "event_action_accuracy": float(payload.get("event_action_accuracy", 0.0) or 0.0),
        "event_action_macro_f1": float(payload.get("event_action_macro_f1", 0.0) or 0.0),
    }


def _format_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _compute_sanity_flags(*, command_success_rate: float, false_trigger_rate: float) -> dict[str, bool]:
    metric_invariant_ok = not (float(false_trigger_rate) >= 0.95 and float(command_success_rate) > 0.05)
    return {"metric_invariant_ok": bool(metric_invariant_ok)}


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
    action_states = {label_to_state[idx] for idx in range(1, len(class_names))}

    total = 0
    action_total = 0
    relax_total = 0
    command_success = 0
    false_release = 0
    false_trigger = 0

    for start_state, target_state, matrix, metadata in loader.iter_clips():
        source_id = str(metadata.get("relative_path", ""))
        if source_id not in test_sources:
            continue

        target_name = str(target_state).strip().upper()
        start_name = str(start_state).strip().upper()
        if target_name not in class_to_idx or start_name not in class_to_idx:
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


def _run(args: argparse.Namespace) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_demo_control_eval")
    args = build_parser().parse_args()

    run_root = Path(args.run_root)
    run_dir = run_root / str(args.run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    model_cfg, data_cfg, _, _ = load_event_training_config(args.training_config)
    runtime_cfg = load_event_runtime_config(args.runtime_config)

    target_keys = _parse_keys(args.target_db5_keys)
    data_cfg.target_db5_keys = list(target_keys)
    runtime_cfg.data.target_db5_keys = list(target_keys)

    if str(args.recordings_manifest or "").strip():
        data_cfg.recordings_manifest_path = str(args.recordings_manifest)
        runtime_cfg.data.recordings_manifest_path = str(args.recordings_manifest)

    split_manifest = _resolve_split_manifest(args=args, run_dir=run_dir, training_data_cfg=data_cfg)
    if not split_manifest.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest}")

    if str(args.checkpoint or "").strip():
        runtime_cfg.checkpoint_path = str(args.checkpoint)
    else:
        runtime_cfg.checkpoint_path = str(run_dir / "checkpoints" / "event_onset_best.ckpt")

    if str(args.model_path or "").strip():
        runtime_cfg.model_path = str(args.model_path)
    if str(args.model_metadata or "").strip():
        runtime_cfg.model_metadata_path = str(args.model_metadata)

    label_spec = get_label_mode_spec(data_cfg.label_mode, data_cfg.target_db5_keys)
    model_cfg.num_classes = int(len(label_spec.class_names))

    label_to_state, mapping_by_name = load_and_validate_actuation_map(
        runtime_cfg.actuation_mapping_path,
        class_names=label_spec.class_names,
    )

    predictor = EventPredictor(
        backend=str(args.backend),
        model_config=model_cfg,
        device_target=str(args.device_target),
        checkpoint_path=runtime_cfg.checkpoint_path,
        model_path=runtime_cfg.model_path,
        model_metadata_path=runtime_cfg.model_metadata_path,
    )
    _validate_runtime_class_contract(
        backend=str(args.backend),
        expected_class_names=list(label_spec.class_names),
        model_num_classes=int(model_cfg.num_classes),
        mapping_by_name=mapping_by_name,
        metadata=predictor.metadata,
    )

    manifest = load_manifest(split_manifest)
    test_sources = set(manifest.test_sources)
    loader = EventClipDatasetLoader(
        str(args.data_dir),
        data_cfg,
        recordings_manifest_path=data_cfg.recordings_manifest_path,
    )

    def _controller_factory() -> EventOnsetController:
        return EventOnsetController(
            data_config=runtime_cfg.data,
            inference_config=runtime_cfg.inference,
            runtime_config=runtime_cfg.runtime,
            class_names=label_spec.class_names,
            label_to_state=label_to_state,
            predict_proba=predictor.predict_proba,
            actuator=None,
        )

    control = _evaluate_control_metrics(
        controller_factory=_controller_factory,
        loader=loader,
        test_sources=test_sources,
        class_names=list(label_spec.class_names),
        label_to_state=label_to_state,
    )
    sanity_flags = _compute_sanity_flags(
        command_success_rate=float(control["command_success_rate"]),
        false_trigger_rate=float(control["false_trigger_rate"]),
    )

    window_metrics = _load_window_metrics(run_dir)

    output = {
        "status": "ok",
        "run_id": str(args.run_id),
        "run_dir": str(run_dir),
        "runtime_config": str(args.runtime_config),
        "training_config": str(args.training_config),
        "checkpoint_path": str(runtime_cfg.checkpoint_path),
        "window_test_accuracy": float(window_metrics["window_test_accuracy"]),
        "window_macro_f1": float(window_metrics["window_macro_f1"]),
        "test_accuracy": float(window_metrics["window_test_accuracy"]),
        "test_macro_f1": float(window_metrics["window_macro_f1"]),
        "event_action_accuracy": float(window_metrics["event_action_accuracy"]),
        "event_action_macro_f1": float(window_metrics["event_action_macro_f1"]),
        "command_success_rate": float(control["command_success_rate"]),
        "false_release_rate": float(control["false_release_rate"]),
        "false_trigger_rate": float(control["false_trigger_rate"]),
        "sanity_flags": sanity_flags,
        "total_clip_count": int(control["total_clip_count"]),
        "action_clip_count": int(control["action_clip_count"]),
        "relax_clip_count": int(control["relax_clip_count"]),
        "target_db5_keys": list(target_keys),
        "mapping": mapping_by_name,
        "split_manifest": str(split_manifest),
        "recordings_manifest": str(data_cfg.recordings_manifest_path),
    }

    output_path = Path(args.output_json) if str(args.output_json or "").strip() else (run_dir / "evaluation" / "control_eval_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("control_eval_summary=%s", output_path)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    try:
        _run(args)
    except Exception as exc:
        run_root = Path(args.run_root)
        run_dir = run_root / str(args.run_id)
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        next_command = _format_cmd(
            [
                sys.executable,
                "scripts/evaluate_event_demo_control.py",
                "--run_root",
                str(args.run_root),
                "--run_id",
                str(args.run_id),
                "--training_config",
                str(args.training_config),
                "--runtime_config",
                str(args.runtime_config),
                "--data_dir",
                str(args.data_dir),
                "--recordings_manifest",
                str(args.recordings_manifest or ""),
                "--split_manifest",
                str(args.split_manifest or ""),
                "--target_db5_keys",
                str(args.target_db5_keys),
                "--backend",
                str(args.backend),
                "--device_target",
                str(args.device_target),
            ]
        )
        failure = {
            "status": "failed",
            "stage": "evaluate_event_demo_control",
            "root_cause": str(exc),
            "next_command": next_command,
        }
        failure_path = eval_dir / "control_eval_failure_report.json"
        failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
