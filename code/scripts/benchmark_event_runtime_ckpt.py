"""Replay event-onset clips through runtime controller for CKPT/Lite benchmarking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.inference import EventPredictor
from event_onset.runtime import EventOnsetController
from shared.gestures import GestureType
from training.data.split_strategy import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay event-onset runtime benchmark")
    parser.add_argument("--training_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_metadata", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--output", default="artifacts/event_runtime_benchmark.json")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--backend", default="both", choices=["ckpt", "lite", "both"])
    parser.add_argument("--mock", action="store_true")
    return parser


def _build_mock_predictor(seed: int, num_classes: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)

    def _predict(_emg: np.ndarray, _imu: np.ndarray) -> np.ndarray:
        logits = rng.standard_normal(num_classes).astype(np.float32)
        exp_logits = np.exp(logits - np.max(logits))
        return (exp_logits / np.sum(exp_logits)).astype(np.float32)

    return _predict


def _benchmark_backend(
    *,
    backend_name: str,
    predict_proba: Callable[[np.ndarray, np.ndarray], np.ndarray],
    loader: EventClipDatasetLoader,
    runtime_cfg,
    test_sources: set[str],
) -> dict:
    transition_hits = 0
    transition_total = 0
    false_triggers = 0
    hold_correct = 0
    hold_total = 0
    release_correct = 0
    release_total = 0
    latencies: list[float] = []
    clip_results: list[dict[str, object]] = []

    for start_state, target_state, matrix, metadata in loader.iter_clips():
        source_id = metadata["relative_path"]
        if source_id not in test_sources:
            continue
        controller = EventOnsetController(
            data_config=runtime_cfg.data,
            inference_config=runtime_cfg.inference,
            runtime_config=runtime_cfg.runtime,
            predict_proba=predict_proba,
        )
        controller.state_machine.current_state = getattr(GestureType, start_state)
        steps = controller.ingest_rows(matrix[:, :14])
        transitions = [step for step in steps if step.decision.changed]
        expected_state = getattr(GestureType, target_state)

        if target_state == "RELAX":
            release_total += 1
            if controller.current_state == GestureType.RELAX:
                release_correct += 1
            if any(step.decision.state != GestureType.RELAX for step in transitions):
                false_triggers += 1
        else:
            transition_total += 1
            first_target = next((step for step in transitions if step.decision.state == expected_state), None)
            if first_target is not None:
                transition_hits += 1
                latencies.append(max(0.0, first_target.now_ms - float(metadata.get("pre_roll_ms") or 0)))
            if any(step.decision.state not in {getattr(GestureType, start_state), expected_state} for step in transitions):
                false_triggers += 1
            hold_total += 1
            if controller.current_state == expected_state:
                hold_correct += 1

        clip_results.append(
            {
                "source_id": source_id,
                "start_state": start_state,
                "target_state": target_state,
                "transition_count": len(transitions),
                "final_state": controller.current_state.name,
            }
        )

    return {
        "backend": backend_name,
        "transition_hit_rate": float(transition_hits / transition_total) if transition_total else 0.0,
        "false_trigger_rate": float(false_triggers / max(1, transition_total + release_total)),
        "state_hold_accuracy": float(hold_correct / hold_total) if hold_total else 0.0,
        "release_accuracy": float(release_correct / release_total) if release_total else 0.0,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "clips": clip_results,
    }


def _compare_metrics(ckpt_report: dict, lite_report: dict) -> dict:
    keys = [
        "transition_hit_rate",
        "false_trigger_rate",
        "state_hold_accuracy",
        "release_accuracy",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    return {f"delta_{key}": float(lite_report[key]) - float(ckpt_report[key]) for key in keys}


def _merge_gate(ckpt_report: dict, lite_report: dict) -> dict:
    checks = {
        "transition_hit_rate": float(lite_report["transition_hit_rate"]) >= float(ckpt_report["transition_hit_rate"]) - 0.03,
        "false_trigger_rate": float(lite_report["false_trigger_rate"]) <= float(ckpt_report["false_trigger_rate"]) + 0.03,
        "state_hold_accuracy": float(lite_report["state_hold_accuracy"]) >= float(ckpt_report["state_hold_accuracy"]) - 0.03,
        "release_accuracy": float(lite_report["release_accuracy"]) >= float(ckpt_report["release_accuracy"]) - 0.03,
        "latency_p95_ms": float(lite_report["latency_p95_ms"]) <= float(ckpt_report["latency_p95_ms"]) * 1.30,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_runtime_benchmark")
    args = build_parser().parse_args()

    model_cfg, data_cfg, _, _ = load_event_training_config(args.training_config)
    runtime_cfg = load_event_runtime_config(args.runtime_config)
    runtime_cfg.device.target = args.device_target
    if args.checkpoint:
        runtime_cfg.checkpoint_path = args.checkpoint
    if args.model_path:
        runtime_cfg.model_path = args.model_path
    if args.model_metadata:
        runtime_cfg.model_metadata_path = args.model_metadata

    manifest_path = args.split_manifest or data_cfg.split_manifest_path
    manifest = load_manifest(manifest_path)
    test_sources = set(manifest.test_sources)
    loader = EventClipDatasetLoader(args.data_dir, data_cfg, recordings_manifest_path=args.recordings_manifest or data_cfg.recordings_manifest_path)

    backends = [args.backend] if args.backend != "both" else ["ckpt", "lite"]
    reports: dict[str, dict] = {}

    for backend in backends:
        if args.mock:
            predictor_fn = _build_mock_predictor(seed=42 if backend == "ckpt" else 123, num_classes=model_cfg.num_classes)
        else:
            predictor = EventPredictor(
                backend=backend,
                model_config=model_cfg,
                device_target=args.device_target,
                checkpoint_path=runtime_cfg.checkpoint_path,
                model_path=runtime_cfg.model_path,
                model_metadata_path=runtime_cfg.model_metadata_path,
            )
            predictor_fn = predictor.predict_proba
        logger.info("Running event benchmark for backend=%s", backend)
        reports[backend] = _benchmark_backend(
            backend_name=backend,
            predict_proba=predictor_fn,
            loader=loader,
            runtime_cfg=runtime_cfg,
            test_sources=test_sources,
        )

    output_payload = {
        "manifest_path": manifest_path,
        "checkpoint_path": runtime_cfg.checkpoint_path,
        "model_path": runtime_cfg.model_path,
        "reports": reports,
    }
    if "ckpt" in reports and "lite" in reports:
        output_payload["comparison"] = _compare_metrics(reports["ckpt"], reports["lite"])
        output_payload["merge_gate"] = _merge_gate(reports["ckpt"], reports["lite"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
