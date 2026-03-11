"""Replay event-onset clips through the event runtime controller."""

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

from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.evaluate import load_event_model_from_checkpoint
from event_onset.runtime import EventOnsetController
from shared.gestures import GestureType
from training.data.split_strategy import load_manifest

try:
    import mindspore as ms
    from mindspore import Tensor, context
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay event-onset runtime benchmark from checkpoint")
    parser.add_argument("--training_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data_event_onset")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--output", default="artifacts/event_runtime_benchmark.json")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--mock", action="store_true")
    return parser


def _build_predictor(checkpoint_path: str | Path, model_config, device_target: str):
    if ms is None or Tensor is None or context is None:
        raise RuntimeError("MindSpore is not available")
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    model = load_event_model_from_checkpoint(checkpoint_path, model_config)

    def _predict(emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
        logits = model(Tensor(emg_feature[np.newaxis, ...], ms.float32), Tensor(imu_feature[np.newaxis, ...], ms.float32)).asnumpy()[0]
        exp_logits = np.exp(logits - np.max(logits))
        return (exp_logits / np.sum(exp_logits)).astype(np.float32)

    return _predict


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    model_cfg, data_cfg, _, _ = load_event_training_config(args.training_config)
    runtime_cfg = load_event_runtime_config(args.runtime_config)
    manifest_path = args.split_manifest or data_cfg.split_manifest_path
    manifest = load_manifest(manifest_path)
    test_sources = set(manifest.test_sources)

    if args.mock:
        rng = np.random.default_rng(42)

        def predictor(_emg, _imu):
            logits = rng.standard_normal(model_cfg.num_classes).astype(np.float32)
            exp_logits = np.exp(logits - np.max(logits))
            return (exp_logits / np.sum(exp_logits)).astype(np.float32)

    else:
        predictor = _build_predictor(args.checkpoint or runtime_cfg.checkpoint_path, model_cfg, args.device_target)

    loader = EventClipDatasetLoader(args.data_dir, data_cfg, recordings_manifest_path=args.recordings_manifest or data_cfg.recordings_manifest_path)
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
            predict_proba=predictor,
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

    report = {
        "manifest_path": manifest_path,
        "checkpoint_path": args.checkpoint or runtime_cfg.checkpoint_path,
        "transition_hit_rate": float(transition_hits / transition_total) if transition_total else 0.0,
        "false_trigger_rate": float(false_triggers / max(1, transition_total + release_total)),
        "state_hold_accuracy": float(hold_correct / hold_total) if hold_total else 0.0,
        "release_accuracy": float(release_correct / release_total) if release_total else 0.0,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "clips": clip_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
