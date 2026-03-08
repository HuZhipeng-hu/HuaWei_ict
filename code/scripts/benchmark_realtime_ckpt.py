"""Trusted dual-branch realtime benchmark by replaying test recordings."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    import mindspore as ms
    from mindspore import Tensor, context
except Exception:  # pragma: no cover
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore

from runtime.control.controller import RuntimeController
from runtime.inference import InferenceRateScheduler, TemporalVoter
from shared.config import (
    get_protocol_input_shape,
    normalize_model_config_channels,
    load_runtime_config,
    load_training_config,
    load_training_data_config,
)
from shared.gestures import GESTURE_DEFINITIONS, GestureType
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, ensure_run_dir
from training.data.csv_dataset import CSVDatasetLoader
from training.data.split_strategy import load_manifest
from training.evaluate import load_model_from_checkpoint
from training.reporting import compute_classification_report, save_classification_report

logger = logging.getLogger("realtime_benchmark")

BENCHMARK_SUMMARY_FIELDS = [
    "run_id",
    "checkpoint_path",
    "manifest_path",
    "model_type",
    "base_channels",
    "use_se",
    "threshold",
    "hysteresis_count",
    "infer_rate_hz",
    "decision_count",
    "hit_rate",
    "false_trigger_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
]

COMBINED_SUMMARY_FIELDS = [
    "run_id",
    "manifest_path",
    "checkpoint_path",
    "model_type",
    "base_channels",
    "use_se",
    "loss_type",
    "hard_mining_ratio",
    "augment_enabled",
    "augment_factor",
    "use_mixup",
    "test_accuracy",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
    "threshold",
    "hysteresis_count",
    "infer_rate_hz",
    "decision_count",
    "hit_rate",
    "false_trigger_rate",
    "latency_p50_ms",
    "latency_p95_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay trusted realtime benchmark from checkpoint")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--output_dir", default="realtime_benchmark")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--hysteresis_count", type=int, default=None)
    parser.add_argument("--infer_rate_hz", type=float, default=None)
    parser.add_argument("--mock", action="store_true", help="Force mock inference for local smoke tests")
    return parser.parse_args()


def _set_device(device_target: str, device_id: int) -> None:
    if ms is None or context is None:
        raise RuntimeError("MindSpore is not available")
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    if device_target.upper() == "GPU":
        context.set_context(device_id=device_id)


def _normalize_model_config(model_cfg, preprocess_cfg):
    return normalize_model_config_channels(
        model_cfg,
        preprocess_cfg,
        logger=logger,
        context="realtime benchmark dual-branch protocol",
    )


def build_runtime_benchmark_plan(runtime_cfg) -> Dict[str, int | tuple[int, int, int, int]]:
    from shared.preprocessing import PreprocessPipeline

    preprocess = PreprocessPipeline(runtime_cfg.preprocess)
    base_window = preprocess.get_required_window_size()
    stride = preprocess.get_required_window_stride()
    read_window_size = RuntimeController._calc_read_window_size(base_window, stride, list(runtime_cfg.inference.tta_offsets or [0.0]))
    if runtime_cfg.control_rate_hz > 0:
        cycle_step_samples = max(1, int(round(runtime_cfg.device.sampling_rate / float(runtime_cfg.control_rate_hz))))
    else:
        cycle_step_samples = max(1, stride)
    return {
        "base_window": base_window,
        "stride": stride,
        "read_window_size": read_window_size,
        "cycle_step_samples": cycle_step_samples,
        "expected_input_shape": get_protocol_input_shape(runtime_cfg.preprocess),
    }


def _iter_control_windows(
    signal: np.ndarray,
    read_window_size: int,
    cycle_step_samples: int,
    min_window_size: Optional[int] = None,
) -> Iterator[tuple[float, np.ndarray]]:
    effective_window_size = min(int(read_window_size), int(signal.shape[0]))
    required_window_size = int(min_window_size or effective_window_size)
    if effective_window_size < required_window_size:
        return
    total_cycles = 0
    for end in range(effective_window_size, signal.shape[0] + 1, cycle_step_samples):
        start = end - effective_window_size
        yield float(total_cycles), signal[start:end]
        total_cycles += 1


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def _predict_probs(model, mock_mode: bool, feature: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if mock_mode:
        logits = rng.standard_normal(len(GESTURE_DEFINITIONS)).astype(np.float32)
        return _softmax(logits).astype(np.float32)
    assert ms is not None and Tensor is not None
    logits = model(Tensor(feature[np.newaxis, ...], ms.float32)).asnumpy()[0]
    return _softmax(logits).astype(np.float32)


def _top_confusion_pair_text(report: dict) -> str:
    pairs = report.get("top_confusion_pairs") or []
    if not pairs:
        return ""
    pair = pairs[0]
    return f"{pair['pair'][0]}<->{pair['pair'][1]}:{pair['count']}"


def _aggregate_decisions(decisions: List[Dict[str, float | int | str]], class_names: List[str], relax_id: int) -> dict:
    y_true = np.asarray([int(row["true_label"]) for row in decisions], dtype=np.int32)
    y_pred = np.asarray([int(row["emitted_label"]) for row in decisions], dtype=np.int32)
    latencies = np.asarray([float(row["latency_ms"]) for row in decisions], dtype=np.float32)

    report = compute_classification_report(y_true=y_true, y_pred=y_pred, class_names=class_names)
    active_mask = y_true != relax_id
    relax_mask = y_true == relax_id
    hit_rate = float(np.mean(y_pred[active_mask] == y_true[active_mask])) if np.any(active_mask) else 0.0
    false_trigger_rate = float(np.mean(y_pred[relax_mask] != relax_id)) if np.any(relax_mask) else 0.0

    return {
        "decision_count": int(len(decisions)),
        "hit_rate": hit_rate,
        "false_trigger_rate": false_trigger_rate,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
        "report": report,
    }


def _write_decisions_csv(path: Path, rows: Iterable[Dict[str, float | int | str]]) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "gesture",
                "cycle_index",
                "true_label",
                "raw_pred_label",
                "emitted_label",
                "confidence",
                "latency_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="realtime")
    output_dir = run_dir / args.output_dir

    training_cfg_path = Path(args.training_config)
    runtime_cfg_path = Path(args.runtime_config)
    copy_config_snapshot(training_cfg_path, run_dir / "config_snapshots" / training_cfg_path.name)
    copy_config_snapshot(runtime_cfg_path, run_dir / "config_snapshots" / runtime_cfg_path.name)

    model_cfg, preprocess_cfg, train_cfg, _ = load_training_config(training_cfg_path)
    data_cfg = load_training_data_config(training_cfg_path)
    runtime_cfg = load_runtime_config(runtime_cfg_path)
    model_cfg = _normalize_model_config(model_cfg, preprocess_cfg)

    manifest_path = args.split_manifest or data_cfg.split_manifest_path
    if not manifest_path:
        raise ValueError("split manifest path is required via --split_manifest or training data config")
    copy_config_snapshot(manifest_path, run_dir / "manifests" / Path(manifest_path).name)

    threshold = runtime_cfg.inference.confidence_threshold if args.threshold is None else args.threshold
    hysteresis_count = runtime_cfg.inference.hysteresis_count if args.hysteresis_count is None else args.hysteresis_count
    infer_rate_hz = runtime_cfg.inference.infer_rate_hz if args.infer_rate_hz is None else args.infer_rate_hz
    runtime_cfg.inference.hysteresis_count = int(hysteresis_count)
    runtime_cfg.inference.infer_rate_hz = float(infer_rate_hz)
    runtime_cfg.infer_rate_hz = float(infer_rate_hz)

    plan = build_runtime_benchmark_plan(runtime_cfg)
    logger.info("Run ID: %s", run_id)
    logger.info("Benchmark plan: %s", plan)

    gesture_to_idx = {g.name: i for i, g in enumerate(GESTURE_DEFINITIONS)}
    class_names = [g.name for g in GESTURE_DEFINITIONS]
    relax_id = int(GestureType.RELAX)

    loader = CSVDatasetLoader(
        args.data_dir,
        gesture_to_idx,
        runtime_cfg.preprocess,
        quality_filter=train_cfg.quality_filter,
        recordings_manifest_path=args.recordings_manifest or data_cfg.recordings_manifest_path,
    )
    manifest = load_manifest(manifest_path)
    test_sources = set(manifest.test_sources)

    mock_mode = bool(args.mock or ms is None)
    if mock_mode:
        logger.warning("Realtime benchmark running in mock inference mode.")
        model = None
    else:
        _set_device(args.device_target, args.device_id)
        model = load_model_from_checkpoint(args.checkpoint, model_cfg, dropout_rate=model_cfg.dropout_rate)

    from shared.preprocessing import PreprocessPipeline

    preprocess = PreprocessPipeline(runtime_cfg.preprocess)
    expected_shape = tuple(plan["expected_input_shape"])
    dummy = np.zeros((int(plan["base_window"]), runtime_cfg.preprocess.num_channels), dtype=np.float32)
    actual_shape = (1,) + tuple(preprocess.process_window(dummy).shape)
    if actual_shape != expected_shape:
        raise ValueError(f"Benchmark preprocess shape mismatch: {actual_shape} != {expected_shape}")
    if model_cfg.in_channels != expected_shape[1]:
        raise ValueError(
            f"Model/preprocess channel mismatch: model.in_channels={model_cfg.in_channels}, expected={expected_shape[1]}"
        )

    decisions: List[Dict[str, float | int | str]] = []
    rng = np.random.default_rng(42)
    cycle_dt_sec = 0.0 if runtime_cfg.control_rate_hz <= 0 else 1.0 / float(runtime_cfg.control_rate_hz)

    for gesture_name, class_id, signal, metadata in loader.iter_recordings():
        if metadata["source_id"] not in test_sources:
            continue

        scheduler = InferenceRateScheduler(infer_rate_hz=float(infer_rate_hz))
        voter = TemporalVoter(
            history_window_ms=runtime_cfg.inference.smoothing_window_ms,
            hysteresis_count=int(hysteresis_count),
        )
        windows = _iter_control_windows(
            signal,
            int(plan["read_window_size"]),
            int(plan["cycle_step_samples"]),
            min_window_size=int(plan["base_window"]),
        )
        for cycle_index, (cycle_number, raw_window) in enumerate(windows):
            now_sec = float(cycle_number) * cycle_dt_sec
            if not scheduler.should_run(now=now_sec):
                continue
            slices = RuntimeController._slice_tta_windows(
                raw_window,
                int(plan["base_window"]),
                int(plan["stride"]),
                list(runtime_cfg.inference.tta_offsets or [0.0]),
            )
            if not slices:
                continue

            probs = []
            t0 = time.perf_counter()
            for segment in slices:
                feature = preprocess.process_window(segment)
                if (1,) + tuple(feature.shape) != expected_shape:
                    raise ValueError(
                        f"Benchmark feature shape mismatch for {metadata['source_id']}: {(1,) + tuple(feature.shape)} != {expected_shape}"
                    )
                probs.append(_predict_probs(model, mock_mode, feature, rng))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
            raw_pred_label = int(np.argmax(mean_prob))
            confidence = float(np.max(mean_prob))
            stable = voter.update(raw_pred_label, confidence, now=now_sec)
            if stable is not None and confidence >= threshold:
                emitted_label = int(stable)
            else:
                emitted_label = relax_id

            decisions.append(
                {
                    "source_id": metadata["source_id"],
                    "gesture": gesture_name,
                    "cycle_index": cycle_index,
                    "true_label": int(class_id),
                    "raw_pred_label": raw_pred_label,
                    "emitted_label": emitted_label,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                }
            )

    if not decisions:
        raise RuntimeError("No realtime benchmark decisions were generated. Check manifest and raw recordings.")

    aggregated = _aggregate_decisions(decisions, class_names, relax_id)
    report = aggregated.pop("report")
    report.update(
        {
            "run_id": run_id,
            "checkpoint_path": args.checkpoint,
            "manifest_path": manifest_path,
            "threshold": threshold,
            "hysteresis_count": hysteresis_count,
            "infer_rate_hz": infer_rate_hz,
            **aggregated,
        }
    )

    outputs = save_classification_report(report, out_dir=output_dir, prefix="realtime")
    decisions_csv = _write_decisions_csv(output_dir / "realtime_decisions.csv", decisions)

    summary = {
        "run_id": run_id,
        "checkpoint_path": args.checkpoint,
        "manifest_path": manifest_path,
        "model_type": model_cfg.model_type,
        "base_channels": model_cfg.base_channels,
        "use_se": model_cfg.use_se,
        "threshold": threshold,
        "hysteresis_count": hysteresis_count,
        "infer_rate_hz": infer_rate_hz,
        "decision_count": report["decision_count"],
        "hit_rate": report["hit_rate"],
        "false_trigger_rate": report["false_trigger_rate"],
        "latency_p50_ms": report["latency_p50_ms"],
        "latency_p95_ms": report["latency_p95_ms"],
        "test_macro_f1": report["macro_f1"],
        "test_macro_recall": report["macro_recall"],
        "top_confusion_pair": _top_confusion_pair_text(report),
    }
    dump_json(output_dir / "realtime_summary.json", summary)
    append_csv_row(Path(args.run_root) / "realtime_results.csv", BENCHMARK_SUMMARY_FIELDS, summary)

    offline_summary_path = run_dir / "offline_summary.json"
    combined = dict(summary)
    if offline_summary_path.exists():
        combined["offline_summary_path"] = str(offline_summary_path)
        offline_summary = json.loads(offline_summary_path.read_text(encoding="utf-8"))
        combined.update(
            {
                "manifest_path": offline_summary.get("manifest_path", combined["manifest_path"]),
                "checkpoint_path": offline_summary.get("checkpoint_path", combined["checkpoint_path"]),
                "model_type": offline_summary.get("model_type", combined["model_type"]),
                "base_channels": offline_summary.get("base_channels", combined["base_channels"]),
                "use_se": offline_summary.get("use_se", combined["use_se"]),
                "loss_type": offline_summary.get("loss_type", ""),
                "hard_mining_ratio": offline_summary.get("hard_mining_ratio", ""),
                "augment_enabled": offline_summary.get("augment_enabled", ""),
                "augment_factor": offline_summary.get("augment_factor", ""),
                "use_mixup": offline_summary.get("use_mixup", ""),
                "test_accuracy": offline_summary.get("test_accuracy", ""),
                "test_macro_f1": offline_summary.get("test_macro_f1", ""),
                "test_macro_recall": offline_summary.get("test_macro_recall", ""),
                "top_confusion_pair": offline_summary.get("top_confusion_pair", combined["top_confusion_pair"]),
            }
        )
    dump_json(output_dir / "combined_result.json", combined)
    append_csv_row(Path(args.run_root) / "combined_results.csv", COMBINED_SUMMARY_FIELDS, combined)

    dump_json(
        run_dir / "realtime_benchmark_metadata.json",
        {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "outputs": outputs,
            "decisions_csv": str(decisions_csv),
            "mock_mode": mock_mode,
            "expected_input_shape": list(expected_shape),
        },
    )


if __name__ == "__main__":
    main()
