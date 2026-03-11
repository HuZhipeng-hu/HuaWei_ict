"""Evaluate event-onset checkpoint with a fixed split manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.evaluate import load_and_evaluate_event
from shared.label_modes import get_label_mode_spec
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, ensure_run_dir
from training.data.split_strategy import load_manifest
from training.reporting import save_classification_report

EVAL_SUMMARY_FIELDS = [
    "run_id",
    "checkpoint_path",
    "manifest_path",
    "model_type",
    "base_channels",
    "use_se",
    "test_accuracy",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
]


def _top_confusion_pair_text(report: dict) -> str:
    pairs = report.get("top_confusion_pairs") or []
    if not pairs:
        return ""
    pair = pairs[0]
    return f"{pair['pair'][0]}<->{pair['pair'][1]}:{pair['count']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event-onset checkpoint on test split")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--output_dir", default="evaluation_recheck")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--eval_protocol", default="same_user_same_day_v1")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    logger = logging.getLogger("event_onset.evaluate_ckpt")
    logger.info("Loading event training config: %s", args.config)

    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="event_eval")
    output_dir = run_dir / args.output_dir
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)
    copy_config_snapshot(args.split_manifest, run_dir / "manifests" / Path(args.split_manifest).name)

    model_cfg, data_cfg, _, _ = load_event_training_config(args.config)
    label_spec = get_label_mode_spec(data_cfg.label_mode)
    loader = EventClipDatasetLoader(
        args.data_dir,
        data_cfg,
        recordings_manifest_path=args.recordings_manifest or data_cfg.recordings_manifest_path,
    )
    emg_samples, imu_samples, labels, _, _ = loader.load_all_with_sources(return_metadata=True)

    manifest = load_manifest(args.split_manifest)
    if manifest.num_samples != int(labels.shape[0]):
        raise ValueError(
            f"Manifest sample count mismatch: manifest={manifest.num_samples}, loaded={labels.shape[0]}. "
            "Regenerate manifest with current event-onset settings."
        )
    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    report = load_and_evaluate_event(
        ckpt_path=args.checkpoint,
        emg_samples=emg_samples[test_idx],
        imu_samples=imu_samples[test_idx],
        labels=labels[test_idx],
        class_names=label_spec.class_names,
        model_config=model_cfg,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    report.update(
        {
            "eval_protocol": args.eval_protocol,
            "manifest_path": args.split_manifest,
            "checkpoint_path": args.checkpoint,
            "run_id": run_id,
        }
    )
    outputs = save_classification_report(report, out_dir=output_dir, prefix="test")
    summary = {
        "run_id": run_id,
        "checkpoint_path": args.checkpoint,
        "manifest_path": args.split_manifest,
        "model_type": model_cfg.model_type,
        "base_channels": model_cfg.base_channels,
        "use_se": model_cfg.use_se,
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro_f1"],
        "test_macro_recall": report["macro_recall"],
        "top_confusion_pair": _top_confusion_pair_text(report),
    }
    dump_json(output_dir / "evaluation_summary.json", summary)
    append_csv_row(Path(args.run_root) / "evaluation_results.csv", EVAL_SUMMARY_FIELDS, summary)
    dump_json(
        run_dir / "evaluation_recheck_metadata.json",
        {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "recordings_manifest_path": str(args.recordings_manifest or data_cfg.recordings_manifest_path),
            "outputs": outputs,
        },
    )
    logger.info("Event checkpoint evaluation finished. Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
