"""Pretrain the EMG encoder on NinaPro DB5 and save a transferable checkpoint."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ninapro_db5.config import load_db5_pretrain_config
from ninapro_db5.dataset import DB5PretrainDatasetLoader
from ninapro_db5.evaluate import evaluate_db5_model, load_db5_model_from_checkpoint
from ninapro_db5.model import build_db5_pretrain_model
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, dump_yaml, ensure_run_dir
from training.data.split_strategy import build_manifest
from training.reporting import save_classification_report
from training.trainer import Trainer


SUMMARY_FIELDS = [
    "run_id",
    "checkpoint_path",
    "num_classes",
    "test_accuracy",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain EMG encoder on NinaPro DB5")
    parser.add_argument("--config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    return parser


def _save_history(history: dict, out_csv: str | Path) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(rows)


def _top_confusion_pair_text(report: dict) -> str:
    pairs = report.get("top_confusion_pairs") or []
    if not pairs:
        return ""
    pair = pairs[0]
    return f"{pair['pair'][0]}<->{pair['pair'][1]}:{pair['count']}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    start = time.time()
    config = load_db5_pretrain_config(args.config)
    data_dir = args.data_dir or config.data_dir
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="db5_pretrain")
    logger = logging.getLogger("ninapro_db5.pretrain")
    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", run_dir)
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)

    dump_yaml(
        run_dir / "config_snapshots" / "effective_config.yaml",
        {
            "data_dir": data_dir,
            "feature": {
                "source_sampling_rate_hz": config.feature.source_sampling_rate_hz,
                "target_sampling_rate_hz": config.feature.target_sampling_rate_hz,
                "context_window_ms": config.feature.context_window_ms,
                "window_step_ms": config.feature.window_step_ms,
                "use_first_myo_only": config.feature.use_first_myo_only,
            },
        },
    )

    loader = DB5PretrainDatasetLoader(data_dir, config)
    samples, labels, source_ids = loader.load_all_with_sources()
    class_names = loader.get_class_names()
    logger.info("Loaded DB5 samples: %s labels=%d classes=%d", tuple(samples.shape), labels.shape[0], len(class_names))

    manifest = build_manifest(
        labels,
        source_ids,
        seed=config.split_seed,
        split_mode="grouped_file",
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_classes=len(class_names),
        class_names=class_names,
        manifest_strategy="v2",
        source_metadata=None,
    )
    dump_json(run_dir / "db5_manifest.json", manifest.to_dict())

    train_idx = np.asarray(manifest.train_indices, dtype=np.int32)
    val_idx = np.asarray(manifest.val_indices, dtype=np.int32)
    test_idx = np.asarray(manifest.test_indices, dtype=np.int32)
    train_x, train_y = samples[train_idx], labels[train_idx]
    val_x, val_y = samples[val_idx], labels[val_idx]
    test_x, test_y = samples[test_idx], labels[test_idx]
    logger.info("Split sizes => train=%d val=%d test=%d", len(train_y), len(val_y), len(test_y))

    config.training.num_workers = 0
    model = build_db5_pretrain_model(config, num_classes=len(class_names))
    trainer = Trainer(model, config.training, class_names, output_dir=str(run_dir))
    trainer.checkpoint_path = run_dir / "checkpoints" / "db5_pretrain_best.ckpt"
    history = trainer.train(train_x, train_y, val_x, val_y)
    history_path = run_dir / "training_history.csv"
    _save_history(history, history_path)

    from ninapro_db5.evaluate import _set_device

    _set_device(target=args.device_target, device_id=args.device_id)
    best_model = load_db5_model_from_checkpoint(trainer.checkpoint_path, config, num_classes=len(class_names))
    report = evaluate_db5_model(best_model, test_x, test_y, class_names)
    report.update(
        {
            "run_id": run_id,
            "checkpoint_path": str(trainer.checkpoint_path),
        }
    )
    outputs = save_classification_report(report, out_dir=run_dir / "evaluation", prefix="test")

    summary = {
        "run_id": run_id,
        "checkpoint_path": str(trainer.checkpoint_path),
        "num_classes": len(class_names),
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro_f1"],
        "test_macro_recall": report["macro_recall"],
        "top_confusion_pair": _top_confusion_pair_text(report),
    }
    dump_json(run_dir / "offline_summary.json", summary)
    append_csv_row(Path(args.run_root) / "db5_pretrain_results.csv", SUMMARY_FIELDS, summary)
    dump_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "data_dir": str(data_dir),
            "history_path": str(history_path),
            "evaluation_outputs": outputs,
            "elapsed_minutes": (time.time() - start) / 60.0,
            "class_names": class_names,
        },
    )


if __name__ == "__main__":
    main()
