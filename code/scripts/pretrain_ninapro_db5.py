"""Pretrain an EMG encoder on NinaPro DB5 full coverage and publish foundation artifacts."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ninapro_db5.config import load_db5_pretrain_config
from ninapro_db5.dataset import DB5PretrainDatasetLoader
from ninapro_db5.evaluate import _set_device, evaluate_db5_model, load_db5_model_from_checkpoint
from ninapro_db5.model import build_db5_pretrain_model
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, dump_yaml, ensure_run_dir
from training.data.split_strategy import build_manifest
from training.reporting import save_classification_report
from training.trainer import Trainer


SUMMARY_FIELDS = [
    "run_id",
    "checkpoint_path",
    "foundation_version",
    "num_classes",
    "best_val_epoch",
    "best_val_acc",
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_macro_recall",
    "top_confusion_pair",
    "include_rest_class",
    "use_first_myo_only",
    "first_myo_channel_count",
    "lowcut_hz",
    "highcut_hz",
    "energy_min",
    "static_std_min",
    "clip_ratio_max",
    "saturation_abs",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain EMG encoder on NinaPro DB5 full coverage")
    parser.add_argument("--config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--ms_mode",
        default="graph",
        choices=["graph", "pynative"],
        help="MindSpore execution mode. Use pynative as fallback on unstable Ascend graph runs.",
    )
    parser.add_argument(
        "--launch_blocking",
        action="store_true",
        help="Enable MindSpore launch_blocking for operator-level error localization.",
    )
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size from config")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override training.learning_rate from config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Override training.early_stopping_patience from config.",
    )
    parser.add_argument("--base_channels", type=int, default=None, help="Override model base channels.")
    parser.add_argument("--classifier_hidden_dim", type=int, default=None, help="Override classifier hidden dim.")
    parser.add_argument("--include_rest_class", choices=["true", "false"], default=None)
    parser.add_argument("--use_first_myo_only", choices=["true", "false"], default=None)
    parser.add_argument("--first_myo_channel_count", type=int, default=None)
    parser.add_argument("--lowcut_hz", type=float, default=None)
    parser.add_argument("--highcut_hz", type=float, default=None)
    parser.add_argument("--energy_min", type=float, default=None)
    parser.add_argument("--static_std_min", type=float, default=None)
    parser.add_argument("--clip_ratio_max", type=float, default=None)
    parser.add_argument("--saturation_abs", type=float, default=None)
    parser.add_argument(
        "--foundation_dir",
        default="artifacts/foundation/db5_full53",
        help="Fixed foundation artifact directory.",
    )
    return parser


def _parse_bool_arg(raw: str | None) -> bool | None:
    if raw is None:
        return None
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


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


def _best_validation_from_history(history: dict) -> tuple[int, float, float]:
    val_f1 = [float(v) for v in history.get("val_macro_f1", [])]
    val_acc = [float(v) for v in history.get("val_acc", [])]
    epochs = [int(v) for v in history.get("epoch", [])]
    if not (val_f1 and val_acc and epochs):
        return -1, 0.0, 0.0
    best_idx = max(range(len(epochs)), key=lambda i: (val_f1[i], val_acc[i]))
    return int(epochs[best_idx]), float(val_acc[best_idx]), float(val_f1[best_idx])


def _validate_label_alignment(labels: np.ndarray, class_names: list[str]) -> None:
    unique_labels = sorted({int(value) for value in np.asarray(labels, dtype=np.int32).tolist()})
    expected_labels = list(range(len(class_names)))
    if unique_labels != expected_labels:
        raise RuntimeError(
            "Class/label alignment mismatch: "
            f"unique_labels={unique_labels}, expected={expected_labels}, class_names={class_names}"
        )


def _maybe_enable_launch_blocking(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from mindspore import runtime

        runtime.launch_blocking()
        logging.getLogger("ninapro_db5.pretrain").info("MindSpore launch_blocking enabled.")
    except Exception as exc:  # pragma: no cover
        logging.getLogger("ninapro_db5.pretrain").warning("Failed to enable launch_blocking: %s", exc)


def _publish_foundation(
    *,
    foundation_dir: Path,
    run_id: str,
    run_dir: Path,
    summary: dict,
    class_names: list[str],
    config_version: str,
    data_dir: str,
) -> dict:
    foundation_dir.mkdir(parents=True, exist_ok=True)
    foundation_ckpt = foundation_dir / "checkpoints" / "db5_full53_foundation.ckpt"
    foundation_ckpt.parent.mkdir(parents=True, exist_ok=True)
    source_ckpt = Path(summary["checkpoint_path"])
    if foundation_ckpt.exists():
        foundation_ckpt.chmod(0o644)
        foundation_ckpt.unlink()
    shutil.copyfile(source_ckpt, foundation_ckpt)
    foundation_ckpt.chmod(0o644)

    manifest = {
        "foundation_version": config_version,
        "coverage": "db5_full53",
        "num_classes": int(summary["num_classes"]),
        "class_names": list(class_names),
        "checkpoint_path": str(foundation_ckpt),
        "source_run_id": run_id,
        "source_run_dir": str(run_dir),
        "source_data_dir": str(data_dir),
        "source_summary_path": str(run_dir / "offline_summary.json"),
        "created_at_unix": int(time.time()),
    }
    dump_json(foundation_dir / "foundation_manifest.json", manifest)
    dump_json(foundation_dir / "foundation_summary.json", summary)
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    start = time.time()
    config = load_db5_pretrain_config(args.config)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be > 0")
        config.training.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning_rate must be > 0")
        config.training.learning_rate = float(args.learning_rate)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be > 0")
        config.training.epochs = int(args.epochs)
    if args.early_stopping_patience is not None:
        if args.early_stopping_patience <= 0:
            raise ValueError("--early_stopping_patience must be > 0")
        config.training.early_stopping_patience = int(args.early_stopping_patience)
    if args.base_channels is not None:
        if args.base_channels <= 0:
            raise ValueError("--base_channels must be > 0")
        config.base_channels = int(args.base_channels)
    if args.classifier_hidden_dim is not None:
        if args.classifier_hidden_dim <= 0:
            raise ValueError("--classifier_hidden_dim must be > 0")
        config.classifier_hidden_dim = int(args.classifier_hidden_dim)
    include_rest_override = _parse_bool_arg(args.include_rest_class)
    if include_rest_override is not None:
        config.include_rest_class = bool(include_rest_override)
    use_first_myo_only_override = _parse_bool_arg(args.use_first_myo_only)
    if use_first_myo_only_override is not None:
        config.feature.use_first_myo_only = bool(use_first_myo_only_override)
    if args.first_myo_channel_count is not None:
        if args.first_myo_channel_count <= 0:
            raise ValueError("--first_myo_channel_count must be > 0")
        config.feature.first_myo_channel_count = int(args.first_myo_channel_count)
    if args.lowcut_hz is not None:
        config.feature.lowcut_hz = float(args.lowcut_hz)
    if args.highcut_hz is not None:
        config.feature.highcut_hz = float(args.highcut_hz)
    if args.energy_min is not None:
        config.feature.energy_min = float(args.energy_min)
    if args.static_std_min is not None:
        config.feature.static_std_min = float(args.static_std_min)
    if args.clip_ratio_max is not None:
        config.feature.clip_ratio_max = float(args.clip_ratio_max)
    if args.saturation_abs is not None:
        config.feature.saturation_abs = float(args.saturation_abs)
    if config.feature.lowcut_hz <= 0 or config.feature.highcut_hz <= config.feature.lowcut_hz:
        raise ValueError(
            f"Invalid bandpass settings: lowcut_hz={config.feature.lowcut_hz}, "
            f"highcut_hz={config.feature.highcut_hz}"
        )
    if config.feature.clip_ratio_max < 0 or config.feature.clip_ratio_max > 1:
        raise ValueError(f"clip_ratio_max must be in [0,1], got {config.feature.clip_ratio_max}")
    data_dir = args.data_dir or config.data_dir
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="db5_pretrain_full53")
    logger = logging.getLogger("ninapro_db5.pretrain")
    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", run_dir)
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)
    _maybe_enable_launch_blocking(args.launch_blocking)
    _set_device(mode=args.ms_mode, target=args.device_target, device_id=args.device_id)
    logger.info(
        "Training device configured: mode=%s target=%s device_id=%d",
        args.ms_mode,
        args.device_target,
        args.device_id,
    )

    dump_yaml(
        run_dir / "config_snapshots" / "effective_config.yaml",
        {
            "data_dir": data_dir,
            "coverage": "full53",
            "foundation_version": config.foundation_version,
            "include_rest_class": bool(config.include_rest_class),
            "use_restimulus": bool(config.use_restimulus),
            "base_channels": int(config.base_channels),
            "classifier_hidden_dim": int(config.classifier_hidden_dim),
            "training": {
                "epochs": int(config.training.epochs),
                "batch_size": int(config.training.batch_size),
                "learning_rate": float(config.training.learning_rate),
                "early_stopping_patience": int(config.training.early_stopping_patience),
            },
            "feature": {
                "source_sampling_rate_hz": config.feature.source_sampling_rate_hz,
                "target_sampling_rate_hz": config.feature.target_sampling_rate_hz,
                "context_window_ms": config.feature.context_window_ms,
                "window_step_ms": config.feature.window_step_ms,
                "use_first_myo_only": config.feature.use_first_myo_only,
                "first_myo_channel_count": config.feature.first_myo_channel_count,
                "lowcut_hz": config.feature.lowcut_hz,
                "highcut_hz": config.feature.highcut_hz,
                "energy_min": config.feature.energy_min,
                "static_std_min": config.feature.static_std_min,
                "clip_ratio_max": config.feature.clip_ratio_max,
                "saturation_abs": config.feature.saturation_abs,
            },
        },
    )

    loader = DB5PretrainDatasetLoader(data_dir, config)
    samples, labels, source_ids = loader.load_all_with_sources()
    class_names = loader.get_class_names()
    window_diagnostics = loader.get_window_diagnostics()
    dump_json(run_dir / "db5_window_diagnostics.json", window_diagnostics)
    _validate_label_alignment(labels, class_names)
    logger.info("Loaded DB5 samples: %s labels=%d classes=%d", tuple(samples.shape), labels.shape[0], len(class_names))
    finite_mask = np.isfinite(samples)
    if not bool(np.all(finite_mask)):
        bad_count = int(np.size(samples) - np.count_nonzero(finite_mask))
        raise RuntimeError(f"DB5 samples contain non-finite values (NaN/Inf): count={bad_count}")
    logger.info(
        "Sample stats: min=%.5f max=%.5f mean=%.5f std=%.5f",
        float(np.min(samples)),
        float(np.max(samples)),
        float(np.mean(samples)),
        float(np.std(samples)),
    )

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
    best_val_epoch, best_val_acc, best_val_f1 = _best_validation_from_history(history)

    _set_device(mode=args.ms_mode, target=args.device_target, device_id=args.device_id)
    best_model = load_db5_model_from_checkpoint(trainer.checkpoint_path, config, num_classes=len(class_names))
    report = evaluate_db5_model(best_model, test_x, test_y, class_names)
    report.update({"run_id": run_id, "checkpoint_path": str(trainer.checkpoint_path)})
    outputs = save_classification_report(report, out_dir=run_dir / "evaluation", prefix="test")

    summary = {
        "run_id": run_id,
        "checkpoint_path": str(trainer.checkpoint_path),
        "foundation_version": config.foundation_version,
        "num_classes": len(class_names),
        "best_val_epoch": int(best_val_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": float(best_val_f1),
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro_f1"],
        "test_macro_recall": report["macro_recall"],
        "top_confusion_pair": _top_confusion_pair_text(report),
        "include_rest_class": bool(config.include_rest_class),
        "use_first_myo_only": bool(config.feature.use_first_myo_only),
        "first_myo_channel_count": int(config.feature.first_myo_channel_count),
        "lowcut_hz": float(config.feature.lowcut_hz),
        "highcut_hz": float(config.feature.highcut_hz),
        "energy_min": float(config.feature.energy_min),
        "static_std_min": float(config.feature.static_std_min),
        "clip_ratio_max": float(config.feature.clip_ratio_max),
        "saturation_abs": float(config.feature.saturation_abs),
    }
    dump_json(run_dir / "offline_summary.json", summary)
    append_csv_row(Path(args.run_root) / "db5_pretrain_results.csv", SUMMARY_FIELDS, summary)
    foundation_manifest = _publish_foundation(
        foundation_dir=Path(args.foundation_dir),
        run_id=run_id,
        run_dir=run_dir,
        summary=summary,
        class_names=class_names,
        config_version=config.foundation_version,
        data_dir=data_dir,
    )
    dump_json(run_dir / "foundation_manifest.json", foundation_manifest)
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
            "window_diagnostics_path": str(run_dir / "db5_window_diagnostics.json"),
            "foundation_manifest_path": str(Path(args.foundation_dir) / "foundation_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
