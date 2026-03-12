"""Pretrain an EMG encoder on NinaPro DB5 full coverage and publish foundation artifacts."""

from __future__ import annotations

import argparse
import csv
import logging
import shlex
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
    "learning_rate",
    "weight_decay",
    "loss_type",
    "label_smoothing",
    "ema_enabled",
    "ema_decay",
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
    "use_adaptive_action_thresholds",
    "action_quantile_percent",
    "manifest_use_source_metadata",
    "split_seed",
    "split_train_min_class_samples",
    "split_val_min_class_samples",
    "split_test_min_class_samples",
    "split_has_any_empty_classes",
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
        "--weight_decay",
        type=float,
        default=None,
        help="Override training.weight_decay from config.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=None,
        help="Override training.label_smoothing from config.",
    )
    parser.add_argument(
        "--loss_type",
        default=None,
        choices=["ce", "cross_entropy", "focal", "cb_focal", "class_balanced_focal"],
        help="Override training.loss.type from config.",
    )
    parser.add_argument("--hard_mining_ratio", type=float, default=None)
    parser.add_argument("--ema_enabled", choices=["true", "false"], default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
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
    parser.add_argument("--use_adaptive_action_thresholds", choices=["true", "false"], default=None)
    parser.add_argument("--action_quantile_percent", type=float, default=None)
    parser.add_argument("--max_windows_per_segment", type=int, default=None)
    parser.add_argument("--max_rest_windows_per_segment", type=int, default=None)
    parser.add_argument("--manifest_use_source_metadata", choices=["true", "false"], default=None)
    parser.add_argument(
        "--foundation_dir",
        default="artifacts/foundation/db5_full53",
        help="Fixed foundation artifact directory.",
    )
    parser.add_argument(
        "--smoke_manifest_only",
        action="store_true",
        help="Build dataset/manifest diagnostics only, then exit before model training.",
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


def _build_split_diagnostics(manifest, class_names: list[str]) -> dict:
    class_dist = dict(getattr(manifest, "class_distribution", {}) or {})
    by_split: dict[str, dict] = {}
    has_any_empty = False
    for split_name in ("train", "val", "test"):
        raw_counts = dict(class_dist.get(split_name, {}) or {})
        counts = {name: int(raw_counts.get(name, 0)) for name in class_names}
        empty = [name for name, count in counts.items() if int(count) <= 0]
        min_count = min(counts.values()) if counts else 0
        has_empty = bool(empty)
        has_any_empty = has_any_empty or has_empty
        by_split[split_name] = {
            "class_counts": counts,
            "min_class_count": int(min_count),
            "empty_classes": empty,
            "has_empty_classes": has_empty,
        }
    return {"by_split": by_split, "overall": {"has_any_empty_classes": bool(has_any_empty)}}


def _has_group_leakage(manifest) -> bool:
    train = set(getattr(manifest, "group_keys_train", []) or [])
    val = set(getattr(manifest, "group_keys_val", []) or [])
    test = set(getattr(manifest, "group_keys_test", []) or [])
    return bool((train & val) or (train & test) or (val & test))


def _build_referee_card_content(*, summary: dict, data_dir: str, command: str) -> str:
    return "\n".join(
        [
            "# DB5 预训练复现实验卡",
            "",
            "## 复现命令",
            "```bash",
            str(command).strip(),
            "```",
            "",
            "## 关键信息",
            f"- 数据路径: `{data_dir}`",
            f"- split_seed: `{summary.get('split_seed', '')}`",
            f"- 最优 checkpoint: `{summary.get('checkpoint_path', '')}`",
            f"- 最优验证轮次: `{summary.get('best_val_epoch', '')}`",
            f"- best_val_macro_f1: `{summary.get('best_val_macro_f1', '')}`",
            f"- best_val_acc: `{summary.get('best_val_acc', '')}`",
            f"- top_confusion_pair: `{summary.get('top_confusion_pair', '')}`",
            "",
            "## 说明",
            "- 本复现流程无需个人校准数据。",
            "- 仅依赖公开 DB5 数据与仓库内预训练脚本配置。",
            "",
        ]
    )


def _write_referee_card(run_dir: Path, *, summary: dict, data_dir: str, command: str) -> Path:
    out_path = run_dir / "referee_repro_card.md"
    out_path.write_text(
        _build_referee_card_content(summary=summary, data_dir=data_dir, command=command),
        encoding="utf-8",
    )
    return out_path


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
    repro_command = "python " + " ".join(shlex.quote(item) for item in sys.argv)
    config = load_db5_pretrain_config(args.config)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be > 0")
        config.training.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning_rate must be > 0")
        config.training.learning_rate = float(args.learning_rate)
    if args.weight_decay is not None:
        if args.weight_decay < 0:
            raise ValueError("--weight_decay must be >= 0")
        config.training.weight_decay = float(args.weight_decay)
    if args.label_smoothing is not None:
        if args.label_smoothing < 0 or args.label_smoothing >= 1:
            raise ValueError("--label_smoothing must be in [0,1)")
        config.training.label_smoothing = float(args.label_smoothing)
    if args.loss_type is not None:
        config.training.loss.type = str(args.loss_type)
    if args.hard_mining_ratio is not None:
        if args.hard_mining_ratio < 0:
            raise ValueError("--hard_mining_ratio must be >= 0")
        config.training.sampler.hard_mining_ratio = float(args.hard_mining_ratio)
    ema_enabled_override = _parse_bool_arg(args.ema_enabled)
    if ema_enabled_override is not None:
        config.training.ema.enabled = bool(ema_enabled_override)
    if args.ema_decay is not None:
        if args.ema_decay <= 0 or args.ema_decay >= 1:
            raise ValueError("--ema_decay must be in (0,1)")
        config.training.ema.decay = float(args.ema_decay)
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
    adaptive_threshold_override = _parse_bool_arg(args.use_adaptive_action_thresholds)
    if adaptive_threshold_override is not None:
        config.feature.use_adaptive_action_thresholds = bool(adaptive_threshold_override)
    if args.action_quantile_percent is not None:
        config.feature.action_quantile_percent = float(args.action_quantile_percent)
    if args.max_windows_per_segment is not None:
        if args.max_windows_per_segment <= 0:
            raise ValueError("--max_windows_per_segment must be > 0")
        config.feature.max_windows_per_segment = int(args.max_windows_per_segment)
    if args.max_rest_windows_per_segment is not None:
        if args.max_rest_windows_per_segment <= 0:
            raise ValueError("--max_rest_windows_per_segment must be > 0")
        config.feature.max_rest_windows_per_segment = int(args.max_rest_windows_per_segment)
    manifest_use_source_metadata_override = _parse_bool_arg(args.manifest_use_source_metadata)
    if manifest_use_source_metadata_override is not None:
        config.manifest_use_source_metadata = bool(manifest_use_source_metadata_override)
    if config.feature.lowcut_hz <= 0 or config.feature.highcut_hz <= config.feature.lowcut_hz:
        raise ValueError(
            f"Invalid bandpass settings: lowcut_hz={config.feature.lowcut_hz}, "
            f"highcut_hz={config.feature.highcut_hz}"
        )
    if config.feature.clip_ratio_max < 0 or config.feature.clip_ratio_max > 1:
        raise ValueError(f"clip_ratio_max must be in [0,1], got {config.feature.clip_ratio_max}")
    if config.feature.action_quantile_percent < 0 or config.feature.action_quantile_percent > 100:
        raise ValueError(
            f"action_quantile_percent must be in [0,100], got {config.feature.action_quantile_percent}"
        )
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
            "manifest_use_source_metadata": bool(config.manifest_use_source_metadata),
            "base_channels": int(config.base_channels),
            "classifier_hidden_dim": int(config.classifier_hidden_dim),
            "training": {
                "epochs": int(config.training.epochs),
                "batch_size": int(config.training.batch_size),
                "learning_rate": float(config.training.learning_rate),
                "weight_decay": float(config.training.weight_decay),
                "early_stopping_patience": int(config.training.early_stopping_patience),
                "loss_type": str(config.training.loss.type),
                "label_smoothing": float(config.training.label_smoothing),
                "hard_mining_ratio": float(config.training.sampler.hard_mining_ratio),
                "ema_enabled": bool(config.training.ema.enabled),
                "ema_decay": float(config.training.ema.decay),
            },
            "feature": {
                "source_sampling_rate_hz": config.feature.source_sampling_rate_hz,
                "target_sampling_rate_hz": config.feature.target_sampling_rate_hz,
                "context_window_ms": config.feature.context_window_ms,
                "window_step_ms": config.feature.window_step_ms,
                "use_first_myo_only": config.feature.use_first_myo_only,
                "first_myo_channel_count": config.feature.first_myo_channel_count,
                "max_windows_per_segment": config.feature.max_windows_per_segment,
                "max_rest_windows_per_segment": config.feature.max_rest_windows_per_segment,
                "lowcut_hz": config.feature.lowcut_hz,
                "highcut_hz": config.feature.highcut_hz,
                "energy_min": config.feature.energy_min,
                "static_std_min": config.feature.static_std_min,
                "clip_ratio_max": config.feature.clip_ratio_max,
                "saturation_abs": config.feature.saturation_abs,
                "use_adaptive_action_thresholds": config.feature.use_adaptive_action_thresholds,
                "action_quantile_percent": config.feature.action_quantile_percent,
            },
        },
    )

    loader = DB5PretrainDatasetLoader(data_dir, config)
    samples, labels, source_ids, source_metadata = loader.load_all_with_sources(return_metadata=True)
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
        source_metadata=source_metadata if config.manifest_use_source_metadata else None,
    )
    dump_json(run_dir / "db5_manifest.json", manifest.to_dict())
    split_diagnostics = _build_split_diagnostics(manifest, class_names)
    dump_json(run_dir / "db5_split_diagnostics.json", split_diagnostics)
    if bool(split_diagnostics["overall"]["has_any_empty_classes"]):
        logger.warning("Split diagnostics detected empty classes in at least one split.")
    has_group_leakage = _has_group_leakage(manifest)
    if has_group_leakage:
        raise RuntimeError("Group leakage detected after manifest construction.")
    if args.smoke_manifest_only:
        if bool(split_diagnostics["overall"]["has_any_empty_classes"]):
            raise RuntimeError("Manifest smoke failed: at least one split has empty classes.")
        smoke_summary = {
            "run_id": run_id,
            "mode": "manifest_smoke",
            "num_classes": int(len(class_names)),
            "split_seed": int(config.split_seed),
            "manifest_use_source_metadata": bool(config.manifest_use_source_metadata),
            "group_leakage_detected": bool(has_group_leakage),
            "has_any_empty_classes": bool(split_diagnostics["overall"]["has_any_empty_classes"]),
            "artifacts": {
                "manifest_path": str(run_dir / "db5_manifest.json"),
                "split_diagnostics_path": str(run_dir / "db5_split_diagnostics.json"),
                "window_diagnostics_path": str(run_dir / "db5_window_diagnostics.json"),
            },
            "repro_command": repro_command,
        }
        dump_json(run_dir / "manifest_smoke_summary.json", smoke_summary)
        dump_json(
            run_dir / "run_metadata.json",
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "data_dir": str(data_dir),
                "smoke_manifest_only": True,
                "elapsed_minutes": (time.time() - start) / 60.0,
                "class_names": class_names,
                "window_diagnostics_path": str(run_dir / "db5_window_diagnostics.json"),
                "split_diagnostics_path": str(run_dir / "db5_split_diagnostics.json"),
                "manifest_path": str(run_dir / "db5_manifest.json"),
            },
        )
        logger.info("Manifest smoke completed successfully; skipped training/evaluation by --smoke_manifest_only.")
        return

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
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "loss_type": str(config.training.loss.type),
        "label_smoothing": float(config.training.label_smoothing),
        "ema_enabled": bool(config.training.ema.enabled),
        "ema_decay": float(config.training.ema.decay),
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
        "use_adaptive_action_thresholds": bool(config.feature.use_adaptive_action_thresholds),
        "action_quantile_percent": float(config.feature.action_quantile_percent),
        "manifest_use_source_metadata": bool(config.manifest_use_source_metadata),
        "split_seed": int(config.split_seed),
        "split_train_min_class_samples": int(split_diagnostics["by_split"]["train"]["min_class_count"]),
        "split_val_min_class_samples": int(split_diagnostics["by_split"]["val"]["min_class_count"]),
        "split_test_min_class_samples": int(split_diagnostics["by_split"]["test"]["min_class_count"]),
        "split_has_any_empty_classes": bool(split_diagnostics["overall"]["has_any_empty_classes"]),
        "repro_command": repro_command,
    }
    dump_json(run_dir / "offline_summary.json", summary)
    append_csv_row(Path(args.run_root) / "db5_pretrain_results.csv", SUMMARY_FIELDS, summary)
    referee_card_path = _write_referee_card(
        run_dir,
        summary=summary,
        data_dir=str(data_dir),
        command=repro_command,
    )
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
            "split_diagnostics_path": str(run_dir / "db5_split_diagnostics.json"),
            "referee_repro_card_path": str(referee_card_path),
            "foundation_manifest_path": str(Path(args.foundation_dir) / "foundation_manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
