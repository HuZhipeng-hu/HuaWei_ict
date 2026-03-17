"""Orchestrate model-line sprint: baseline -> screening -> longrun -> threshold tuning."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from shared.config import load_config
from shared.event_labels import public_event_labels
from shared.label_modes import get_label_mode_spec
from training.data.split_strategy import build_manifest, save_manifest


DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW"
DEFAULT_SOURCE_RECORDINGS_MANIFEST = "recordings_manifest.csv"
DEFAULT_PREPARE_SESSION_ID = "s2"
DEFAULT_PREPARE_TARGET_PER_CLASS = 12
DEFAULT_PREPARE_RELAX_TARGET_COUNT = 24
DEFAULT_PREPARE_ACTION_MIN_WINDOWS = 2
DEFAULT_PREPARE_RELAX_MIN_WINDOWS = 1


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in cmd)


def _run_checked(stage: str, cmd: list[str]) -> str:
    cmd_text = _format_cmd(cmd)
    print(f"[MODEL90] {stage} -> {cmd_text}", flush=True)
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {cmd_text}")
    return cmd_text


def _parse_tokens(raw: str, *, name: str) -> list[str]:
    tokens = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not tokens:
        raise ValueError(f"{name} must contain at least one value")
    return tokens


def _parse_int_tokens(raw: str, *, name: str) -> list[int]:
    return [int(token) for token in _parse_tokens(raw, name=name)]


def _parse_float_tokens(raw: str, *, name: str) -> list[float]:
    return [float(token) for token in _parse_tokens(raw, name=name)]


def _parse_bool_arg(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def _safe_float_token(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be object: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _metric_or(payload: dict, key: str, default: float = 0.0) -> float:
    value = payload.get(key, None)
    if value is None:
        value = default
    elif isinstance(value, str) and not value.strip():
        value = default
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_path(raw: str | Path, *, prefer_data_dir: str | Path | None = None) -> Path:
    path = Path(str(raw).strip())
    candidates = [path]
    if not path.is_absolute():
        if prefer_data_dir is not None:
            candidates.insert(0, Path(prefer_data_dir) / path)
        candidates.insert(0, CODE_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _parse_target_action_keys(raw: str) -> list[str]:
    keys = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not keys:
        raise ValueError("target_db5_keys must contain at least one action key")
    return keys


def _prepare_target_states(args: argparse.Namespace) -> list[str]:
    return ["RELAX", *_parse_target_action_keys(args.target_db5_keys)]


def _training_manifest_for_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "_prepared_recordings_manifest", "") or args.recordings_manifest)


def _prepare_output_manifest_path(args: argparse.Namespace) -> Path:
    raw = str(getattr(args, "prepare_output_manifest", "") or "").strip()
    if raw:
        return _resolve_path(raw, prefer_data_dir=args.data_dir)
    return _resolve_path(f"{args.run_prefix}_demo3_train_manifest.csv", prefer_data_dir=args.data_dir)


def _model90_split_manifest_path(args: argparse.Namespace, seed: int) -> Path:
    explicit = str(getattr(args, "screen_split_manifest", "") or "").strip()
    if explicit and int(seed) == int(args.screen_split_seed):
        return _resolve_path(explicit)
    return _resolve_path(f"artifacts/splits/{args.run_prefix}_demo3_seed{int(seed)}_v2.json")


def _prepare_split_seeds(args: argparse.Namespace) -> list[int]:
    seeds = {int(args.screen_split_seed)}
    for seed in _parse_int_tokens(args.longrun_seeds, name="--longrun_seeds"):
        seeds.add(int(seed))
    if int(args.neighbor_split_seed) >= 0:
        seeds.add(int(args.neighbor_split_seed))
    return sorted(seeds)


def _build_split_manifest_from_recordings(
    *,
    training_config: str,
    data_dir: str,
    recordings_manifest: str | Path,
    target_db5_keys: str,
    split_seed: int,
    output_path: str | Path,
) -> dict:
    _, data_cfg, train_cfg, _ = load_event_training_config(training_config)
    data_cfg.target_db5_keys = _parse_target_action_keys(target_db5_keys)
    label_spec = get_label_mode_spec(data_cfg.label_mode, data_cfg.target_db5_keys)
    loader = EventClipDatasetLoader(data_dir, data_cfg, recordings_manifest_path=recordings_manifest)
    _, _, labels, source_ids, source_meta = loader.load_all_with_sources(return_metadata=True)
    manifest = build_manifest(
        labels,
        source_ids,
        seed=int(split_seed),
        split_mode=data_cfg.split_mode,
        val_ratio=train_cfg.val_ratio,
        test_ratio=train_cfg.test_ratio,
        num_classes=len(label_spec.class_names),
        class_names=label_spec.class_names,
        manifest_strategy="v2",
        source_metadata=source_meta,
    )
    saved = save_manifest(manifest, str(output_path))
    return {
        "seed": int(split_seed),
        "path": str(saved),
        "num_samples": int(manifest.num_samples),
        "train_indices": int(len(manifest.train_indices)),
        "val_indices": int(len(manifest.val_indices)),
        "test_indices": int(len(manifest.test_indices)),
        "class_distribution": dict(manifest.class_distribution),
    }


def _reuse_trial_outputs_if_compatible(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    recordings_manifest_path: str,
    split_manifest_path: str,
    loss_type: str,
    base_channels: int,
    freeze_emg_epochs: int,
    encoder_lr_ratio: float,
    pretrained_mode: str,
) -> bool:
    offline_summary_path = run_dir / "offline_summary.json"
    test_metrics_path = run_dir / "evaluation" / "test_metrics.json"
    control_eval_path = run_dir / "evaluation" / "control_eval_summary.json"
    overrides_path = run_dir / "config_snapshots" / "effective_overrides.yaml"
    run_metadata_path = run_dir / "run_metadata.json"

    required = [offline_summary_path, test_metrics_path, overrides_path, run_metadata_path]
    if not all(path.exists() for path in required):
        return False
    if not bool(args.skip_control_eval) and not control_eval_path.exists():
        return False

    try:
        offline = _load_json(offline_summary_path)
        run_metadata = _load_json(run_metadata_path)
        overrides = load_config(overrides_path)
    except Exception:
        return False

    expected_manifest = _resolve_path(recordings_manifest_path, prefer_data_dir=args.data_dir)
    expected_split = _resolve_path(split_manifest_path)
    actual_manifest = _resolve_path(str(run_metadata.get("recordings_manifest_path", "")), prefer_data_dir=args.data_dir)
    actual_split = _resolve_path(str(offline.get("manifest_path", "")))
    if actual_manifest != expected_manifest:
        return False
    if actual_split != expected_split:
        return False

    model_section = dict(overrides.get("model", {}) or {})
    train_section = dict(overrides.get("training", {}) or {})
    device_section = dict(overrides.get("device", {}) or {})
    if str(train_section.get("loss_type", "")).strip() != str(loss_type):
        return False
    if int(model_section.get("base_channels", -1)) != int(base_channels):
        return False
    if int(train_section.get("freeze_emg_epochs", -1)) != int(freeze_emg_epochs):
        return False
    if abs(float(train_section.get("encoder_lr_ratio", -1.0)) - float(encoder_lr_ratio)) > 1e-9:
        return False
    if str(device_section.get("device_target", "")).strip() != str(args.device_target):
        return False
    if int(device_section.get("device_id", -1)) != int(args.device_id):
        return False

    expected_pretrained = str(args.pretrained_emg_checkpoint or "").strip() if str(pretrained_mode).strip().lower() == "on" else ""
    actual_pretrained = str(model_section.get("pretrained_emg_checkpoint", "") or "").strip()
    if expected_pretrained:
        if not actual_pretrained:
            return False
        if _resolve_path(actual_pretrained) != _resolve_path(expected_pretrained):
            return False
    elif actual_pretrained:
        return False
    return True


def _stage_prepare(args: argparse.Namespace) -> dict:
    source_manifest = _resolve_path(args.recordings_manifest, prefer_data_dir=args.data_dir)
    prepare_dir = _resolve_path(f"artifacts/runs/{args.run_prefix}_prepare")
    audit_run_id = f"{args.run_prefix}_prepare_audit"
    prepared_manifest_path = _prepare_output_manifest_path(args)
    drop_report_path = prepare_dir / "prepare_drop_report.json"

    audit_cmd = [
        sys.executable,
        "scripts/audit_event_collection.py",
        "--config",
        str(args.training_config),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(source_manifest),
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(audit_run_id),
    ]
    _run_checked("prepare:audit_collection", audit_cmd)

    audit_dir = _resolve_path(Path(args.run_root) / audit_run_id)
    audit_summary_path = audit_dir / "collection_audit_summary.json"
    audit_details_path = audit_dir / "collection_audit_details.json"
    audit_summary = _load_json(audit_summary_path)
    audit_details = _load_json(audit_details_path)

    build_cmd = [
        sys.executable,
        "scripts/build_s2_train_manifest.py",
        "--recordings_manifest",
        str(source_manifest),
        "--audit_details_json",
        str(audit_details_path),
        "--session_id",
        str(args.prepare_session_id),
        "--target_states",
        ",".join(_prepare_target_states(args)),
        "--target_per_class",
        str(int(args.prepare_target_per_class)),
        "--relax_target_count",
        str(int(args.prepare_relax_target_count)),
        "--action_min_selected_windows",
        str(int(args.prepare_action_min_selected_windows)),
        "--relax_min_selected_windows",
        str(int(args.prepare_relax_min_selected_windows)),
        "--relax_allow_retake_quality",
        str(bool(args.prepare_relax_allow_retake_quality)).lower(),
        "--output_manifest",
        str(prepared_manifest_path),
        "--output_drop_report",
        str(drop_report_path),
    ]
    _run_checked("prepare:build_manifest", build_cmd)

    drop_report = _load_json(drop_report_path)
    split_summaries: list[dict] = []
    for seed in _prepare_split_seeds(args):
        split_summaries.append(
            _build_split_manifest_from_recordings(
                training_config=str(args.training_config),
                data_dir=str(args.data_dir),
                recordings_manifest=str(prepared_manifest_path),
                target_db5_keys=str(args.target_db5_keys),
                split_seed=int(seed),
                output_path=_model90_split_manifest_path(args, int(seed)),
            )
        )

    summary = {
        "status": "ok",
        "stage": "prepare",
        "run_prefix": str(args.run_prefix),
        "source_recordings_manifest": str(source_manifest),
        "prepared_recordings_manifest": str(prepared_manifest_path),
        "audit_summary_json": str(audit_summary_path),
        "audit_details_json": str(audit_details_path),
        "drop_report_json": str(drop_report_path),
        "target_states": public_event_labels(_prepare_target_states(args)),
        "session_id": str(args.prepare_session_id),
        "target_per_class": int(args.prepare_target_per_class),
        "relax_target_count": int(args.prepare_relax_target_count),
        "coverage": dict(drop_report.get("coverage", {})),
        "counts": dict(drop_report.get("counts", {})),
        "kept_by_class": dict(drop_report.get("kept_by_class_after_cap", {})),
        "dropped_by_class": dict(drop_report.get("by_target_dropped", {})),
        "zero_selected_window_clips": int(audit_summary.get("zero_selected_window_clips", 0) or 0),
        "split_manifests": split_summaries,
    }
    output_path = _resolve_path(f"artifacts/runs/{args.run_prefix}_prepare_summary.json")
    _write_json(output_path, summary)
    setattr(args, "_prepared_recordings_manifest", str(prepared_manifest_path))
    setattr(args, "_prepared_summary_path", str(output_path))
    return summary


def _strict_online_pass(row: dict, *, target_command_success_rate: float, max_false_trigger_rate: float, max_false_release_rate: float) -> bool:
    return bool(
        float(row.get("command_success_rate", 0.0)) >= float(target_command_success_rate)
        and float(row.get("false_trigger_rate", 1.0)) <= float(max_false_trigger_rate)
        and float(row.get("false_release_rate", 1.0)) <= float(max_false_release_rate)
    )


def _rank_row(row: dict) -> tuple[int, float, float, float, float, float]:
    return (
        int(bool(row.get("pass_strict_online_gate", False))),
        float(row.get("event_action_accuracy", 0.0)),
        float(row.get("event_action_macro_f1", 0.0)),
        float(row.get("command_success_rate", 0.0)),
        -float(row.get("false_trigger_rate", 1.0)),
        -float(row.get("false_release_rate", 1.0)),
    )


def _build_finetune_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    recordings_manifest_path: str,
    split_seed: int,
    split_manifest_path: str,
    loss_type: str | None,
    base_channels: int | None,
    freeze_emg_epochs: int | None,
    encoder_lr_ratio: float | None,
    pretrained_mode: str,
) -> list[str]:
    split_manifest_raw = str(split_manifest_path).strip()
    split_manifest_abs = Path(split_manifest_raw)
    if not split_manifest_abs.is_absolute():
        split_manifest_abs = (CODE_ROOT / split_manifest_abs).resolve()

    manifest_flags: list[str] = []
    if split_manifest_raw:
        if split_manifest_abs.exists():
            manifest_flags.extend(
                [
                    "--split_manifest_in",
                    split_manifest_raw,
                    "--split_manifest_out",
                    split_manifest_raw,
                ]
            )
        else:
            print(
                f"[MODEL90] split_manifest missing, will auto-build: {split_manifest_raw}",
                flush=True,
            )
            manifest_flags.extend(["--split_manifest_out", split_manifest_raw])

    cmd = [
        sys.executable,
        "scripts/finetune_event_onset.py",
        "--config",
        str(args.training_config),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(recordings_manifest_path),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
        "--split_seed",
        str(int(split_seed)),
        "--manifest_strategy",
        "v2",
        "--budget_per_class",
        str(int(args.budget_per_class)),
        "--budget_seed",
        str(int(args.budget_seed)),
    ]
    cmd.extend(manifest_flags)
    if loss_type:
        cmd.extend(["--loss_type", str(loss_type)])
    if base_channels is not None:
        cmd.extend(["--base_channels", str(int(base_channels))])
    if freeze_emg_epochs is not None:
        cmd.extend(["--freeze_emg_epochs", str(int(freeze_emg_epochs))])
    if encoder_lr_ratio is not None:
        cmd.extend(["--encoder_lr_ratio", str(float(encoder_lr_ratio))])

    if pretrained_mode == "on":
        ckpt = str(args.pretrained_emg_checkpoint or "").strip()
        if not ckpt:
            raise ValueError("pretrained_mode=on requires --pretrained_emg_checkpoint")
        cmd.extend(["--pretrained_emg_checkpoint", ckpt])
    return cmd


def _build_control_eval_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    recordings_manifest_path: str,
    split_manifest_path: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_event_demo_control.py",
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
        "--training_config",
        str(args.training_config),
        "--runtime_config",
        str(args.runtime_config),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(recordings_manifest_path),
        "--split_manifest",
        str(split_manifest_path),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--backend",
        str(args.control_backend),
        "--device_target",
        str(args.device_target),
    ]


def _collect_metrics_row(
    args: argparse.Namespace,
    *,
    stage: str,
    run_id: str,
    split_seed: int,
    split_manifest_path: str,
    loss_type: str,
    base_channels: int,
    freeze_emg_epochs: int,
    encoder_lr_ratio: float,
    pretrained_mode: str,
) -> dict:
    run_root = Path(args.run_root)
    run_dir = run_root / run_id
    offline = _load_json(run_dir / "offline_summary.json")
    test_metrics = _load_json(run_dir / "evaluation" / "test_metrics.json")
    control_path = run_dir / "evaluation" / "control_eval_summary.json"
    control = _load_json(control_path) if control_path.exists() else {}
    row = {
        "stage": str(stage),
        "run_id": str(run_id),
        "recordings_manifest_path": _training_manifest_for_args(args),
        "split_seed": int(split_seed),
        "split_manifest_path": str(split_manifest_path),
        "loss_type": str(loss_type),
        "base_channels": int(base_channels),
        "freeze_emg_epochs": int(freeze_emg_epochs),
        "encoder_lr_ratio": float(encoder_lr_ratio),
        "pretrained_mode": str(pretrained_mode),
        "checkpoint_path": str(offline.get("checkpoint_path", "")),
        "event_action_accuracy": _metric_or(test_metrics, "event_action_accuracy"),
        "event_action_macro_f1": _metric_or(test_metrics, "event_action_macro_f1"),
        "test_accuracy": _metric_or(test_metrics, "accuracy"),
        "test_macro_f1": _metric_or(test_metrics, "macro_f1"),
        "command_success_rate": _metric_or(
            control,
            "command_success_rate",
            default=_metric_or(offline, "test_command_success_rate", default=_metric_or(offline, "command_success_rate")),
        ),
        "false_trigger_rate": _metric_or(
            control,
            "false_trigger_rate",
            default=_metric_or(offline, "test_false_trigger_rate", default=_metric_or(offline, "false_trigger_rate", default=1.0)),
        ),
        "false_release_rate": _metric_or(
            control,
            "false_release_rate",
            default=_metric_or(offline, "test_false_release_rate", default=_metric_or(offline, "false_release_rate", default=1.0)),
        ),
        "control_eval_present": bool(control_path.exists()),
        "summary_path": str(run_dir / "offline_summary.json"),
        "test_metrics_path": str(run_dir / "evaluation" / "test_metrics.json"),
        "control_eval_path": str(control_path),
    }
    row["pass_strict_online_gate"] = _strict_online_pass(
        row,
        target_command_success_rate=float(args.target_command_success_rate),
        max_false_trigger_rate=float(args.max_false_trigger_rate),
        max_false_release_rate=float(args.max_false_release_rate),
    )
    return row


def _run_single_trial(
    args: argparse.Namespace,
    *,
    stage: str,
    run_id: str,
    split_seed: int,
    split_manifest_path: str,
    loss_type: str,
    base_channels: int,
    freeze_emg_epochs: int,
    encoder_lr_ratio: float,
    pretrained_mode: str,
) -> dict:
    run_dir = Path(args.run_root) / str(run_id)
    offline_summary_path = run_dir / "offline_summary.json"
    test_metrics_path = run_dir / "evaluation" / "test_metrics.json"
    control_eval_path = run_dir / "evaluation" / "control_eval_summary.json"
    recordings_manifest_path = _training_manifest_for_args(args)

    has_finetune_outputs = _reuse_trial_outputs_if_compatible(
        args,
        run_dir=run_dir,
        recordings_manifest_path=recordings_manifest_path,
        split_manifest_path=split_manifest_path,
        loss_type=loss_type,
        base_channels=base_channels,
        freeze_emg_epochs=freeze_emg_epochs,
        encoder_lr_ratio=encoder_lr_ratio,
        pretrained_mode=pretrained_mode,
    )
    if not has_finetune_outputs:
        finetune_cmd = _build_finetune_cmd(
            args,
            run_id=run_id,
            recordings_manifest_path=recordings_manifest_path,
            split_seed=split_seed,
            split_manifest_path=split_manifest_path,
            loss_type=loss_type,
            base_channels=base_channels,
            freeze_emg_epochs=freeze_emg_epochs,
            encoder_lr_ratio=encoder_lr_ratio,
            pretrained_mode=pretrained_mode,
        )
        _run_checked(f"{stage}:finetune:{run_id}", finetune_cmd)
    else:
        print(
            f"[MODEL90] {stage}:reuse_finetune:{run_id} -> "
            f"{offline_summary_path} + {test_metrics_path}",
            flush=True,
        )

    if not bool(args.skip_control_eval):
        if control_eval_path.exists():
            print(f"[MODEL90] {stage}:reuse_control_eval:{run_id} -> {control_eval_path}", flush=True)
        else:
            control_cmd = _build_control_eval_cmd(
                args,
                run_id=run_id,
                recordings_manifest_path=recordings_manifest_path,
                split_manifest_path=split_manifest_path,
            )
            _run_checked(f"{stage}:control_eval:{run_id}", control_cmd)
    return _collect_metrics_row(
        args,
        stage=stage,
        run_id=run_id,
        split_seed=split_seed,
        split_manifest_path=split_manifest_path,
        loss_type=loss_type,
        base_channels=base_channels,
        freeze_emg_epochs=freeze_emg_epochs,
        encoder_lr_ratio=encoder_lr_ratio,
        pretrained_mode=pretrained_mode,
    )


def _screen_candidates(args: argparse.Namespace) -> list[dict]:
    loss_types = _parse_tokens(args.screen_loss_types, name="--screen_loss_types")
    base_channels = _parse_int_tokens(args.screen_base_channels, name="--screen_base_channels")
    freeze_epochs = _parse_int_tokens(args.screen_freeze_emg_epochs, name="--screen_freeze_emg_epochs")
    encoder_lrs = _parse_float_tokens(args.screen_encoder_lr_ratios, name="--screen_encoder_lr_ratios")
    pretrained_modes = _parse_tokens(args.screen_pretrained_modes, name="--screen_pretrained_modes")
    candidates: list[dict] = []
    for loss_type, base_ch, freeze_ep, enc_lr, pt_mode in itertools.product(
        loss_types, base_channels, freeze_epochs, encoder_lrs, pretrained_modes
    ):
        mode = str(pt_mode).strip().lower()
        if mode not in {"off", "on"}:
            raise ValueError(f"Unsupported pretrained mode: {pt_mode!r}")
        if mode == "on" and not str(args.pretrained_emg_checkpoint or "").strip():
            continue
        candidates.append(
            {
                "loss_type": str(loss_type),
                "base_channels": int(base_ch),
                "freeze_emg_epochs": int(freeze_ep),
                "encoder_lr_ratio": float(enc_lr),
                "pretrained_mode": mode,
            }
        )
    if not candidates:
        raise RuntimeError("No screening candidates generated.")
    return candidates


def _stage_baseline(args: argparse.Namespace) -> dict:
    run_id = f"{args.run_prefix}_baseline_s{int(args.screen_split_seed)}"
    split_manifest = str(_model90_split_manifest_path(args, int(args.screen_split_seed)))
    row = _run_single_trial(
        args,
        stage="baseline",
        run_id=run_id,
        split_seed=int(args.screen_split_seed),
        split_manifest_path=split_manifest,
        loss_type=str(args.baseline_loss_type),
        base_channels=int(args.baseline_base_channels),
        freeze_emg_epochs=int(args.baseline_freeze_emg_epochs),
        encoder_lr_ratio=float(args.baseline_encoder_lr_ratio),
        pretrained_mode=str(args.baseline_pretrained_mode),
    )
    summary = {
        "status": "ok",
        "stage": "baseline",
        "run_prefix": str(args.run_prefix),
        "row": row,
        "targets": {
            "target_command_success_rate": float(args.target_command_success_rate),
            "max_false_trigger_rate": float(args.max_false_trigger_rate),
            "max_false_release_rate": float(args.max_false_release_rate),
        },
    }
    out_path = Path(args.run_root) / f"{args.run_prefix}_baseline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _stage_screen(args: argparse.Namespace) -> dict:
    candidates = _screen_candidates(args)
    rows: list[dict] = []
    split_manifest = str(_model90_split_manifest_path(args, int(args.screen_split_seed)))
    for idx, candidate in enumerate(candidates, start=1):
        run_id = (
            f"{args.run_prefix}_scr_{idx:02d}_"
            f"l{candidate['loss_type']}_bc{candidate['base_channels']}_"
            f"fz{candidate['freeze_emg_epochs']}_"
            f"elr{_safe_float_token(candidate['encoder_lr_ratio'])}_pt{candidate['pretrained_mode']}"
        )
        row = _run_single_trial(
            args,
            stage="screen",
            run_id=run_id,
            split_seed=int(args.screen_split_seed),
            split_manifest_path=split_manifest,
            loss_type=str(candidate["loss_type"]),
            base_channels=int(candidate["base_channels"]),
            freeze_emg_epochs=int(candidate["freeze_emg_epochs"]),
            encoder_lr_ratio=float(candidate["encoder_lr_ratio"]),
            pretrained_mode=str(candidate["pretrained_mode"]),
        )
        row["candidate_index"] = int(idx)
        rows.append(row)

    ranked = sorted(rows, key=_rank_row, reverse=True)
    topk = max(1, int(args.topk_for_longrun))
    top_rows = ranked[:topk]
    top_ids = {str(item["run_id"]) for item in top_rows}
    for row in rows:
        row["selected_for_longrun"] = bool(str(row.get("run_id")) in top_ids)

    summary = {
        "status": "ok",
        "stage": "screen",
        "run_prefix": str(args.run_prefix),
        "split_seed": int(args.screen_split_seed),
        "split_manifest_path": split_manifest,
        "rank_rule": (
            "pass_strict_online_gate desc, event_action_accuracy desc, "
            "event_action_macro_f1 desc, command_success_rate desc, "
            "false_trigger_rate asc, false_release_rate asc"
        ),
        "rows": rows,
        "top_candidates": top_rows,
        "best_run_id": str(top_rows[0]["run_id"]) if top_rows else "",
        "best_rank": 1 if top_rows else 0,
        "targets": {
            "target_command_success_rate": float(args.target_command_success_rate),
            "max_false_trigger_rate": float(args.max_false_trigger_rate),
            "max_false_release_rate": float(args.max_false_release_rate),
        },
    }

    out_json = Path(args.run_root) / f"{args.run_prefix}_screen_summary.json"
    out_csv = Path(args.run_root) / f"{args.run_prefix}_screen_summary.csv"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        out_csv,
        rows,
        [
            "run_id",
            "candidate_index",
            "loss_type",
            "base_channels",
            "freeze_emg_epochs",
            "encoder_lr_ratio",
            "pretrained_mode",
            "event_action_accuracy",
            "event_action_macro_f1",
            "command_success_rate",
            "false_trigger_rate",
            "false_release_rate",
            "control_eval_present",
            "pass_strict_online_gate",
            "selected_for_longrun",
            "checkpoint_path",
        ],
    )
    return summary


def _aggregate_candidate(rows: list[dict], *, candidate_key: str) -> dict:
    event_acc = [float(row.get("event_action_accuracy", 0.0)) for row in rows]
    event_f1 = [float(row.get("event_action_macro_f1", 0.0)) for row in rows]
    cmd_success = [float(row.get("command_success_rate", 0.0)) for row in rows]
    false_trigger = [float(row.get("false_trigger_rate", 1.0)) for row in rows]
    false_release = [float(row.get("false_release_rate", 1.0)) for row in rows]
    pass_count = sum(1 for row in rows if bool(row.get("pass_strict_online_gate", False)))
    return {
        "candidate_key": str(candidate_key),
        "num_runs": int(len(rows)),
        "strict_online_pass_count": int(pass_count),
        "strict_online_pass_rate": float(pass_count / len(rows)) if rows else 0.0,
        "event_action_accuracy_mean": float(mean(event_acc)) if rows else 0.0,
        "event_action_accuracy_std": float(pstdev(event_acc)) if len(rows) > 1 else 0.0,
        "event_action_macro_f1_mean": float(mean(event_f1)) if rows else 0.0,
        "event_action_macro_f1_std": float(pstdev(event_f1)) if len(rows) > 1 else 0.0,
        "command_success_rate_mean": float(mean(cmd_success)) if rows else 0.0,
        "command_success_rate_std": float(pstdev(cmd_success)) if len(rows) > 1 else 0.0,
        "false_trigger_rate_mean": float(mean(false_trigger)) if rows else 1.0,
        "false_release_rate_mean": float(mean(false_release)) if rows else 1.0,
    }


def _longrun_rank_key(summary_row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(summary_row.get("strict_online_pass_rate", 0.0)),
        float(summary_row.get("event_action_accuracy_mean", 0.0)),
        float(summary_row.get("event_action_macro_f1_mean", 0.0)),
        float(summary_row.get("command_success_rate_mean", 0.0)),
        -float(summary_row.get("false_trigger_rate_mean", 1.0)),
        -float(summary_row.get("false_release_rate_mean", 1.0)),
    )


def _build_neighbor_candidates(args: argparse.Namespace, *, reference: dict) -> list[dict]:
    base_ch = int(reference["base_channels"])
    freeze_ep = int(reference["freeze_emg_epochs"])
    enc_lr = float(reference["encoder_lr_ratio"])
    loss_type = str(reference["loss_type"])
    pretrained_mode = str(reference.get("pretrained_mode", "off"))

    lr_delta_ratio = max(0.01, float(args.neighbor_lr_delta_ratio))
    freeze_delta = max(1, int(args.neighbor_freeze_delta))
    raw_candidates = [
        {
            "variant": "ref",
            "loss_type": loss_type,
            "base_channels": base_ch,
            "freeze_emg_epochs": freeze_ep,
            "encoder_lr_ratio": enc_lr,
            "pretrained_mode": pretrained_mode,
        },
        {
            "variant": "lr_down",
            "loss_type": loss_type,
            "base_channels": base_ch,
            "freeze_emg_epochs": freeze_ep,
            "encoder_lr_ratio": max(0.01, enc_lr * (1.0 - lr_delta_ratio)),
            "pretrained_mode": pretrained_mode,
        },
        {
            "variant": "lr_up",
            "loss_type": loss_type,
            "base_channels": base_ch,
            "freeze_emg_epochs": freeze_ep,
            "encoder_lr_ratio": enc_lr * (1.0 + lr_delta_ratio),
            "pretrained_mode": pretrained_mode,
        },
        {
            "variant": "freeze_down",
            "loss_type": loss_type,
            "base_channels": base_ch,
            "freeze_emg_epochs": max(1, freeze_ep - freeze_delta),
            "encoder_lr_ratio": enc_lr,
            "pretrained_mode": pretrained_mode,
        },
        {
            "variant": "freeze_up",
            "loss_type": loss_type,
            "base_channels": base_ch,
            "freeze_emg_epochs": freeze_ep + freeze_delta,
            "encoder_lr_ratio": enc_lr,
            "pretrained_mode": pretrained_mode,
        },
    ]

    deduped: list[dict] = []
    seen = set()
    for item in raw_candidates:
        key = (
            str(item["loss_type"]),
            int(item["base_channels"]),
            int(item["freeze_emg_epochs"]),
            round(float(item["encoder_lr_ratio"]), 6),
            str(item["pretrained_mode"]),
        )
        if key in seen:
            continue
        seen.add(key)
        item["encoder_lr_ratio"] = float(f"{float(item['encoder_lr_ratio']):.6f}")
        deduped.append(item)
    return deduped


def _stage_neighbor(args: argparse.Namespace, *, longrun_summary: dict | None = None) -> dict:
    run_root = Path(args.run_root)
    if longrun_summary is None:
        longrun_summary = _load_json(run_root / f"{args.run_prefix}_longrun_summary.json")

    reference = dict(longrun_summary.get("best_run") or {})
    if not reference:
        rows = list(longrun_summary.get("rows") or [])
        if rows:
            reference = dict(sorted(rows, key=_rank_row, reverse=True)[0])
    if not reference:
        raise RuntimeError("Cannot determine neighbor reference run from longrun summary.")

    split_seed = int(reference.get("split_seed", args.screen_split_seed))
    if int(args.neighbor_split_seed) >= 0:
        split_seed = int(args.neighbor_split_seed)
    split_manifest = str(_model90_split_manifest_path(args, int(split_seed)))

    candidates = _build_neighbor_candidates(args, reference=reference)
    rows: list[dict] = []
    for idx, candidate in enumerate(candidates, start=1):
        run_id = (
            f"{args.run_prefix}_nbr_{idx:02d}_"
            f"{candidate['variant']}_"
            f"l{candidate['loss_type']}_bc{int(candidate['base_channels'])}_"
            f"fz{int(candidate['freeze_emg_epochs'])}_"
            f"elr{_safe_float_token(float(candidate['encoder_lr_ratio']))}"
        )
        row = _run_single_trial(
            args,
            stage="neighbor",
            run_id=run_id,
            split_seed=int(split_seed),
            split_manifest_path=split_manifest,
            loss_type=str(candidate["loss_type"]),
            base_channels=int(candidate["base_channels"]),
            freeze_emg_epochs=int(candidate["freeze_emg_epochs"]),
            encoder_lr_ratio=float(candidate["encoder_lr_ratio"]),
            pretrained_mode=str(candidate["pretrained_mode"]),
        )
        row["neighbor_variant"] = str(candidate["variant"])
        row["neighbor_index"] = int(idx)
        rows.append(row)

    ranked = sorted(rows, key=_rank_row, reverse=True)
    best = dict(ranked[0]) if ranked else {}
    ref_event = float(reference.get("event_action_accuracy", 0.0))
    ref_cmd = float(reference.get("command_success_rate", 0.0))
    event_gain = float(best.get("event_action_accuracy", 0.0)) - ref_event
    cmd_gain = float(best.get("command_success_rate", 0.0)) - ref_cmd
    significant = bool(
        event_gain >= float(args.neighbor_min_event_gain)
        or cmd_gain >= float(args.neighbor_min_command_gain)
    )

    summary = {
        "status": "ok",
        "stage": "neighbor",
        "run_prefix": str(args.run_prefix),
        "split_seed": int(split_seed),
        "split_manifest_path": split_manifest,
        "reference_run": reference,
        "rows": rows,
        "best_neighbor_run": best,
        "event_action_accuracy_gain": float(event_gain),
        "command_success_rate_gain": float(cmd_gain),
        "neighbor_min_event_gain": float(args.neighbor_min_event_gain),
        "neighbor_min_command_gain": float(args.neighbor_min_command_gain),
        "significant_improvement_found": significant,
    }
    out_json = run_root / f"{args.run_prefix}_neighbor_summary.json"
    out_csv = run_root / f"{args.run_prefix}_neighbor_summary.csv"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        out_csv,
        rows,
        [
            "run_id",
            "neighbor_index",
            "neighbor_variant",
            "loss_type",
            "base_channels",
            "freeze_emg_epochs",
            "encoder_lr_ratio",
            "pretrained_mode",
            "event_action_accuracy",
            "event_action_macro_f1",
            "command_success_rate",
            "false_trigger_rate",
            "false_release_rate",
            "pass_strict_online_gate",
            "checkpoint_path",
        ],
    )
    return summary


def _stage_longrun(args: argparse.Namespace, *, screen_summary: dict | None = None) -> dict:
    if screen_summary is None:
        screen_summary = _load_json(Path(args.run_root) / f"{args.run_prefix}_screen_summary.json")
    top_candidates = list(screen_summary.get("top_candidates") or [])
    if not top_candidates:
        raise RuntimeError("screen summary has no top_candidates")

    seeds = _parse_int_tokens(args.longrun_seeds, name="--longrun_seeds")
    rows: list[dict] = []
    for candidate_rank, candidate in enumerate(top_candidates, start=1):
        for seed in seeds:
            split_manifest = str(_model90_split_manifest_path(args, int(seed)))
            run_id = f"{args.run_prefix}_long_c{candidate_rank}_s{int(seed)}"
            row = _run_single_trial(
                args,
                stage="longrun",
                run_id=run_id,
                split_seed=int(seed),
                split_manifest_path=split_manifest,
                loss_type=str(candidate["loss_type"]),
                base_channels=int(candidate["base_channels"]),
                freeze_emg_epochs=int(candidate["freeze_emg_epochs"]),
                encoder_lr_ratio=float(candidate["encoder_lr_ratio"]),
                pretrained_mode=str(candidate["pretrained_mode"]),
            )
            row["candidate_rank"] = int(candidate_rank)
            row["candidate_key"] = (
                f"rank{candidate_rank}:loss={candidate['loss_type']},bc={candidate['base_channels']},"
                f"freeze={candidate['freeze_emg_epochs']},elr={candidate['encoder_lr_ratio']},"
                f"pt={candidate['pretrained_mode']}"
            )
            rows.append(row)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_key"]), []).append(row)
    candidate_summaries = [_aggregate_candidate(items, candidate_key=key) for key, items in grouped.items()]
    candidate_summaries_sorted = sorted(candidate_summaries, key=_longrun_rank_key, reverse=True)
    best_candidate_summary = dict(candidate_summaries_sorted[0]) if candidate_summaries_sorted else {}

    best_run = {}
    if rows:
        best_run = dict(sorted(rows, key=_rank_row, reverse=True)[0])

    summary = {
        "status": "ok",
        "stage": "longrun",
        "run_prefix": str(args.run_prefix),
        "rows": rows,
        "candidate_summaries": candidate_summaries_sorted,
        "best_candidate_summary": best_candidate_summary,
        "best_run": best_run,
        "best_run_id": str(best_run.get("run_id", "")),
    }
    out_json = Path(args.run_root) / f"{args.run_prefix}_longrun_summary.json"
    out_csv = Path(args.run_root) / f"{args.run_prefix}_longrun_summary.csv"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        out_csv,
        rows,
        [
            "run_id",
            "candidate_rank",
            "candidate_key",
            "split_seed",
            "loss_type",
            "base_channels",
            "freeze_emg_epochs",
            "encoder_lr_ratio",
            "pretrained_mode",
            "event_action_accuracy",
            "event_action_macro_f1",
            "command_success_rate",
            "false_trigger_rate",
            "false_release_rate",
            "control_eval_present",
            "pass_strict_online_gate",
            "checkpoint_path",
        ],
    )
    return summary


def _stage_tune(
    args: argparse.Namespace,
    *,
    longrun_summary: dict | None = None,
    screen_summary: dict | None = None,
    neighbor_summary: dict | None = None,
) -> dict:
    run_root = Path(args.run_root)
    if longrun_summary is None and (run_root / f"{args.run_prefix}_longrun_summary.json").exists():
        longrun_summary = _load_json(run_root / f"{args.run_prefix}_longrun_summary.json")
    if screen_summary is None and (run_root / f"{args.run_prefix}_screen_summary.json").exists():
        screen_summary = _load_json(run_root / f"{args.run_prefix}_screen_summary.json")
    if neighbor_summary is None and (run_root / f"{args.run_prefix}_neighbor_summary.json").exists():
        neighbor_summary = _load_json(run_root / f"{args.run_prefix}_neighbor_summary.json")

    best_run_id = ""
    if neighbor_summary:
        best_run_id = str((neighbor_summary.get("best_neighbor_run") or {}).get("run_id", "")).strip()
    if longrun_summary:
        if not best_run_id:
            best_run_id = str((longrun_summary.get("best_run") or {}).get("run_id", "")).strip()
    if not best_run_id and screen_summary:
        best_run_id = str((screen_summary.get("top_candidates") or [{}])[0].get("run_id", "")).strip()
    if not best_run_id:
        raise RuntimeError("Cannot determine best run id for threshold tuning.")

    output_json = run_root / f"{args.run_prefix}_runtime_threshold_tuning_summary.json"
    output_csv = run_root / f"{args.run_prefix}_runtime_threshold_tuning_summary.csv"
    output_runtime_config = run_root / f"{args.run_prefix}_runtime_event_onset_demo3_latch_tuned.yaml"
    cmd = [
        sys.executable,
        "scripts/tune_event_runtime_thresholds.py",
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(best_run_id),
        "--training_config",
        str(args.training_config),
        "--runtime_config",
        str(args.runtime_config),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(args.recordings_manifest),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--backend",
        str(args.control_backend),
        "--device_target",
        str(args.device_target),
        "--output_json",
        str(output_json),
        "--output_csv",
        str(output_csv),
        "--output_runtime_config",
        str(output_runtime_config),
    ]
    _run_checked("tune:runtime_thresholds", cmd)
    summary = {
        "status": "ok",
        "stage": "tune",
        "best_run_id": str(best_run_id),
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "output_runtime_config": str(output_runtime_config),
    }
    out = run_root / f"{args.run_prefix}_tune_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _stage_audit(args: argparse.Namespace) -> dict:
    output_json = Path(args.run_root) / f"{args.run_prefix}_audit_report.json"
    neighbor_summary = Path(args.run_root) / f"{args.run_prefix}_neighbor_summary.json"
    runtime_tuning_summary = Path(args.run_root) / f"{args.run_prefix}_runtime_threshold_tuning_summary.json"
    tune_summary = Path(args.run_root) / f"{args.run_prefix}_tune_summary.json"
    cmd = [
        sys.executable,
        "scripts/audit_event_model90_pipeline.py",
        "--run_root",
        str(args.run_root),
        "--run_prefix",
        str(args.run_prefix),
        "--training_config",
        str(args.training_config),
        "--runtime_config",
        str(args.runtime_config),
        "--split_manifest",
        str(_model90_split_manifest_path(args, int(args.screen_split_seed))),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(_training_manifest_for_args(args)),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--control_backend",
        str(args.control_backend),
        "--device_target",
        str(args.device_target),
        "--screen_loss_types",
        str(args.screen_loss_types),
        "--screen_base_channels",
        str(args.screen_base_channels),
        "--screen_freeze_emg_epochs",
        str(args.screen_freeze_emg_epochs),
        "--screen_encoder_lr_ratios",
        str(args.screen_encoder_lr_ratios),
        "--screen_pretrained_modes",
        str(args.screen_pretrained_modes),
        "--longrun_seeds",
        str(args.longrun_seeds),
        "--neighbor_summary",
        str(neighbor_summary),
        "--runtime_tuning_summary",
        str(runtime_tuning_summary),
        "--tune_summary",
        str(tune_summary),
        "--stability_repeats",
        str(int(args.audit_stability_repeats)),
        "--stability_tolerance",
        str(float(args.audit_stability_tolerance)),
        "--target_event_action_accuracy",
        str(float(args.target_event_action_accuracy)),
        "--target_event_action_macro_f1",
        str(float(args.target_event_action_macro_f1)),
        "--target_command_success_rate",
        str(float(args.target_command_success_rate)),
        "--max_false_trigger_rate",
        str(float(args.max_false_trigger_rate)),
        "--max_false_release_rate",
        str(float(args.max_false_release_rate)),
        "--output_json",
        str(output_json),
    ]
    if bool(args.audit_skip_stability_check):
        cmd.append("--skip_stability_check")
    if str(args.audit_stability_run_id or "").strip():
        cmd.extend(["--stability_run_id", str(args.audit_stability_run_id)])
    _run_checked("audit:design_impl_params", cmd)
    summary = {
        "status": "ok",
        "stage": "audit",
        "output_json": str(output_json),
    }
    out = Path(args.run_root) / f"{args.run_prefix}_audit_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model 0.9 sprint orchestration")
    parser.add_argument("--stage", default="all", choices=["prepare", "baseline", "screen", "longrun", "neighbor", "tune", "audit", "all"])
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="s2_model90")
    parser.add_argument("--training_config", default="configs/training_event_onset_demo3_two_stage.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo3_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=DEFAULT_SOURCE_RECORDINGS_MANIFEST)
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--device_target", default="GPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--control_backend", default="ckpt", choices=["ckpt", "lite"])
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--prepare_session_id", default=DEFAULT_PREPARE_SESSION_ID)
    parser.add_argument("--prepare_target_per_class", type=int, default=DEFAULT_PREPARE_TARGET_PER_CLASS)
    parser.add_argument("--prepare_relax_target_count", type=int, default=DEFAULT_PREPARE_RELAX_TARGET_COUNT)
    parser.add_argument("--prepare_action_min_selected_windows", type=int, default=DEFAULT_PREPARE_ACTION_MIN_WINDOWS)
    parser.add_argument("--prepare_relax_min_selected_windows", type=int, default=DEFAULT_PREPARE_RELAX_MIN_WINDOWS)
    parser.add_argument("--prepare_relax_allow_retake_quality", type=_parse_bool_arg, default=True)
    parser.add_argument("--prepare_output_manifest", default="")
    parser.add_argument("--skip_control_eval", action="store_true")
    parser.add_argument("--budget_per_class", type=int, default=0)
    parser.add_argument("--budget_seed", type=int, default=42)
    parser.add_argument("--pretrained_emg_checkpoint", default="")

    parser.add_argument("--baseline_loss_type", default="cross_entropy")
    parser.add_argument("--baseline_base_channels", type=int, default=16)
    parser.add_argument("--baseline_freeze_emg_epochs", type=int, default=6)
    parser.add_argument("--baseline_encoder_lr_ratio", type=float, default=0.3)
    parser.add_argument("--baseline_pretrained_mode", default="off", choices=["off", "on"])

    parser.add_argument("--screen_split_seed", type=int, default=42)
    parser.add_argument("--screen_split_manifest", default="")
    parser.add_argument("--screen_loss_types", default="cross_entropy,cb_focal")
    parser.add_argument("--screen_base_channels", default="16,24")
    parser.add_argument("--screen_freeze_emg_epochs", default="6,8,10")
    parser.add_argument("--screen_encoder_lr_ratios", default="0.24,0.3,0.36")
    parser.add_argument("--screen_pretrained_modes", default="off")
    parser.add_argument("--topk_for_longrun", type=int, default=2)
    parser.add_argument("--longrun_seeds", default="42,52,62")
    parser.add_argument("--neighbor_lr_delta_ratio", type=float, default=0.2)
    parser.add_argument("--neighbor_freeze_delta", type=int, default=2)
    parser.add_argument("--neighbor_split_seed", type=int, default=-1)
    parser.add_argument("--neighbor_min_event_gain", type=float, default=0.01)
    parser.add_argument("--neighbor_min_command_gain", type=float, default=0.02)

    parser.add_argument("--target_event_action_accuracy", type=float, default=0.9)
    parser.add_argument("--target_event_action_macro_f1", type=float, default=0.88)
    parser.add_argument("--target_command_success_rate", type=float, default=0.9)
    parser.add_argument("--max_false_trigger_rate", type=float, default=0.05)
    parser.add_argument("--max_false_release_rate", type=float, default=0.05)
    parser.add_argument("--audit_stability_repeats", type=int, default=2)
    parser.add_argument("--audit_stability_tolerance", type=float, default=1e-6)
    parser.add_argument("--audit_stability_run_id", default="")
    parser.add_argument("--audit_skip_stability_check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    last_stage = "init"
    last_cmd = ""
    try:
        prepare_summary = None
        baseline_summary = None
        screen_summary = None
        longrun_summary = None
        neighbor_summary = None
        tune_summary = None
        audit_summary = None

        if not bool(args.skip_prepare):
            last_stage = "prepare"
            prepare_summary = _stage_prepare(args)
        else:
            setattr(
                args,
                "_prepared_recordings_manifest",
                str(_resolve_path(args.recordings_manifest, prefer_data_dir=args.data_dir)),
            )

        if args.stage == "prepare":
            report = {
                "status": "ok",
                "stage": "prepare",
                "run_prefix": str(args.run_prefix),
                "artifacts": {
                    "prepare_summary": str(Path(args.run_root) / f"{args.run_prefix}_prepare_summary.json"),
                },
                "prepare_done": bool(prepare_summary),
            }
            out = Path(args.run_root) / f"{args.run_prefix}_pipeline_report.json"
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[MODEL90] pipeline_report={out}", flush=True)
            return

        if args.stage in {"baseline", "all"}:
            last_stage = "baseline"
            baseline_summary = _stage_baseline(args)

        if args.stage in {"screen", "all"}:
            last_stage = "screen"
            screen_summary = _stage_screen(args)

        if args.stage in {"longrun", "all"}:
            last_stage = "longrun"
            longrun_summary = _stage_longrun(args, screen_summary=screen_summary)

        if args.stage in {"neighbor", "all"}:
            last_stage = "neighbor"
            neighbor_summary = _stage_neighbor(args, longrun_summary=longrun_summary)

        if args.stage in {"tune", "all"}:
            last_stage = "tune"
            tune_summary = _stage_tune(
                args,
                longrun_summary=longrun_summary,
                screen_summary=screen_summary,
                neighbor_summary=neighbor_summary,
            )

        if args.stage in {"audit", "all"}:
            last_stage = "audit"
            audit_summary = _stage_audit(args)

        report = {
            "status": "ok",
            "stage": str(args.stage),
            "run_prefix": str(args.run_prefix),
            "artifacts": {
                "prepare_summary": str(Path(args.run_root) / f"{args.run_prefix}_prepare_summary.json"),
                "baseline_summary": str(Path(args.run_root) / f"{args.run_prefix}_baseline_summary.json"),
                "screen_summary": str(Path(args.run_root) / f"{args.run_prefix}_screen_summary.json"),
                "longrun_summary": str(Path(args.run_root) / f"{args.run_prefix}_longrun_summary.json"),
                "neighbor_summary": str(Path(args.run_root) / f"{args.run_prefix}_neighbor_summary.json"),
                "tune_summary": str(Path(args.run_root) / f"{args.run_prefix}_tune_summary.json"),
                "audit_summary": str(Path(args.run_root) / f"{args.run_prefix}_audit_summary.json"),
            },
            "prepare_done": bool(prepare_summary),
            "baseline_done": bool(baseline_summary),
            "screen_done": bool(screen_summary),
            "longrun_done": bool(longrun_summary),
            "neighbor_done": bool(neighbor_summary),
            "tune_done": bool(tune_summary),
            "audit_done": bool(audit_summary),
        }
        out = Path(args.run_root) / f"{args.run_prefix}_pipeline_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[MODEL90] pipeline_report={out}", flush=True)
    except Exception as exc:
        exc_text = str(exc)
        next_command = str(last_cmd or "")
        marker = ": "
        if marker in exc_text and "failed with rc=" in exc_text:
            next_command = exc_text.split(marker, 1)[1]
        failure = {
            "status": "failed",
            "stage": str(last_stage),
            "root_cause": exc_text,
            "next_command": str(next_command),
        }
        out = Path(args.run_root) / f"{args.run_prefix}_pipeline_failure_report.json"
        out.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
