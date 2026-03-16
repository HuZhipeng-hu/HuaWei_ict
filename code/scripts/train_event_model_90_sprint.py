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


DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW"
DEFAULT_SCREEN_MANIFEST = "artifacts/splits/s2_relax12_4class_seed42_v2.json"


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


def _safe_float_token(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


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
    try:
        return float(payload.get(key, default) or default)
    except Exception:
        return float(default)


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
    split_seed: int,
    split_manifest_path: str,
    loss_type: str | None,
    base_channels: int | None,
    freeze_emg_epochs: int | None,
    encoder_lr_ratio: float | None,
    pretrained_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/finetune_event_onset.py",
        "--config",
        str(args.training_config),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(args.recordings_manifest),
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
        "--split_manifest_in",
        str(split_manifest_path),
        "--split_manifest_out",
        str(split_manifest_path),
        "--budget_per_class",
        str(int(args.budget_per_class)),
        "--budget_seed",
        str(int(args.budget_seed)),
    ]
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
        str(args.recordings_manifest),
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
    finetune_cmd = _build_finetune_cmd(
        args,
        run_id=run_id,
        split_seed=split_seed,
        split_manifest_path=split_manifest_path,
        loss_type=loss_type,
        base_channels=base_channels,
        freeze_emg_epochs=freeze_emg_epochs,
        encoder_lr_ratio=encoder_lr_ratio,
        pretrained_mode=pretrained_mode,
    )
    _run_checked(f"{stage}:finetune:{run_id}", finetune_cmd)
    if not bool(args.skip_control_eval):
        control_cmd = _build_control_eval_cmd(
            args,
            run_id=run_id,
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
    row = _run_single_trial(
        args,
        stage="baseline",
        run_id=run_id,
        split_seed=int(args.screen_split_seed),
        split_manifest_path=str(args.screen_split_manifest),
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
    split_manifest = str(args.screen_split_manifest)
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
            split_manifest = f"artifacts/splits/s2_relax12_4class_seed{int(seed)}_v2.json"
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


def _stage_tune(args: argparse.Namespace, *, longrun_summary: dict | None = None, screen_summary: dict | None = None) -> dict:
    run_root = Path(args.run_root)
    if longrun_summary is None and (run_root / f"{args.run_prefix}_longrun_summary.json").exists():
        longrun_summary = _load_json(run_root / f"{args.run_prefix}_longrun_summary.json")
    if screen_summary is None and (run_root / f"{args.run_prefix}_screen_summary.json").exists():
        screen_summary = _load_json(run_root / f"{args.run_prefix}_screen_summary.json")

    best_run_id = ""
    if longrun_summary:
        best_run_id = str((longrun_summary.get("best_run") or {}).get("run_id", "")).strip()
    if not best_run_id and screen_summary:
        best_run_id = str((screen_summary.get("top_candidates") or [{}])[0].get("run_id", "")).strip()
    if not best_run_id:
        raise RuntimeError("Cannot determine best run id for threshold tuning.")

    output_json = run_root / f"{args.run_prefix}_runtime_threshold_tuning_summary.json"
    output_csv = run_root / f"{args.run_prefix}_runtime_threshold_tuning_summary.csv"
    output_runtime_config = run_root / f"{args.run_prefix}_runtime_event_onset_demo_latch_tuned.yaml"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model 0.9 sprint orchestration")
    parser.add_argument("--stage", default="all", choices=["baseline", "screen", "longrun", "tune", "all"])
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="s2_model90")
    parser.add_argument("--training_config", default="configs/training_event_onset_demo_p0.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default="s2_train_manifest_relax12.csv")
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--device_target", default="GPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--control_backend", default="ckpt", choices=["ckpt", "lite"])
    parser.add_argument("--skip_control_eval", action="store_true")
    parser.add_argument("--budget_per_class", type=int, default=0)
    parser.add_argument("--budget_seed", type=int, default=42)
    parser.add_argument("--pretrained_emg_checkpoint", default="")

    parser.add_argument("--baseline_loss_type", default="cross_entropy")
    parser.add_argument("--baseline_base_channels", type=int, default=16)
    parser.add_argument("--baseline_freeze_emg_epochs", type=int, default=5)
    parser.add_argument("--baseline_encoder_lr_ratio", type=float, default=0.3)
    parser.add_argument("--baseline_pretrained_mode", default="off", choices=["off", "on"])

    parser.add_argument("--screen_split_seed", type=int, default=42)
    parser.add_argument("--screen_split_manifest", default=DEFAULT_SCREEN_MANIFEST)
    parser.add_argument("--screen_loss_types", default="cross_entropy,cb_focal")
    parser.add_argument("--screen_base_channels", default="16,24")
    parser.add_argument("--screen_freeze_emg_epochs", default="5,8")
    parser.add_argument("--screen_encoder_lr_ratios", default="0.3,0.2")
    parser.add_argument("--screen_pretrained_modes", default="off,on")
    parser.add_argument("--topk_for_longrun", type=int, default=2)
    parser.add_argument("--longrun_seeds", default="42,52,62")

    parser.add_argument("--target_command_success_rate", type=float, default=0.9)
    parser.add_argument("--max_false_trigger_rate", type=float, default=0.05)
    parser.add_argument("--max_false_release_rate", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    last_stage = "init"
    last_cmd = ""
    try:
        baseline_summary = None
        screen_summary = None
        longrun_summary = None
        tune_summary = None

        if args.stage in {"baseline", "all"}:
            last_stage = "baseline"
            baseline_summary = _stage_baseline(args)

        if args.stage in {"screen", "all"}:
            last_stage = "screen"
            screen_summary = _stage_screen(args)

        if args.stage in {"longrun", "all"}:
            last_stage = "longrun"
            longrun_summary = _stage_longrun(args, screen_summary=screen_summary)

        if args.stage in {"tune", "all"}:
            last_stage = "tune"
            tune_summary = _stage_tune(args, longrun_summary=longrun_summary, screen_summary=screen_summary)

        report = {
            "status": "ok",
            "stage": str(args.stage),
            "run_prefix": str(args.run_prefix),
            "artifacts": {
                "baseline_summary": str(Path(args.run_root) / f"{args.run_prefix}_baseline_summary.json"),
                "screen_summary": str(Path(args.run_root) / f"{args.run_prefix}_screen_summary.json"),
                "longrun_summary": str(Path(args.run_root) / f"{args.run_prefix}_longrun_summary.json"),
                "tune_summary": str(Path(args.run_root) / f"{args.run_prefix}_tune_summary.json"),
            },
            "baseline_done": bool(baseline_summary),
            "screen_done": bool(screen_summary),
            "longrun_done": bool(longrun_summary),
            "tune_done": bool(tune_summary),
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
