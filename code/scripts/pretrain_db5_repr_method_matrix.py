"""Run DB5 representation pretraining stage-2 matrix (pretrain-only, 4 rounds)."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ninapro_db5.config import load_db5_pretrain_config
from ninapro_db5.dataset import DB5PretrainDatasetLoader
from ninapro_db5.evaluate import _set_device
from ninapro_db5.model import build_db5_encoder_model
from scripts import pretrain_ninapro_db5_repr as repr_script
from shared.run_utils import dump_json
from training.data.split_strategy import build_manifest


PUBLIC_RANK_RULE = "best_val_macro_f1 desc, best_val_acc desc, test_macro_f1 desc"
FEWSHOT_RANK_RULE = "recommended_budget asc, macro_f1_mean desc, acc_mean desc, macro_f1_std asc"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_checked(stage: str, cmd: list[str]) -> str:
    cmd_str = _format_cmd(cmd)
    print(f"[REPR-MATRIX] {stage} -> {cmd_str}")
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {cmd_str}")
    return cmd_str


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


# Backward-compatible helpers retained for existing tests and downstream imports.
def _compute_recommended_budget(curve_rows: list[dict], tolerance: float) -> tuple[int, dict]:
    if not curve_rows:
        raise RuntimeError("few-shot curve rows are empty.")
    sorted_rows = sorted(curve_rows, key=lambda row: int(row["budget"]))
    best_f1 = max(float(row.get("macro_f1_mean", 0.0)) for row in sorted_rows)
    floor = float(best_f1 - float(tolerance))
    for row in sorted_rows:
        if float(row.get("macro_f1_mean", 0.0)) >= floor:
            return int(row["budget"]), row
    return int(sorted_rows[-1]["budget"]), sorted_rows[-1]


def _resolve_fewshot_plan(*, mode: str, recordings_manifest: str | None) -> tuple[bool, str, str]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"off", "auto", "on"}:
        raise ValueError(f"Invalid fewshot_mode={mode!r}, expected one of: off/auto/on")

    raw_manifest = str(recordings_manifest or "").strip()
    if normalized_mode == "off":
        return False, "skipped", "fewshot_mode=off"

    if not raw_manifest:
        if normalized_mode == "auto":
            return False, "skipped", "recordings_manifest not provided"
        raise RuntimeError("fewshot_mode=on requires --recordings_manifest <path>")

    manifest_path = Path(raw_manifest)
    if not manifest_path.exists() or not manifest_path.is_file():
        if normalized_mode == "auto":
            return False, "skipped", f"recordings_manifest not found: {manifest_path}"
        raise RuntimeError(f"fewshot_mode=on requires existing recordings_manifest, got: {manifest_path}")

    return True, "enabled", ""


def _update_plateau_streak(*, previous_budget: int | None, current_budget: int, current_streak: int) -> int:
    if previous_budget is None:
        return 0
    if int(current_budget) < int(previous_budget):
        return 0
    return int(current_streak) + 1


def _rank_key(row: dict) -> tuple[float, float, float, float]:
    rec_budget = int(row.get("recommended_budget", 10**9))
    rec_row = dict(row.get("recommended_budget_row", {}) or {})
    macro_f1 = float(rec_row.get("macro_f1_mean", 0.0))
    acc = float(rec_row.get("acc_mean", 0.0))
    std = float(rec_row.get("macro_f1_std", 1.0))
    return (float(rec_budget), -macro_f1, -acc, std)


def _pretrain_rank_key(row: dict) -> tuple[float, float, float]:
    val_f1 = float(row.get("pretrain_best_val_macro_f1", 0.0))
    val_acc = float(row.get("pretrain_best_val_acc", 0.0))
    test_f1 = float(row.get("pretrain_test_macro_f1", 0.0))
    return (-val_f1, -val_acc, -test_f1)


def _parse_float_grid(raw: str, *, name: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"{name} must contain at least one numeric value")
    return values


def _parse_int_grid(raw: str, *, name: str) -> list[int]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{name} must contain at least one integer value")
    return values


def _safe_float_token(value: float) -> str:
    token = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _as_pretrain_row(*, phase: str, candidate: str, run_id: str, summary: dict, temperature: float, projection_dim: int, knn_k: int) -> dict:
    return {
        "phase": phase,
        "candidate": candidate,
        "run_id": str(run_id),
        "temperature": float(temperature),
        "projection_dim": int(projection_dim),
        "knn_k": int(knn_k),
        "repr_objective": str(summary.get("repr_objective", "supcon")),
        "sampler_mode": str(summary.get("sampler_mode", "class_source_balanced")),
        "augmentation_profile": str(summary.get("augmentation_profile", "strong")),
        "learning_rate": float(summary.get("learning_rate", 0.0) or 0.0),
        "weight_decay": float(summary.get("weight_decay", 0.0) or 0.0),
        "pretrain_best_val_macro_f1": float(summary.get("best_val_macro_f1", 0.0) or 0.0),
        "pretrain_best_val_acc": float(summary.get("best_val_acc", 0.0) or 0.0),
        "pretrain_test_macro_f1": float(summary.get("test_macro_f1", 0.0) or 0.0),
        "pretrain_test_accuracy": float(summary.get("test_accuracy", 0.0) or 0.0),
        "checkpoint_path": str(summary.get("checkpoint_path", "")),
        "pretrain_summary_path": str(summary.get("offline_summary_path", "")),
        "selected": False,
    }


def _build_pretrain_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    temperature: float,
    projection_dim: int,
    knn_k: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/pretrain_ninapro_db5_repr.py",
        "--config",
        str(args.pretrain_config),
        "--data_dir",
        str(args.db5_data_dir),
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--foundation_dir",
        str(args.foundation_dir),
        "--ms_mode",
        str(args.ms_mode),
        "--repr_objective",
        "supcon",
        "--sampler_mode",
        "class_source_balanced",
        "--augmentation_profile",
        "strong",
        "--ce_weight",
        "0.0",
        "--label_smoothing",
        str(float(args.label_smoothing)),
        "--learning_rate",
        str(float(args.learning_rate)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--epochs",
        str(int(args.epochs)),
        "--early_stopping_patience",
        str(int(args.early_stopping_patience)),
        "--temperature",
        str(float(temperature)),
        "--projection_dim",
        str(int(projection_dim)),
        "--knn_k",
        str(int(knn_k)),
        "--run_downstream_fewshot",
        "false",
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(int(args.batch_size))])
    if args.manifest_use_source_metadata is not None:
        cmd.extend(["--manifest_use_source_metadata", str(args.manifest_use_source_metadata).lower()])
    return cmd


def _run_pretrain_candidate(
    *,
    args: argparse.Namespace,
    phase: str,
    candidate: str,
    run_id: str,
    temperature: float,
    projection_dim: int,
    knn_k: int,
) -> tuple[dict, str]:
    stage = f"{phase}_{candidate}"
    cmd = _build_pretrain_cmd(
        args,
        run_id=run_id,
        temperature=temperature,
        projection_dim=projection_dim,
        knn_k=knn_k,
    )
    cmd_str = _run_checked(stage, cmd)
    summary_path = Path(args.run_root) / run_id / "offline_summary.json"
    summary = _load_json(summary_path)
    summary["offline_summary_path"] = str(summary_path)
    row = _as_pretrain_row(
        phase=phase,
        candidate=candidate,
        run_id=run_id,
        summary=summary,
        temperature=temperature,
        projection_dim=projection_dim,
        knn_k=knn_k,
    )
    if not str(row.get("checkpoint_path", "")).strip():
        raise RuntimeError(f"missing checkpoint_path in {summary_path}")
    return row, cmd_str


def _pick_best_pretrain(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("no pretrain rows available for ranking")
    return sorted(rows, key=_pretrain_rank_key)[0]


def _run4_knn_robustness(
    *,
    args: argparse.Namespace,
    checkpoint_path: str,
    knn_grid: list[int],
    run_dir: Path,
) -> dict:
    config = load_db5_pretrain_config(str(args.pretrain_config))
    config.data_dir = str(args.db5_data_dir)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.manifest_use_source_metadata is not None:
        config.manifest_use_source_metadata = bool(args.manifest_use_source_metadata)

    loader = DB5PretrainDatasetLoader(str(config.data_dir), config)
    samples, labels, source_ids, source_metadata = loader.load_all_with_sources(return_metadata=True)
    class_names = loader.get_class_names()
    if not class_names:
        raise RuntimeError("DB5 class_names is empty in run4 evaluation.")

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
    if repr_script._has_group_leakage(manifest):
        raise RuntimeError("Group leakage detected in run4 kNN robustness manifest.")

    train_idx = manifest.train_indices
    val_idx = manifest.val_indices
    test_idx = manifest.test_indices
    train_x, train_y = samples[train_idx], labels[train_idx]
    val_x, val_y = samples[val_idx], labels[val_idx]
    test_x, test_y = samples[test_idx], labels[test_idx]

    _set_device(mode=str(args.ms_mode), target=str(args.device_target), device_id=int(args.device_id))
    encoder = build_db5_encoder_model(config)

    try:
        from mindspore import load_checkpoint, load_param_into_net
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("MindSpore is required for run4 kNN robustness evaluation") from exc

    ckpt = str(checkpoint_path).strip()
    if not ckpt:
        raise RuntimeError("run4 robustness evaluation requires non-empty checkpoint path")
    params = load_checkpoint(ckpt)
    load_param_into_net(encoder, params)
    encoder.set_train(False)

    grid_rows: list[dict] = []
    for k in knn_grid:
        val_report = repr_script._evaluate_repr_knn(
            encoder,
            train_x=train_x,
            train_y=train_y,
            eval_x=val_x,
            eval_y=val_y,
            class_names=class_names,
            batch_size=int(config.training.batch_size),
            knn_k=int(k),
        )
        test_report = repr_script._evaluate_repr_knn(
            encoder,
            train_x=train_x,
            train_y=train_y,
            eval_x=test_x,
            eval_y=test_y,
            class_names=class_names,
            batch_size=int(config.training.batch_size),
            knn_k=int(k),
        )
        grid_rows.append(
            {
                "knn_k": int(k),
                "val_macro_f1": float(val_report.get("macro_f1", 0.0)),
                "val_acc": float(val_report.get("accuracy", 0.0)),
                "test_macro_f1": float(test_report.get("macro_f1", 0.0)),
                "test_accuracy": float(test_report.get("accuracy", 0.0)),
            }
        )

    def _grid_rank_key(row: dict) -> tuple[float, float, float]:
        return (
            -float(row.get("val_macro_f1", 0.0)),
            -float(row.get("val_acc", 0.0)),
            -float(row.get("test_macro_f1", 0.0)),
        )

    best_row = sorted(grid_rows, key=_grid_rank_key)[0]
    payload = {
        "status": "completed",
        "checkpoint_path": ckpt,
        "rank_rule": PUBLIC_RANK_RULE,
        "rows": grid_rows,
        "best": best_row,
    }
    dump_json(run_dir / "run4_knn_robustness.json", payload)
    _write_csv(
        run_dir / "run4_knn_robustness.csv",
        grid_rows,
        fields=["knn_k", "val_macro_f1", "val_acc", "test_macro_f1", "test_accuracy"],
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DB5 representation pretraining stage-2 matrix: "
            "Run1 baseline -> Run2 temperature search -> Run3 projection search -> Run4 kNN robustness."
        )
    )
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="db5_repr_stage2_v1")
    parser.add_argument("--pretrain_config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--db5_data_dir", default="../data_ninaproDB5")
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--ms_mode", default="graph", choices=["graph", "pynative"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--baseline_temperature", type=float, default=0.07)
    parser.add_argument("--baseline_projection_dim", type=int, default=128)
    parser.add_argument("--baseline_knn_k", type=int, default=5)
    parser.add_argument("--temperature_grid", default="0.05,0.07,0.10")
    parser.add_argument("--projection_dim_grid", default="128,192,256")
    parser.add_argument("--knn_k_grid", default="3,5,11")
    parser.add_argument("--min_val_f1_gain", type=float, default=0.01)
    parser.add_argument("--fewshot_mode", default="off", choices=["off", "auto", "on"])
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--manifest_use_source_metadata", choices=["true", "false"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_ts = time.time()
    run_root = Path(args.run_root)
    run_dir = run_root / f"{args.run_prefix}_summary"
    run_dir.mkdir(parents=True, exist_ok=True)

    temperature_grid = _parse_float_grid(args.temperature_grid, name="temperature_grid")
    projection_grid = _parse_int_grid(args.projection_dim_grid, name="projection_dim_grid")
    knn_grid = _parse_int_grid(args.knn_k_grid, name="knn_k_grid")

    rows: list[dict] = []
    commands: list[dict] = []
    last_stage = "setup"
    last_cmd = ""
    low_return_zone = False
    stop_reason = ""
    run4_payload: dict | None = None

    fewshot_mode = str(args.fewshot_mode).strip().lower()
    if fewshot_mode != "off":
        fewshot_status = "skipped"
        fewshot_skip_reason = "stage2 matrix is pretrain-only; few-shot is intentionally disabled"
    else:
        fewshot_status = "skipped"
        fewshot_skip_reason = "fewshot_mode=off"

    try:
        # Run-1: baseline reproducibility rerun (fixed backbone + baseline hyperparams).
        last_stage = "run_1_baseline"
        run1_id = f"{args.run_prefix}_r1_base"
        row1, cmd_str = _run_pretrain_candidate(
            args=args,
            phase="run_1",
            candidate="baseline",
            run_id=run1_id,
            temperature=float(args.baseline_temperature),
            projection_dim=int(args.baseline_projection_dim),
            knn_k=int(args.baseline_knn_k),
        )
        row1["selected"] = True
        last_cmd = cmd_str
        rows.append(row1)
        commands.append({"stage": last_stage, "command": cmd_str})

        # Run-2: temperature search (3 points)
        run2_rows: list[dict] = []
        for temp in temperature_grid:
            token = _safe_float_token(temp)
            run_id = f"{args.run_prefix}_r2_t_{token}"
            last_stage = f"run_2_temperature_{token}"
            row, cmd_str = _run_pretrain_candidate(
                args=args,
                phase="run_2",
                candidate=f"temp_{token}",
                run_id=run_id,
                temperature=float(temp),
                projection_dim=int(args.baseline_projection_dim),
                knn_k=int(args.baseline_knn_k),
            )
            run2_rows.append(row)
            last_cmd = cmd_str
            rows.append(row)
            commands.append({"stage": last_stage, "command": cmd_str})
        best_run2 = _pick_best_pretrain(run2_rows)
        best_run2["selected"] = True

        # Run-3: projection search (3 points) conditioned on best temperature.
        run3_rows: list[dict] = []
        best_temp = float(best_run2["temperature"])
        for proj in projection_grid:
            run_id = f"{args.run_prefix}_r3_p_{int(proj)}"
            last_stage = f"run_3_projection_{int(proj)}"
            row, cmd_str = _run_pretrain_candidate(
                args=args,
                phase="run_3",
                candidate=f"proj_{int(proj)}",
                run_id=run_id,
                temperature=best_temp,
                projection_dim=int(proj),
                knn_k=int(args.baseline_knn_k),
            )
            run3_rows.append(row)
            last_cmd = cmd_str
            rows.append(row)
            commands.append({"stage": last_stage, "command": cmd_str})
        best_run3 = _pick_best_pretrain(run3_rows)
        best_run3["selected"] = True

        run3_gain = float(best_run3["pretrain_best_val_macro_f1"]) - float(row1["pretrain_best_val_macro_f1"])
        if run3_gain < float(args.min_val_f1_gain):
            low_return_zone = True
            stop_reason = (
                f"run3 gain below threshold: gain={run3_gain:.4f} < min_val_f1_gain={float(args.min_val_f1_gain):.4f}"
            )
        else:
            # Run-4: kNN robustness grid on Run-3 best checkpoint (evaluation-only, no retraining).
            last_stage = "run_4_knn_robustness"
            checkpoint_path = str(best_run3.get("checkpoint_path", "")).strip()
            if not checkpoint_path:
                raise RuntimeError("run_4_knn_robustness requires non-empty checkpoint_path from run_3 best")
            run4_payload = _run4_knn_robustness(
                args=args,
                checkpoint_path=checkpoint_path,
                knn_grid=knn_grid,
                run_dir=run_dir,
            )
            commands.append(
                {
                    "stage": last_stage,
                    "command": "internal_eval:run_4_knn_robustness",
                    "checkpoint_path": checkpoint_path,
                    "knn_grid": knn_grid,
                }
            )

    except Exception as exc:
        default_cmd = (
            "python scripts/pretrain_db5_repr_method_matrix.py "
            f"--run_prefix {args.run_prefix} --db5_data_dir {args.db5_data_dir} --fewshot_mode off"
        )
        dump_json(
            run_dir / "repr_method_matrix_failure_report.json",
            {
                "status": "failed",
                "stage": str(last_stage),
                "root_cause": str(exc),
                "next_command": str(last_cmd or default_cmd),
                "generated_at_unix": int(time.time()),
            },
        )
        raise

    if not rows:
        raise RuntimeError("no successful stage2 pretrain rows")

    best_pretrain = _pick_best_pretrain(rows)
    elapsed_minutes = float((time.time() - start_ts) / 60.0)

    matrix_payload = {
        "run_prefix": str(args.run_prefix),
        "fewshot_mode": fewshot_mode,
        "fewshot_status": fewshot_status,
        "fewshot_skip_reason": fewshot_skip_reason,
        "public_rank_rule": PUBLIC_RANK_RULE,
        "fewshot_rank_rule": FEWSHOT_RANK_RULE,
        "fixed_backbone": {
            "repr_objective": "supcon",
            "sampler_mode": "class_source_balanced",
            "augmentation_profile": "strong",
            "ce_weight": 0.0,
        },
        "search_space": {
            "temperature_grid": temperature_grid,
            "projection_dim_grid": projection_grid,
            "knn_k_grid": knn_grid,
        },
        "run_1_baseline": row1,
        "run_2_best": best_run2,
        "run_3_best": best_run3,
        "run3_minus_run1_val_f1_gain": float(
            float(best_run3["pretrain_best_val_macro_f1"]) - float(row1["pretrain_best_val_macro_f1"])
        ),
        "min_val_f1_gain_threshold": float(args.min_val_f1_gain),
        "entered_low_return_zone": bool(low_return_zone),
        "stop_reason": str(stop_reason),
        "run_4_knn_robustness": run4_payload,
        "best_pretrain_run": best_pretrain,
        "best_pretrain_run_id": str(best_pretrain.get("run_id", "")),
        "rows": rows,
        "commands": commands,
        "elapsed_minutes": elapsed_minutes,
    }
    dump_json(run_dir / "db5_repr_method_matrix_summary.json", matrix_payload)
    _write_csv(
        run_dir / "db5_repr_method_matrix_summary.csv",
        rows,
        fields=[
            "phase",
            "candidate",
            "selected",
            "run_id",
            "temperature",
            "projection_dim",
            "knn_k",
            "repr_objective",
            "sampler_mode",
            "augmentation_profile",
            "learning_rate",
            "weight_decay",
            "pretrain_best_val_macro_f1",
            "pretrain_best_val_acc",
            "pretrain_test_macro_f1",
            "pretrain_test_accuracy",
            "checkpoint_path",
            "pretrain_summary_path",
        ],
    )

    card = "\n".join(
        [
            "# DB5 Repr Stage-2 Matrix Repro Card",
            "",
            "## Goal",
            "- strengthen DB5 foundation representation pretraining",
            "- keep public reproducibility track pretrain-only",
            "",
            "## Fixed Backbone",
            "- repr_objective: `supcon`",
            "- sampler_mode: `class_source_balanced`",
            "- augmentation_profile: `strong`",
            "",
            "## Stage Results",
            f"- run_1_baseline: `{row1['run_id']}` val_f1={float(row1['pretrain_best_val_macro_f1']):.4f}",
            f"- run_2_best: `{best_run2['run_id']}` temp={float(best_run2['temperature']):.4f} val_f1={float(best_run2['pretrain_best_val_macro_f1']):.4f}",
            f"- run_3_best: `{best_run3['run_id']}` proj={int(best_run3['projection_dim'])} val_f1={float(best_run3['pretrain_best_val_macro_f1']):.4f}",
            f"- run3_gain_vs_run1: `{float(best_run3['pretrain_best_val_macro_f1']) - float(row1['pretrain_best_val_macro_f1']):.4f}`",
            f"- low_return_zone: `{bool(low_return_zone)}`",
            f"- stop_reason: `{stop_reason}`",
            f"- run_4_status: `{'completed' if run4_payload else 'skipped'}`",
            f"- run_4_best_knn_k: `{int(run4_payload['best']['knn_k']) if run4_payload else ''}`",
            "",
            "## Best Pretrain",
            f"- run_id: `{best_pretrain['run_id']}`",
            f"- best_val_macro_f1: `{float(best_pretrain['pretrain_best_val_macro_f1']):.4f}`",
            f"- best_val_acc: `{float(best_pretrain['pretrain_best_val_acc']):.4f}`",
            f"- test_macro_f1: `{float(best_pretrain['pretrain_test_macro_f1']):.4f}`",
            f"- checkpoint: `{best_pretrain['checkpoint_path']}`",
            f"- public_rank_rule: `{PUBLIC_RANK_RULE}`",
            "",
            "## Commands",
            "- public_pretrain_only: `python scripts/pretrain_db5_repr_method_matrix.py --fewshot_mode off --db5_data_dir ../data_ninaproDB5 --run_prefix <run_id>`",
            "",
            "## Artifacts",
            f"- summary_json: `{run_dir / 'db5_repr_method_matrix_summary.json'}`",
            f"- summary_csv: `{run_dir / 'db5_repr_method_matrix_summary.csv'}`",
            f"- run4_knn_json: `{run_dir / 'run4_knn_robustness.json'}`",
            f"- run4_knn_csv: `{run_dir / 'run4_knn_robustness.csv'}`",
        ]
    )
    (run_dir / "referee_repro_card.md").write_text(card, encoding="utf-8")
    print(f"[REPR-MATRIX] summary={run_dir / 'db5_repr_method_matrix_summary.json'}")
    print(f"[REPR-MATRIX] card={run_dir / 'referee_repro_card.md'}")


if __name__ == "__main__":
    main()
