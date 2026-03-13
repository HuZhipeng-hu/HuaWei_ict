"""Run DB5 representation pretraining method matrix with data-budget plateau early stop."""

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

from shared.run_utils import dump_json


ROUND_LIBRARY: list[dict[str, object]] = [
    {
        "name": "baseline_supcon",
        "repr_objective": "supcon",
        "sampler_mode": "class_balanced",
        "augmentation_profile": "mild",
        "ce_weight": 0.0,
        "learning_rate": 5e-4,
        "weight_decay": 3e-4,
    },
    {
        "name": "source_balanced_strong_aug",
        "repr_objective": "supcon",
        "sampler_mode": "class_source_balanced",
        "augmentation_profile": "strong",
        "ce_weight": 0.0,
        "learning_rate": 5e-4,
        "weight_decay": 3e-4,
    },
    {
        "name": "joint_supcon_ce",
        "repr_objective": "supcon_ce",
        "sampler_mode": "class_source_balanced",
        "augmentation_profile": "strong",
        "ce_weight": 0.35,
        "learning_rate": 5e-4,
        "weight_decay": 3e-4,
    },
    {
        "name": "joint_lr3e4",
        "repr_objective": "supcon_ce",
        "sampler_mode": "class_source_balanced",
        "augmentation_profile": "strong",
        "ce_weight": 0.35,
        "learning_rate": 3e-4,
        "weight_decay": 3e-4,
    },
    {
        "name": "joint_wd1e3",
        "repr_objective": "supcon_ce",
        "sampler_mode": "class_source_balanced",
        "augmentation_profile": "strong",
        "ce_weight": 0.35,
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
    },
    {
        "name": "joint_lr8e4",
        "repr_objective": "supcon_ce",
        "sampler_mode": "class_source_balanced",
        "augmentation_profile": "strong",
        "ce_weight": 0.35,
        "learning_rate": 8e-4,
        "weight_decay": 3e-4,
    },
]

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DB5 representation method matrix. Public reproducibility mode is default (few-shot disabled); "
            "few-shot can be enabled explicitly."
        )
    )
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="db5_repr_matrix_v1")
    parser.add_argument("--pretrain_config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--fewshot_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--db5_data_dir", default="../data_ninaproDB5")
    parser.add_argument("--wearer_data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--fewshot_mode", default="off", choices=["off", "auto", "on"])
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--target_db5_keys", default="E1_G01,E1_G02,E1_G03,E1_G04")
    parser.add_argument("--budgets", default="10,20,35,60")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--plateau_patience", type=int, default=2)
    parser.add_argument("--budget_tolerance", type=float, default=0.02)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--ms_mode", default="graph", choices=["graph", "pynative"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_ts = time.time()
    run_root = Path(args.run_root)
    run_dir = run_root / f"{args.run_prefix}_summary"
    run_dir.mkdir(parents=True, exist_ok=True)

    round_specs = ROUND_LIBRARY[: max(1, int(args.max_rounds))]
    rows: list[dict] = []
    commands: list[dict] = []
    plateau_streak = 0
    prev_budget: int | None = None
    stopped_early = False
    stop_reason = ""
    last_stage = "setup"
    last_cmd = ""
    default_public_command = (
        "python scripts/pretrain_db5_repr_method_matrix.py "
        f"--pretrain_config {args.pretrain_config} --db5_data_dir {args.db5_data_dir} "
        f"--run_root {args.run_root} --run_prefix {args.run_prefix} "
        f"--device_target {args.device_target} --device_id {int(args.device_id)} --fewshot_mode off"
    )
    fewshot_mode = str(args.fewshot_mode).strip().lower()
    run_fewshot = False
    fewshot_status = "skipped"
    fewshot_skip_reason = ""

    try:
        last_stage = "fewshot_gate"
        run_fewshot, fewshot_status, fewshot_skip_reason = _resolve_fewshot_plan(
            mode=fewshot_mode,
            recordings_manifest=args.recordings_manifest,
        )
        if not run_fewshot and not fewshot_skip_reason:
            fewshot_skip_reason = f"fewshot_mode={fewshot_mode}"

        for idx, spec in enumerate(round_specs, start=1):
            last_stage = f"round_{idx}_pretrain"
            pretrain_run_id = f"{args.run_prefix}_r{idx}_repr"
            pretrain_cmd = [
                sys.executable,
                "scripts/pretrain_ninapro_db5_repr.py",
                "--config",
                str(args.pretrain_config),
                "--data_dir",
                str(args.db5_data_dir),
                "--run_root",
                str(args.run_root),
                "--run_id",
                pretrain_run_id,
                "--device_target",
                str(args.device_target),
                "--device_id",
                str(int(args.device_id)),
                "--foundation_dir",
                str(args.foundation_dir),
                "--ms_mode",
                str(args.ms_mode),
                "--repr_objective",
                str(spec["repr_objective"]),
                "--sampler_mode",
                str(spec["sampler_mode"]),
                "--augmentation_profile",
                str(spec["augmentation_profile"]),
                "--ce_weight",
                str(float(spec["ce_weight"])),
                "--label_smoothing",
                str(float(args.label_smoothing)),
                "--learning_rate",
                str(float(spec["learning_rate"])),
                "--weight_decay",
                str(float(spec["weight_decay"])),
                "--run_downstream_fewshot",
                "false",
            ]
            if args.epochs is not None:
                pretrain_cmd.extend(["--epochs", str(int(args.epochs))])
            if args.batch_size is not None:
                pretrain_cmd.extend(["--batch_size", str(int(args.batch_size))])
            last_cmd = _run_checked(last_stage, pretrain_cmd)
            commands.append({"stage": last_stage, "command": last_cmd})
            pretrain_summary = _load_json(run_root / pretrain_run_id / "offline_summary.json")
            encoder_ckpt = str(pretrain_summary.get("checkpoint_path", "")).strip()
            if not encoder_ckpt:
                raise RuntimeError(f"missing checkpoint_path in {pretrain_run_id}/offline_summary.json")

            row = {
                "round": int(idx),
                "name": str(spec["name"]),
                "pretrain_run_id": pretrain_run_id,
                "fewshot_run_id": "",
                "repr_objective": str(spec["repr_objective"]),
                "sampler_mode": str(spec["sampler_mode"]),
                "augmentation_profile": str(spec["augmentation_profile"]),
                "learning_rate": float(spec["learning_rate"]),
                "weight_decay": float(spec["weight_decay"]),
                "pretrain_best_val_macro_f1": float(pretrain_summary.get("best_val_macro_f1", 0.0)),
                "pretrain_best_val_acc": float(pretrain_summary.get("best_val_acc", 0.0)),
                "pretrain_test_macro_f1": float(pretrain_summary.get("test_macro_f1", 0.0)),
                "pretrain_test_accuracy": float(pretrain_summary.get("test_accuracy", 0.0)),
                "encoder_checkpoint_path": encoder_ckpt,
                "pretrain_summary_path": str(run_root / pretrain_run_id / "offline_summary.json"),
                "fewshot_status": fewshot_status if not run_fewshot else "enabled",
                "fewshot_skip_reason": fewshot_skip_reason if not run_fewshot else "",
                "recommended_budget": None,
                "recommended_macro_f1_mean": None,
                "recommended_acc_mean": None,
                "recommended_macro_f1_std": None,
                "best_budget": None,
                "plateau_streak": int(plateau_streak),
                "fewshot_report_path": "",
            }

            if run_fewshot:
                last_stage = f"round_{idx}_fewshot"
                fewshot_run_id = f"{args.run_prefix}_r{idx}_fewshot"
                fewshot_cmd = [
                    sys.executable,
                    "scripts/evaluate_event_fewshot_curve.py",
                    "--config",
                    str(args.fewshot_config),
                    "--data_dir",
                    str(args.wearer_data_dir),
                    "--run_root",
                    str(args.run_root),
                    "--run_id",
                    fewshot_run_id,
                    "--device_target",
                    str(args.device_target),
                    "--device_id",
                    str(int(args.device_id)),
                    "--pretrained_emg_checkpoint",
                    encoder_ckpt,
                    "--target_db5_keys",
                    str(args.target_db5_keys),
                    "--budgets",
                    str(args.budgets),
                    "--seeds",
                    str(args.seeds),
                    "--recordings_manifest",
                    str(args.recordings_manifest).strip(),
                ]
                last_cmd = _run_checked(last_stage, fewshot_cmd)
                commands.append({"stage": last_stage, "command": last_cmd})
                fewshot_report = _load_json(run_root / fewshot_run_id / "downstream_fewshot_report.json")
                curve_rows = list(fewshot_report.get("curve_summary") or [])
                recommended_budget, recommended_row = _compute_recommended_budget(
                    curve_rows,
                    tolerance=float(args.budget_tolerance),
                )
                plateau_streak = _update_plateau_streak(
                    previous_budget=prev_budget,
                    current_budget=int(recommended_budget),
                    current_streak=plateau_streak,
                )
                row.update(
                    {
                        "fewshot_run_id": fewshot_run_id,
                        "fewshot_status": "enabled",
                        "recommended_budget": int(recommended_budget),
                        "recommended_macro_f1_mean": float(recommended_row.get("macro_f1_mean", 0.0)),
                        "recommended_acc_mean": float(recommended_row.get("acc_mean", 0.0)),
                        "recommended_macro_f1_std": float(recommended_row.get("macro_f1_std", 0.0)),
                        "best_budget": int(fewshot_report.get("best_budget", recommended_budget)),
                        "best_budget_row": dict(fewshot_report.get("best_budget_row") or {}),
                        "recommended_budget_row": recommended_row,
                        "plateau_streak": int(plateau_streak),
                        "fewshot_report_path": str(run_root / fewshot_run_id / "downstream_fewshot_report.json"),
                    }
                )
                prev_budget = int(recommended_budget)

            rows.append(row)
            if run_fewshot and plateau_streak >= int(args.plateau_patience):
                stopped_early = True
                stop_reason = (
                    f"recommended budget did not improve for {int(args.plateau_patience)} consecutive rounds"
                )
                break

    except Exception as exc:
        dump_json(
            run_dir / "repr_method_matrix_failure_report.json",
            {
                "status": "failed",
                "stage": str(last_stage),
                "root_cause": str(exc),
                "next_command": str(last_cmd or default_public_command),
                "generated_at_unix": int(time.time()),
            },
        )
        raise

    if not rows:
        raise RuntimeError("no successful method-matrix rounds.")
    best_pretrain = sorted(rows, key=_pretrain_rank_key)[0]
    fewshot_rows = [row for row in rows if row.get("recommended_budget") is not None]
    best_fewshot = sorted(fewshot_rows, key=_rank_key)[0] if fewshot_rows else None
    elapsed_minutes = float((time.time() - start_ts) / 60.0)

    public_cmd_template = (
        "python scripts/pretrain_db5_repr_method_matrix.py "
        "--fewshot_mode off --db5_data_dir ../data_ninaproDB5 --run_prefix <run_id>"
    )
    internal_cmd_template = (
        "python scripts/pretrain_db5_repr_method_matrix.py "
        "--fewshot_mode on --recordings_manifest <path/to/recordings_manifest.csv> "
        "--db5_data_dir ../data_ninaproDB5 --wearer_data_dir ../data --run_prefix <run_id>"
    )

    matrix_payload = {
        "run_prefix": str(args.run_prefix),
        "fewshot_mode": fewshot_mode,
        "fewshot_status": "enabled" if run_fewshot else fewshot_status,
        "fewshot_skip_reason": "" if run_fewshot else fewshot_skip_reason,
        "recordings_manifest": str(args.recordings_manifest or ""),
        "public_rank_rule": PUBLIC_RANK_RULE,
        "fewshot_rank_rule": FEWSHOT_RANK_RULE,
        "budget_tolerance": float(args.budget_tolerance),
        "plateau_patience": int(args.plateau_patience),
        "max_rounds": int(args.max_rounds),
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "best_pretrain_run": best_pretrain,
        "best_pretrain_round": int(best_pretrain["round"]),
        "best_fewshot_run": best_fewshot,
        "best_fewshot_round": int(best_fewshot["round"]) if best_fewshot else None,
        "recommended_budget": int(best_fewshot["recommended_budget"]) if best_fewshot else None,
        "command_templates": {
            "public_pretrain_only": public_cmd_template,
            "internal_with_fewshot": internal_cmd_template,
        },
        "rows": rows,
        "commands": commands,
        "elapsed_minutes": elapsed_minutes,
    }
    dump_json(run_dir / "db5_repr_method_matrix_summary.json", matrix_payload)
    _write_csv(
        run_dir / "db5_repr_method_matrix_summary.csv",
        rows,
        fields=[
            "round",
            "name",
            "repr_objective",
            "sampler_mode",
            "augmentation_profile",
            "learning_rate",
            "weight_decay",
            "pretrain_best_val_macro_f1",
            "pretrain_best_val_acc",
            "pretrain_test_macro_f1",
            "pretrain_test_accuracy",
            "fewshot_status",
            "fewshot_skip_reason",
            "recommended_budget",
            "recommended_macro_f1_mean",
            "recommended_acc_mean",
            "recommended_macro_f1_std",
            "best_budget",
            "plateau_streak",
            "pretrain_run_id",
            "fewshot_run_id",
            "encoder_checkpoint_path",
            "pretrain_summary_path",
            "fewshot_report_path",
        ],
    )

    card = "\n".join(
        [
            "# DB5 Repr Method Matrix Repro Card",
            "",
            "## Goal",
            "- public reproducibility: foundation pretraining with DB5 only",
            "- few-shot is optional transfer evaluation (internal extension)",
            "- no personal calibration data required",
            "",
            "## Public Track (Default)",
            f"- fewshot_mode: `{fewshot_mode}`",
            f"- best_pretrain_round: `{best_pretrain['round']}` ({best_pretrain['name']})",
            f"- best_val_macro_f1: `{float(best_pretrain.get('pretrain_best_val_macro_f1', 0.0)):.4f}`",
            f"- best_val_acc: `{float(best_pretrain.get('pretrain_best_val_acc', 0.0)):.4f}`",
            f"- test_macro_f1: `{float(best_pretrain.get('pretrain_test_macro_f1', 0.0)):.4f}`",
            f"- encoder_checkpoint: `{best_pretrain['encoder_checkpoint_path']}`",
            f"- public_rank_rule: `{PUBLIC_RANK_RULE}`",
            "",
            "## Internal Few-shot Track",
            f"- fewshot_status: `{'enabled' if run_fewshot else fewshot_status}`",
            f"- fewshot_skip_reason: `{'' if run_fewshot else fewshot_skip_reason}`",
            f"- recommended_budget: `{int(best_fewshot['recommended_budget']) if best_fewshot else ''}`",
            f"- fewshot_rank_rule: `{FEWSHOT_RANK_RULE}`",
            "",
            "## Commands",
            f"- public_pretrain_only: `{public_cmd_template}`",
            f"- internal_with_fewshot: `{internal_cmd_template}`",
            "",
            "## Artifacts",
            f"- stopped_early: `{stopped_early}`",
            f"- stop_reason: `{stop_reason}`",
            f"- summary_json: `{run_dir / 'db5_repr_method_matrix_summary.json'}`",
            f"- summary_csv: `{run_dir / 'db5_repr_method_matrix_summary.csv'}`",
        ]
    )
    (run_dir / "referee_repro_card.md").write_text(card, encoding="utf-8")
    print(f"[REPR-MATRIX] summary={run_dir / 'db5_repr_method_matrix_summary.json'}")
    print(f"[REPR-MATRIX] card={run_dir / 'referee_repro_card.md'}")


if __name__ == "__main__":
    main()
