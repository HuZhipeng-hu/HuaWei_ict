"""Direct training matrix for competition demo actions (6 actions + RELAX)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


DEFAULT_TARGET_KEYS = "TENSE_OPEN,V_SIGN,OK_SIGN,THUMB_UP,WRIST_CW,WRIST_CCW"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run_checked(stage: str, cmd: list[str]) -> None:
    print(f"[DEMO-DIRECT] {stage} -> {_format_cmd(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {_format_cmd(cmd)}")


def _load_offline_summary(run_root: Path, run_id: str) -> dict:
    path = run_root / run_id / "offline_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"offline_summary missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["run_id"] = run_id
    data["summary_path"] = str(path)
    data["per_class_csv"] = str((run_root / run_id / "evaluation" / "test_per_class_metrics.csv"))
    data["confusion_csv"] = str((run_root / run_id / "evaluation" / "test_confusion_matrix.csv"))
    data["metrics_json"] = str((run_root / run_id / "evaluation" / "test_metrics.json"))
    return data


def _build_finetune_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    with_pretrain: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/finetune_event_onset.py",
        "--config",
        str(args.config),
        "--data_dir",
        str(args.data_dir),
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(args.device_id),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--budget_per_class",
        str(int(args.budget_per_class)),
        "--budget_seed",
        str(int(args.budget_seed)),
        "--split_seed",
        str(int(args.split_seed)),
    ]
    if str(args.recordings_manifest or "").strip():
        cmd.extend(["--recordings_manifest", str(args.recordings_manifest)])
    if with_pretrain:
        if not str(args.pretrained_emg_checkpoint or "").strip():
            raise ValueError("with_pretrain run requested but --pretrained_emg_checkpoint is empty.")
        cmd.extend(["--pretrained_emg_checkpoint", str(args.pretrained_emg_checkpoint)])
    return cmd


def _rank_key(row: dict) -> tuple[float, float]:
    return (float(row.get("test_macro_f1", 0.0)), float(row.get("test_accuracy", 0.0)))


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "stage",
        "track",
        "run_id",
        "used_pretrained_init",
        "test_macro_f1",
        "test_accuracy",
        "test_macro_recall",
        "checkpoint_path",
        "per_class_csv",
        "confusion_csv",
        "metrics_json",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def _track_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "mean_test_macro_f1": 0.0, "std_test_macro_f1": 0.0, "best_run_id": None}
    f1s = [float(row.get("test_macro_f1", 0.0)) for row in rows]
    best = sorted(rows, key=_rank_key, reverse=True)[0]
    return {
        "count": len(rows),
        "mean_test_macro_f1": float(mean(f1s)),
        "std_test_macro_f1": float(pstdev(f1s)) if len(f1s) > 1 else 0.0,
        "best_run_id": best.get("run_id"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct event-onset training matrix for 6 demo actions.")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="event_demo6_direct_v1")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--budget_per_class", type=int, default=60)
    parser.add_argument("--budget_seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--pretrained_emg_checkpoint",
        default=None,
        help="Optional DB5 pretrained encoder checkpoint. When provided, with_pretrain track will be executed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    stages: list[tuple[str, str, bool]] = [
        ("scratch_baseline", "scratch", False),
        ("scratch_rerun", "scratch", False),
    ]
    if str(args.pretrained_emg_checkpoint or "").strip():
        stages.extend(
            [
                ("pretrain_baseline", "pretrain", True),
                ("pretrain_rerun", "pretrain", True),
            ]
        )

    rows: list[dict] = []
    last_stage = "build_commands"
    last_cmd = ""
    try:
        for stage, track, with_pretrain in stages:
            run_id = f"{args.run_prefix}_{stage}"
            cmd = _build_finetune_cmd(args, run_id=run_id, with_pretrain=with_pretrain)
            last_stage = stage
            last_cmd = _format_cmd(cmd)
            _run_checked(stage, cmd)
            summary = _load_offline_summary(run_root, run_id)
            summary["stage"] = stage
            summary["track"] = track
            rows.append(summary)
    except Exception as exc:
        failure = {
            "status": "failed",
            "stage": last_stage,
            "root_cause": str(exc),
            "next_command": last_cmd,
        }
        out_dir = run_root / f"{args.run_prefix}_summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "event_demo_direct_failure_report.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise

    scratch_rows = [row for row in rows if row.get("track") == "scratch"]
    pretrain_rows = [row for row in rows if row.get("track") == "pretrain"]
    best_overall = sorted(rows, key=_rank_key, reverse=True)[0] if rows else {}
    scratch_stats = _track_stats(scratch_rows)
    pretrain_stats = _track_stats(pretrain_rows)

    summary = {
        "run_prefix": args.run_prefix,
        "target_db5_keys": [item.strip().upper() for item in str(args.target_db5_keys).split(",") if item.strip()],
        "budget_per_class": int(args.budget_per_class),
        "budget_seed": int(args.budget_seed),
        "split_seed": int(args.split_seed),
        "rows": rows,
        "best_run": best_overall,
        "scratch_track": scratch_stats,
        "pretrain_track": pretrain_stats,
        "with_pretrain_enabled": bool(pretrain_rows),
        "comparison": {
            "pretrain_minus_scratch_mean_f1": (
                float(pretrain_stats["mean_test_macro_f1"]) - float(scratch_stats["mean_test_macro_f1"])
                if pretrain_rows
                else None
            )
        },
        "rank_rule": "test_macro_f1 desc, then test_accuracy desc",
    }

    out_dir = run_root / f"{args.run_prefix}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "event_demo_direct_summary.json"
    summary_csv = out_dir / "event_demo_direct_summary.csv"
    summary_md = out_dir / "event_demo_direct_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, rows)

    lines = [
        "# Event Direct Training Summary",
        "",
        f"- run_prefix: `{args.run_prefix}`",
        f"- target_db5_keys: `{','.join(summary['target_db5_keys'])}`",
        f"- budget_per_class: `{int(args.budget_per_class)}`",
        "",
        "## Best Run",
        f"- run_id: `{best_overall.get('run_id')}`",
        f"- stage: `{best_overall.get('stage')}`",
        f"- test_macro_f1: `{float(best_overall.get('test_macro_f1', 0.0)):.6f}`",
        f"- test_accuracy: `{float(best_overall.get('test_accuracy', 0.0)):.6f}`",
        "",
        "## Scratch Track",
        f"- count: `{scratch_stats['count']}`",
        f"- mean_test_macro_f1: `{float(scratch_stats['mean_test_macro_f1']):.6f}`",
        f"- std_test_macro_f1: `{float(scratch_stats['std_test_macro_f1']):.6f}`",
        f"- best_run_id: `{scratch_stats['best_run_id']}`",
    ]
    if pretrain_rows:
        lines.extend(
            [
                "",
                "## With Pretrain Track",
                f"- count: `{pretrain_stats['count']}`",
                f"- mean_test_macro_f1: `{float(pretrain_stats['mean_test_macro_f1']):.6f}`",
                f"- std_test_macro_f1: `{float(pretrain_stats['std_test_macro_f1']):.6f}`",
                f"- best_run_id: `{pretrain_stats['best_run_id']}`",
                f"- pretrain_minus_scratch_mean_f1: `{float(summary['comparison']['pretrain_minus_scratch_mean_f1']):.6f}`",
            ]
        )
    lines.extend(["", "## Artifacts", f"- summary_json: `{summary_json}`", f"- summary_csv: `{summary_csv}`"])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DEMO-DIRECT] summary_json={summary_json}")
    print(f"[DEMO-DIRECT] summary_csv={summary_csv}")
    print(f"[DEMO-DIRECT] summary_md={summary_md}")


if __name__ == "__main__":
    main()
