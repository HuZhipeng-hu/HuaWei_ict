"""Evaluate event-onset few-shot transfer curve and incremental action extension."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.head_expansion import normalize_action_keys
from shared.run_utils import dump_json, ensure_run_dir


@dataclass(frozen=True)
class FewshotRunResult:
    budget: int
    seed: int
    run_id: str
    test_macro_f1: float
    test_accuracy: float
    checkpoint_path: str
    top_confusion_pair: str
    command: str


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"Expected a comma-separated integer list, got: {raw!r}")
    return values


def _parse_action_keys(raw: str) -> list[str]:
    keys = normalize_action_keys(raw)
    if not keys:
        raise ValueError("target_db5_keys is empty after parsing.")
    return keys


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _append_recordings_manifest_arg(cmd: list[str], recordings_manifest: str | None) -> list[str]:
    raw = str(recordings_manifest or "").strip()
    if not raw:
        return cmd
    return [*cmd, "--recordings_manifest", raw]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _run_finetune_and_collect(cmd: list[str], *, run_root: Path, run_id: str) -> dict:
    cmd_str = _format_cmd(cmd)
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"finetune subprocess failed: rc={completed.returncode}, cmd={cmd_str}")
    summary_path = run_root / run_id / "offline_summary.json"
    summary = _load_json(summary_path)
    required = ["test_macro_f1", "test_accuracy", "checkpoint_path"]
    missing = [key for key in required if key not in summary]
    if missing:
        raise RuntimeError(f"offline_summary missing required fields: {', '.join(missing)} ({summary_path})")
    return {"summary": summary, "command": cmd_str}


def _aggregate_budget_rows(rows: list[FewshotRunResult]) -> list[dict]:
    grouped: dict[int, list[FewshotRunResult]] = {}
    for row in rows:
        grouped.setdefault(int(row.budget), []).append(row)
    out: list[dict] = []
    for budget in sorted(grouped.keys()):
        items = grouped[budget]
        f1 = np.asarray([float(x.test_macro_f1) for x in items], dtype=np.float64)
        acc = np.asarray([float(x.test_accuracy) for x in items], dtype=np.float64)
        out.append(
            {
                "budget": int(budget),
                "runs": int(len(items)),
                "macro_f1_mean": float(np.mean(f1)),
                "macro_f1_std": float(np.std(f1)),
                "acc_mean": float(np.mean(acc)),
                "acc_std": float(np.std(acc)),
            }
        )
    return out


def _choose_best_budget(agg_rows: list[dict]) -> dict:
    if not agg_rows:
        raise RuntimeError("No few-shot rows to rank.")
    return max(
        agg_rows,
        key=lambda row: (
            float(row.get("macro_f1_mean", 0.0)),
            float(row.get("acc_mean", 0.0)),
            -float(row.get("macro_f1_std", 0.0)),
        ),
    )


def _choose_elbow_budget(agg_rows: list[dict], *, tolerance: float) -> int:
    if not agg_rows:
        raise RuntimeError("No few-shot rows to compute elbow.")
    best_f1 = max(float(item.get("macro_f1_mean", 0.0)) for item in agg_rows)
    for item in sorted(agg_rows, key=lambda row: int(row["budget"])):
        if float(item.get("macro_f1_mean", 0.0)) >= best_f1 - float(tolerance):
            return int(item["budget"])
    return int(sorted(agg_rows, key=lambda row: int(row["budget"]))[-1]["budget"])


def _generate_budget_curve_mermaid(path: Path, rows: list[dict]) -> None:
    x_values = ", ".join(str(int(item["budget"])) for item in rows)
    f1_values = ", ".join(f"{float(item['macro_f1_mean']):.4f}" for item in rows)
    acc_values = ", ".join(f"{float(item['acc_mean']):.4f}" for item in rows)
    content = "\n".join(
        [
            "xychart-beta",
            '  title "Few-shot Budget Performance Curve"',
            '  x-axis "clips/action" [' + x_values + "]",
            '  y-axis "score" 0 --> 1',
            '  line "macro_f1_mean" [' + f1_values + "]",
            '  line "acc_mean" [' + acc_values + "]",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_incremental_mermaid(path: Path, *, base_f1: float, expanded_f1: float, base_acc: float, expanded_acc: float) -> None:
    content = "\n".join(
        [
            "xychart-beta",
            '  title "Incremental Action Delta"',
            '  x-axis "protocol" ["base", "expanded"]',
            '  y-axis "score" 0 --> 1',
            '  bar "macro_f1_mean" [' + f"{float(base_f1):.4f}, {float(expanded_f1):.4f}" + "]",
            '  bar "acc_mean" [' + f"{float(base_acc):.4f}, {float(expanded_acc):.4f}" + "]",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _try_generate_budget_plot_png(path: Path, rows: list[dict]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib_not_available"

    budgets = [int(item["budget"]) for item in rows]
    f1_mean = [float(item["macro_f1_mean"]) for item in rows]
    f1_std = [float(item["macro_f1_std"]) for item in rows]
    acc_mean = [float(item["acc_mean"]) for item in rows]
    acc_std = [float(item["acc_std"]) for item in rows]

    fig = plt.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111)
    ax.errorbar(budgets, f1_mean, yerr=f1_std, marker="o", label="macro_f1_mean")
    ax.errorbar(budgets, acc_mean, yerr=acc_std, marker="s", label="acc_mean")
    ax.set_xlabel("clips/action")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Few-shot Budget Performance Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return "ok"


def _try_generate_incremental_plot_png(path: Path, *, base_f1: float, expanded_f1: float, base_acc: float, expanded_acc: float) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib_not_available"

    fig = plt.figure(figsize=(6, 4), dpi=120)
    ax = fig.add_subplot(111)
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [base_f1, expanded_f1], width=width, label="macro_f1_mean")
    ax.bar(x + width / 2, [base_acc, expanded_acc], width=width, label="acc_mean")
    ax.set_xticks(x)
    ax.set_xticklabels(["base", "expanded"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Incremental Action Delta")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return "ok"


def _write_failure_report(report_dir: Path, *, stage: str, root_cause: str, next_command: str) -> None:
    payload = {
        "status": "failed",
        "stage": str(stage),
        "root_cause": str(root_cause),
        "next_command": str(next_command),
        "generated_at_unix": int(time.time()),
    }
    dump_json(report_dir / "fewshot_failure_report.json", payload)
    markdown = "\n".join(
        [
            "# Few-shot Failure Report",
            "",
            f"- stage: `{stage}`",
            f"- root_cause: `{root_cause}`",
            f"- next_command: `{next_command}`",
            "",
        ]
    )
    (report_dir / "fewshot_failure_report.md").write_text(markdown, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run event few-shot curve and incremental expansion evaluation.")
    parser.add_argument("--config", default="configs/training_event_onset.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--pretrained_emg_checkpoint", default=None)
    parser.add_argument("--target_db5_keys", default="E1_G01,E1_G02,E1_G03,E1_G04")
    parser.add_argument("--budgets", default="10,20,35,60")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--elbow_tolerance", type=float, default=0.02)
    parser.add_argument("--run_incremental_delta", choices=["true", "false"], default="true")
    parser.add_argument("--incremental_base_keys", default=None)
    parser.add_argument("--incremental_budget", type=int, default=35)
    parser.add_argument("--incremental_head_only", choices=["true", "false"], default="true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="event_fewshot_curve")
    run_root = Path(args.run_root)
    report_dir = Path(run_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    budgets = _parse_int_list(args.budgets)
    seeds = _parse_int_list(args.seeds)
    target_keys = _parse_action_keys(args.target_db5_keys)
    shared_manifest = report_dir / "manifests" / "fewshot_shared_split_manifest.json"
    shared_manifest.parent.mkdir(parents=True, exist_ok=True)

    run_rows: list[FewshotRunResult] = []
    matrix_commands: list[str] = []
    stage = "fewshot_curve"
    try:
        for budget in budgets:
            for seed in seeds:
                matrix_run_id = f"{run_id}_fewshot_b{int(budget)}_s{int(seed)}"
                cmd = [
                    sys.executable,
                    "scripts/finetune_event_onset.py",
                    "--config",
                    str(args.config),
                    "--data_dir",
                    str(args.data_dir),
                    "--device_target",
                    str(args.device_target),
                    "--device_id",
                    str(int(args.device_id)),
                    "--run_root",
                    str(args.run_root),
                    "--run_id",
                    matrix_run_id,
                    "--target_db5_keys",
                    ",".join(target_keys),
                    "--budget_per_class",
                    str(int(budget)),
                    "--budget_seed",
                    str(int(seed)),
                    "--split_seed",
                    str(int(args.split_seed)),
                    "--manifest_strategy",
                    "v2",
                ]
                if str(args.pretrained_emg_checkpoint or "").strip():
                    cmd.extend(["--pretrained_emg_checkpoint", str(args.pretrained_emg_checkpoint)])
                if shared_manifest.exists():
                    cmd.extend(["--split_manifest_in", str(shared_manifest)])
                else:
                    cmd.extend(["--split_manifest_out", str(shared_manifest)])
                cmd = _append_recordings_manifest_arg(cmd, args.recordings_manifest)

                result = _run_finetune_and_collect(cmd, run_root=run_root, run_id=matrix_run_id)
                matrix_commands.append(str(result["command"]))
                summary = dict(result["summary"])
                run_rows.append(
                    FewshotRunResult(
                        budget=int(budget),
                        seed=int(seed),
                        run_id=matrix_run_id,
                        test_macro_f1=float(summary["test_macro_f1"]),
                        test_accuracy=float(summary["test_accuracy"]),
                        checkpoint_path=str(summary["checkpoint_path"]),
                        top_confusion_pair=str(summary.get("top_confusion_pair", "")),
                        command=str(result["command"]),
                    )
                )
    except Exception as exc:
        last_cmd = matrix_commands[-1] if matrix_commands else "python scripts/finetune_event_onset.py <args>"
        _write_failure_report(report_dir, stage=stage, root_cause=str(exc), next_command=last_cmd)
        raise

    agg_rows = _aggregate_budget_rows(run_rows)
    best_budget_row = _choose_best_budget(agg_rows)
    elbow_budget = _choose_elbow_budget(agg_rows, tolerance=float(args.elbow_tolerance))

    per_run_rows = [
        {
            "budget": row.budget,
            "seed": row.seed,
            "run_id": row.run_id,
            "test_macro_f1": row.test_macro_f1,
            "test_accuracy": row.test_accuracy,
            "checkpoint_path": row.checkpoint_path,
            "top_confusion_pair": row.top_confusion_pair,
            "command": row.command,
        }
        for row in run_rows
    ]
    _write_csv(
        report_dir / "fewshot_curve_runs.csv",
        per_run_rows,
        fieldnames=[
            "budget",
            "seed",
            "run_id",
            "test_macro_f1",
            "test_accuracy",
            "checkpoint_path",
            "top_confusion_pair",
            "command",
        ],
    )
    _write_csv(
        report_dir / "fewshot_curve_summary.csv",
        agg_rows,
        fieldnames=["budget", "runs", "macro_f1_mean", "macro_f1_std", "acc_mean", "acc_std"],
    )

    curve_mermaid = report_dir / "charts" / "fewshot_budget_curve.mmd"
    _generate_budget_curve_mermaid(curve_mermaid, agg_rows)
    curve_png = report_dir / "charts" / "fewshot_budget_curve.png"
    curve_png_status = _try_generate_budget_plot_png(curve_png, agg_rows)

    incremental_payload: dict = {
        "enabled": bool(str(args.run_incremental_delta).strip().lower() == "true"),
        "status": "skipped",
    }
    if incremental_payload["enabled"]:
        stage = "incremental_delta"
        try:
            base_keys = (
                _parse_action_keys(args.incremental_base_keys)
                if args.incremental_base_keys
                else list(target_keys[:-1] if len(target_keys) > 1 else target_keys)
            )
            if not base_keys:
                raise ValueError("incremental_base_keys resolved to empty list.")

            base_rows: list[dict] = []
            expanded_rows: list[dict] = []
            for seed in seeds:
                base_run_id = f"{run_id}_inc_base_s{int(seed)}"
                base_cmd = [
                    sys.executable,
                    "scripts/finetune_event_onset.py",
                    "--config",
                    str(args.config),
                    "--data_dir",
                    str(args.data_dir),
                    "--device_target",
                    str(args.device_target),
                    "--device_id",
                    str(int(args.device_id)),
                    "--run_root",
                    str(args.run_root),
                    "--run_id",
                    base_run_id,
                    "--target_db5_keys",
                    ",".join(base_keys),
                    "--budget_per_class",
                    str(int(args.incremental_budget)),
                    "--budget_seed",
                    str(int(seed)),
                    "--split_seed",
                    str(int(args.split_seed)),
                    "--manifest_strategy",
                    "v2",
                ]
                if str(args.pretrained_emg_checkpoint or "").strip():
                    base_cmd.extend(["--pretrained_emg_checkpoint", str(args.pretrained_emg_checkpoint)])
                base_cmd = _append_recordings_manifest_arg(base_cmd, args.recordings_manifest)
                base_ret = _run_finetune_and_collect(base_cmd, run_root=run_root, run_id=base_run_id)
                base_summary = dict(base_ret["summary"])
                base_rows.append(
                    {
                        "seed": int(seed),
                        "run_id": base_run_id,
                        "test_macro_f1": float(base_summary["test_macro_f1"]),
                        "test_accuracy": float(base_summary["test_accuracy"]),
                        "checkpoint_path": str(base_summary["checkpoint_path"]),
                        "command": str(base_ret["command"]),
                    }
                )

                expanded_run_id = f"{run_id}_inc_expanded_s{int(seed)}"
                expanded_cmd = [
                    sys.executable,
                    "scripts/finetune_event_onset.py",
                    "--config",
                    str(args.config),
                    "--data_dir",
                    str(args.data_dir),
                    "--device_target",
                    str(args.device_target),
                    "--device_id",
                    str(int(args.device_id)),
                    "--run_root",
                    str(args.run_root),
                    "--run_id",
                    expanded_run_id,
                    "--target_db5_keys",
                    ",".join(target_keys),
                    "--budget_per_class",
                    str(int(args.incremental_budget)),
                    "--budget_seed",
                    str(int(seed)),
                    "--split_seed",
                    str(int(args.split_seed)),
                    "--manifest_strategy",
                    "v2",
                    "--incremental_from_checkpoint",
                    str(base_summary["checkpoint_path"]),
                    "--incremental_old_target_db5_keys",
                    ",".join(base_keys),
                    "--incremental_head_only",
                    str(args.incremental_head_only),
                ]
                if str(args.pretrained_emg_checkpoint or "").strip():
                    expanded_cmd.extend(["--pretrained_emg_checkpoint", str(args.pretrained_emg_checkpoint)])
                expanded_cmd = _append_recordings_manifest_arg(expanded_cmd, args.recordings_manifest)
                expanded_ret = _run_finetune_and_collect(expanded_cmd, run_root=run_root, run_id=expanded_run_id)
                expanded_summary = dict(expanded_ret["summary"])
                expanded_rows.append(
                    {
                        "seed": int(seed),
                        "run_id": expanded_run_id,
                        "test_macro_f1": float(expanded_summary["test_macro_f1"]),
                        "test_accuracy": float(expanded_summary["test_accuracy"]),
                        "checkpoint_path": str(expanded_summary["checkpoint_path"]),
                        "command": str(expanded_ret["command"]),
                    }
                )

            base_f1 = float(np.mean([item["test_macro_f1"] for item in base_rows]))
            exp_f1 = float(np.mean([item["test_macro_f1"] for item in expanded_rows]))
            base_acc = float(np.mean([item["test_accuracy"] for item in base_rows]))
            exp_acc = float(np.mean([item["test_accuracy"] for item in expanded_rows]))

            incremental_csv = report_dir / "incremental_action_delta.csv"
            _write_csv(
                incremental_csv,
                [
                    {"protocol": "base", "macro_f1_mean": base_f1, "acc_mean": base_acc},
                    {"protocol": "expanded", "macro_f1_mean": exp_f1, "acc_mean": exp_acc},
                ],
                fieldnames=["protocol", "macro_f1_mean", "acc_mean"],
            )

            inc_mermaid = report_dir / "charts" / "incremental_action_delta.mmd"
            _generate_incremental_mermaid(
                inc_mermaid,
                base_f1=base_f1,
                expanded_f1=exp_f1,
                base_acc=base_acc,
                expanded_acc=exp_acc,
            )
            inc_png = report_dir / "charts" / "incremental_action_delta.png"
            inc_png_status = _try_generate_incremental_plot_png(
                inc_png,
                base_f1=base_f1,
                expanded_f1=exp_f1,
                base_acc=base_acc,
                expanded_acc=exp_acc,
            )

            incremental_payload = {
                "enabled": True,
                "status": "ok",
                "base_keys": base_keys,
                "expanded_keys": target_keys,
                "budget": int(args.incremental_budget),
                "base_macro_f1_mean": base_f1,
                "expanded_macro_f1_mean": exp_f1,
                "delta_macro_f1": float(exp_f1 - base_f1),
                "base_acc_mean": base_acc,
                "expanded_acc_mean": exp_acc,
                "delta_acc": float(exp_acc - base_acc),
                "base_rows": base_rows,
                "expanded_rows": expanded_rows,
                "chart_mermaid": str(inc_mermaid),
                "chart_png": str(inc_png) if inc_png.exists() else "",
                "chart_png_status": inc_png_status,
            }
        except Exception as exc:
            _write_failure_report(
                report_dir,
                stage=stage,
                root_cause=str(exc),
                next_command="python scripts/finetune_event_onset.py --incremental_from_checkpoint <base_ckpt> ...",
            )
            raise

    report_payload = {
        "run_id": run_id,
        "recordings_manifest": str(args.recordings_manifest or ""),
        "target_db5_keys": target_keys,
        "budgets": budgets,
        "seeds": seeds,
        "rank_rule": "fewshot macro_f1 mean desc, then acc mean desc, then macro_f1 std asc",
        "best_budget": int(best_budget_row["budget"]),
        "best_budget_row": best_budget_row,
        "elbow_budget": int(elbow_budget),
        "curve_runs": per_run_rows,
        "curve_summary": agg_rows,
        "shared_split_manifest": str(shared_manifest),
        "curve_chart_mermaid": str(curve_mermaid),
        "curve_chart_png": str(curve_png) if curve_png.exists() else "",
        "curve_chart_png_status": curve_png_status,
        "incremental_delta": incremental_payload,
    }

    dump_json(report_dir / "downstream_fewshot_report.json", report_payload)
    card_lines = [
        "# Event Few-shot Repro Card",
        "",
        "## Protocol",
        f"- target_db5_keys: `{','.join(target_keys)}`",
        f"- budgets(clips/action): `{','.join(str(x) for x in budgets)}`",
        f"- seeds: `{','.join(str(x) for x in seeds)}`",
        f"- split_seed: `{int(args.split_seed)}`",
        "- no personal calibration data required",
        "",
        "## Result",
        f"- best_budget: `{int(best_budget_row['budget'])}`",
        f"- best_macro_f1_mean: `{float(best_budget_row['macro_f1_mean']):.4f}`",
        f"- best_acc_mean: `{float(best_budget_row['acc_mean']):.4f}`",
        f"- elbow_budget: `{int(elbow_budget)}`",
        "",
        "## Artifacts",
        f"- report_json: `{report_dir / 'downstream_fewshot_report.json'}`",
        f"- curve_csv: `{report_dir / 'fewshot_curve_summary.csv'}`",
        f"- curve_chart: `{curve_mermaid}`",
    ]
    (report_dir / "referee_repro_card.md").write_text("\n".join(card_lines), encoding="utf-8")
    print(f"[FEWSHOT] report={report_dir / 'downstream_fewshot_report.json'}")
    print(f"[FEWSHOT] card={report_dir / 'referee_repro_card.md'}")


if __name__ == "__main__":
    main()
