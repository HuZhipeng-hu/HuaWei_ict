"""Run an emergency DB5 pretraining matrix and select the best foundation run."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


@dataclass(frozen=True)
class MatrixRun:
    name: str
    run_id: str
    overrides: dict[str, str]


RUN4_VARIANTS: dict[str, dict[str, str]] = {
    "lr3e4_wd3e4": {"learning_rate": "0.0003", "weight_decay": "0.0003"},
    "lr5e4_wd3e4": {"learning_rate": "0.0005", "weight_decay": "0.0003"},
    "lr3e4_wd1e3": {"learning_rate": "0.0003", "weight_decay": "0.001"},
}

REQUIRED_SUMMARY_FIELDS = ("best_val_epoch", "best_val_macro_f1", "best_val_acc")
REQUIRED_EVAL_FILES = ("test_metrics.json", "test_per_class_metrics.csv", "test_confusion_matrix.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DB5 pretraining emergency matrix (Run-0..Run-4).")
    parser.add_argument("--config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default="../data_ninaproDB5")
    parser.add_argument("--wearer_data_dir", default="../data")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--run_prefix", default="db5_sprint")
    parser.add_argument("--ms_mode", default="graph", choices=["graph", "pynative"])
    parser.add_argument("--auto_fallback_pynative", choices=["true", "false"], default="true")
    parser.add_argument(
        "--run_timeout_seconds",
        type=int,
        default=0,
        help="Optional timeout per run. 0 disables timeout watchdog.",
    )
    parser.add_argument("--run4_variant", choices=sorted(RUN4_VARIANTS.keys()), default="lr5e4_wd3e4")
    parser.add_argument(
        "--run4_grid_search",
        action="store_true",
        help="Run all three Run-4 candidates instead of one recommended variant.",
    )
    parser.add_argument("--min_run3_gain", type=float, default=0.03)
    parser.add_argument("--skip_quality_gate", action="store_true")
    parser.add_argument("--quality_gate_script", default="scripts/pretrain_db5_quality_gate.py")
    parser.add_argument("--quality_gate_fail_on_warning", choices=["true", "false"], default="false")
    parser.add_argument("--quality_gate_skip_db5_probe", action="store_true")
    parser.add_argument("--quality_gate_skip_budget_probe", action="store_true")
    parser.add_argument("--quality_gate_pytest_basetemp_root", default=".tmp_pytest_runcheck")
    return parser.parse_args()


def _read_summary(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary must be a JSON object: {summary_path}")
    return payload


def _rank_key(summary: dict) -> tuple[float, float]:
    return (
        float(summary.get("best_val_macro_f1", 0.0) or 0.0),
        float(summary.get("best_val_acc", 0.0) or 0.0),
    )


def _should_skip_run4(*, run0_val_f1: float, run3_val_f1: float, min_gain: float = 0.03) -> bool:
    return float(run3_val_f1) - float(run0_val_f1) < float(min_gain)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return CODE_ROOT / path


def _missing_summary_fields(summary: dict) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_SUMMARY_FIELDS:
        value = summary.get(field, None)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
    return missing


def _validate_run_artifacts(args: argparse.Namespace, run: MatrixRun, summary: dict) -> list[str]:
    issues: list[str] = []
    missing_fields = _missing_summary_fields(summary)
    if missing_fields:
        issues.append(f"offline_summary missing required fields: {', '.join(missing_fields)}")

    run_dir = Path(args.run_root) / run.run_id
    summary_path = run_dir / "offline_summary.json"
    if not summary_path.exists():
        issues.append(f"missing artifact: {summary_path}")

    checkpoint = str(summary.get("checkpoint_path", "")).strip()
    if not checkpoint:
        issues.append("offline_summary.checkpoint_path is empty")
    else:
        checkpoint_path = _resolve_path(checkpoint)
        if not checkpoint_path.exists():
            issues.append(f"checkpoint does not exist: {checkpoint_path}")

    eval_dir = run_dir / "evaluation"
    if not eval_dir.exists():
        issues.append(f"missing evaluation directory: {eval_dir}")
    else:
        for filename in REQUIRED_EVAL_FILES:
            path = eval_dir / filename
            if not path.exists():
                issues.append(f"missing evaluation artifact: {path}")

    run_card = run_dir / "referee_repro_card.md"
    if not run_card.exists():
        issues.append(f"missing run referee card: {run_card}")

    return issues


def _build_pretrain_command(args: argparse.Namespace, run: MatrixRun, *, ms_mode: str | None = None) -> list[str]:
    selected_mode = str(ms_mode or args.ms_mode)
    cmd: list[str] = [
        sys.executable,
        "scripts/pretrain_ninapro_db5.py",
        "--config",
        str(args.config),
        "--data_dir",
        str(args.data_dir),
        "--run_root",
        str(args.run_root),
        "--run_id",
        run.run_id,
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--ms_mode",
        selected_mode,
        "--foundation_dir",
        str(args.foundation_dir),
    ]
    for key, value in run.overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _run_pretrain(args: argparse.Namespace, run: MatrixRun) -> dict:
    cmd = _build_pretrain_command(args, run, ms_mode=str(args.ms_mode))
    cmd_str = " ".join(shlex.quote(item) for item in cmd)
    print(f"[MATRIX] {run.name} -> {cmd_str}")
    timeout = int(args.run_timeout_seconds) if int(args.run_timeout_seconds) > 0 else None
    fallback_enabled = bool(str(args.auto_fallback_pynative).strip().lower() == "true")
    fallback_used = False
    fallback_cause = ""
    try:
        completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        completed = None
        fallback_cause = f"timeout({exc.timeout}s)"

    if completed is not None and completed.returncode == 0:
        pass
    else:
        can_fallback = str(args.ms_mode).lower() == "graph" and fallback_enabled
        if not can_fallback:
            if completed is None:
                raise RuntimeError(f"training subprocess timed out: run_id={run.run_id}, cmd={cmd_str}")
            raise RuntimeError(
                f"training subprocess failed: run_id={run.run_id}, rc={completed.returncode}, cmd={cmd_str}"
            )

        fallback_cmd = _build_pretrain_command(args, run, ms_mode="pynative")
        fallback_cmd_str = " ".join(shlex.quote(item) for item in fallback_cmd)
        if completed is None:
            print(f"[MATRIX] {run.name} graph fallback -> {fallback_cmd_str} (reason={fallback_cause})")
        else:
            fallback_cause = f"non_zero_rc({completed.returncode})"
            print(f"[MATRIX] {run.name} graph fallback -> {fallback_cmd_str} (reason={fallback_cause})")
        completed_fb = subprocess.run(fallback_cmd, cwd=str(CODE_ROOT), check=False, timeout=timeout)
        if completed_fb.returncode != 0:
            raise RuntimeError(
                f"training subprocess failed after fallback: run_id={run.run_id}, rc={completed_fb.returncode}, "
                f"graph_cmd={cmd_str}, fallback_cmd={fallback_cmd_str}"
            )
        fallback_used = True
        cmd_str = fallback_cmd_str

    summary_path = Path(args.run_root) / run.run_id / "offline_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing offline_summary after run: {summary_path}")
    summary = _read_summary(summary_path)
    issues = _validate_run_artifacts(args, run, summary)
    if issues:
        raise RuntimeError("; ".join(issues))
    summary["matrix_name"] = run.name
    summary["matrix_run_id"] = run.run_id
    summary["matrix_command"] = cmd_str
    summary["overrides"] = dict(run.overrides)
    summary["auto_fallback_used"] = bool(fallback_used)
    summary["auto_fallback_cause"] = str(fallback_cause)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix_name",
        "matrix_run_id",
        "best_val_epoch",
        "best_val_macro_f1",
        "best_val_acc",
        "test_macro_f1",
        "test_accuracy",
        "test_macro_recall",
        "learning_rate",
        "weight_decay",
        "checkpoint_path",
        "foundation_version",
        "num_classes",
        "top_confusion_pair",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _build_referee_card(best: dict, matrix_output: dict, data_dir: str) -> str:
    return "\n".join(
        [
            "# DB5 预训练评委复现实验卡",
            "",
            "## 一键复现（推荐）",
            "```bash",
            str(best.get("matrix_command", "")).strip(),
            "```",
            "",
            "## 最佳运行结果",
            f"- matrix_run_id: `{best.get('matrix_run_id', '')}`",
            f"- checkpoint: `{best.get('checkpoint_path', '')}`",
            f"- best_val_epoch: `{best.get('best_val_epoch', '')}`",
            f"- best_val_macro_f1: `{best.get('best_val_macro_f1', '')}`",
            f"- best_val_acc: `{best.get('best_val_acc', '')}`",
            f"- top_confusion_pair: `{best.get('top_confusion_pair', '')}`",
            f"- split_seed: `{best.get('split_seed', '')}`",
            "",
            "## 数据与约束",
            f"- 数据路径: `{data_dir}`",
            "- 无需个人校准数据。",
            "- 仅依赖仓库脚本、配置与 DB5 数据可复现。",
            "",
            "## 矩阵规则",
            f"- 排序: `{matrix_output.get('rank_rule', '')}`",
            f"- Run-3 相对 Run-0 最低提升阈值: `{matrix_output.get('min_run3_gain', '')}`",
            f"- Run-4 是否跳过: `{matrix_output.get('run4_skipped', False)}`",
            "",
        ]
    )


def _build_core_runs(prefix: str) -> list[MatrixRun]:
    baseline_common = {
        "include_rest_class": "false",
        "use_first_myo_only": "false",
        "first_myo_channel_count": "16",
        "lowcut_hz": "20",
        "highcut_hz": "180",
        "energy_min": "0.25",
        "static_std_min": "0.08",
        "clip_ratio_max": "0.08",
        "saturation_abs": "126",
        "use_adaptive_action_thresholds": "false",
        "max_windows_per_segment": "6",
        "epochs": "80",
        "early_stopping_patience": "12",
        "loss_type": "cb_focal",
        "label_smoothing": "0.05",
        "hard_mining_ratio": "0.2",
        "ema_enabled": "false",
        "learning_rate": "0.0005",
        "weight_decay": "0.0003",
    }
    run0 = MatrixRun(
        name="run_0_baseline",
        run_id=f"{prefix}_run_0",
        overrides={**baseline_common, "manifest_use_source_metadata": "false"},
    )
    run1 = MatrixRun(
        name="run_1_split",
        run_id=f"{prefix}_run_1",
        overrides={**baseline_common, "manifest_use_source_metadata": "true"},
    )
    run2 = MatrixRun(
        name="run_2_quality",
        run_id=f"{prefix}_run_2",
        overrides={
            **baseline_common,
            "manifest_use_source_metadata": "true",
            "clip_ratio_max": "0.12",
            "use_adaptive_action_thresholds": "true",
            "action_quantile_percent": "30",
            "max_windows_per_segment": "10",
        },
    )
    run3 = MatrixRun(
        name="run_3_stability",
        run_id=f"{prefix}_run_3",
        overrides={
            **baseline_common,
            "manifest_use_source_metadata": "true",
            "clip_ratio_max": "0.12",
            "use_adaptive_action_thresholds": "true",
            "action_quantile_percent": "30",
            "max_windows_per_segment": "10",
            "epochs": "100",
            "early_stopping_patience": "15",
            "loss_type": "cross_entropy",
            "label_smoothing": "0.1",
            "hard_mining_ratio": "0.0",
            "ema_enabled": "true",
            "ema_decay": "0.999",
        },
    )
    return [run0, run1, run2, run3]


def _build_run4_candidates(prefix: str, variant: str, *, run_all: bool) -> list[MatrixRun]:
    names = sorted(RUN4_VARIANTS.keys()) if run_all else [variant]
    runs: list[MatrixRun] = []
    for item in names:
        pair = RUN4_VARIANTS[item]
        runs.append(
            MatrixRun(
                name=f"run_4_{item}",
                run_id=f"{prefix}_run_4_{item}",
                overrides={
                    "manifest_use_source_metadata": "true",
                    "include_rest_class": "false",
                    "use_first_myo_only": "false",
                    "first_myo_channel_count": "16",
                    "lowcut_hz": "20",
                    "highcut_hz": "180",
                    "energy_min": "0.25",
                    "static_std_min": "0.08",
                    "clip_ratio_max": "0.12",
                    "saturation_abs": "126",
                    "use_adaptive_action_thresholds": "true",
                    "action_quantile_percent": "30",
                    "max_windows_per_segment": "10",
                    "epochs": "100",
                    "early_stopping_patience": "15",
                    "loss_type": "cross_entropy",
                    "label_smoothing": "0.1",
                    "hard_mining_ratio": "0.0",
                    "ema_enabled": "true",
                    "ema_decay": "0.999",
                    "learning_rate": str(pair["learning_rate"]),
                    "weight_decay": str(pair["weight_decay"]),
                },
            )
        )
    return runs


def _write_failure_report(matrix_dir: Path, failures: list[dict]) -> None:
    report_json = matrix_dir / "db5_pretrain_failure_report.json"
    report_md = matrix_dir / "db5_pretrain_failure_report.md"
    payload = {"failed": bool(failures), "failure_count": len(failures), "failures": failures}
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DB5 Pretrain Failure Report",
        "",
        f"- failed: `{payload['failed']}`",
        f"- failure_count: `{payload['failure_count']}`",
        "",
    ]
    if failures:
        lines.extend(
            [
                "| stage | matrix_run_id | root_cause | next_command |",
                "|---|---|---|---|",
            ]
        )
        for item in failures:
            lines.append(
                "| "
                f"{item.get('stage', '')} | {item.get('matrix_run_id', '')} | "
                f"{item.get('root_cause', '')} | `{item.get('next_command', '')}` |"
            )
    else:
        lines.append("No failures.")
    report_md.write_text("\n".join(lines), encoding="utf-8")


def _run_quality_gate(args: argparse.Namespace, prefix: str) -> None:
    gate_prefix = f"{prefix}_gate"
    cmd: list[str] = [
        sys.executable,
        str(args.quality_gate_script),
        "--config",
        str(args.config),
        "--data_dir",
        str(args.data_dir),
        "--wearer_data_dir",
        str(args.wearer_data_dir),
        "--run_root",
        str(args.run_root),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--foundation_dir",
        str(args.foundation_dir),
        "--run_prefix",
        gate_prefix,
        "--fail_on_warning",
        str(args.quality_gate_fail_on_warning),
        "--pytest_basetemp_root",
        str(args.quality_gate_pytest_basetemp_root),
    ]
    if args.quality_gate_skip_db5_probe:
        cmd.append("--skip_db5_probe")
    if args.quality_gate_skip_budget_probe:
        cmd.append("--skip_budget_probe")

    cmd_str = " ".join(shlex.quote(item) for item in cmd)
    print(f"[MATRIX] quality_gate -> {cmd_str}")
    subprocess.run(cmd, cwd=str(CODE_ROOT), check=True)


def main() -> None:
    args = _parse_args()
    prefix = str(args.run_prefix).strip() or "db5_sprint"
    matrix_dir = Path(args.run_root) / f"{prefix}_summary"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures: list[dict] = []

    def run_or_fail(run: MatrixRun) -> dict:
        cmd = _build_pretrain_command(args, run)
        cmd_str = " ".join(shlex.quote(item) for item in cmd)
        try:
            summary = _run_pretrain(args, run)
            results.append(summary)
            return summary
        except Exception as exc:
            failure = {
                "stage": run.name,
                "matrix_run_id": run.run_id,
                "root_cause": str(exc),
                "next_command": cmd_str,
            }
            failures.append(failure)
            _write_failure_report(matrix_dir, failures)
            raise

    if not args.skip_quality_gate:
        try:
            _run_quality_gate(args, prefix)
        except Exception as exc:
            failures.append(
                {
                    "stage": "quality_gate",
                    "matrix_run_id": f"{prefix}_gate",
                    "root_cause": str(exc),
                    "next_command": (
                        f"{sys.executable} {args.quality_gate_script} "
                        f"--config {args.config} --data_dir {args.data_dir}"
                    ),
                }
            )
            _write_failure_report(matrix_dir, failures)
            raise

    core_runs = _build_core_runs(prefix)
    run0_summary = run_or_fail(core_runs[0])
    run1_summary = run_or_fail(core_runs[1])
    run2_summary = run_or_fail(core_runs[2])
    run3_summary = run_or_fail(core_runs[3])

    run3_gain = float(run3_summary.get("best_val_macro_f1", 0.0) or 0.0) - float(
        run0_summary.get("best_val_macro_f1", 0.0) or 0.0
    )
    run4_skipped = _should_skip_run4(
        run0_val_f1=float(run0_summary.get("best_val_macro_f1", 0.0) or 0.0),
        run3_val_f1=float(run3_summary.get("best_val_macro_f1", 0.0) or 0.0),
        min_gain=float(args.min_run3_gain),
    )
    skip_reason = ""
    if run4_skipped:
        skip_reason = (
            "Run-3 gain below threshold: "
            f"run3-run0={run3_gain:.4f} < required={float(args.min_run3_gain):.4f}. "
            "Stop blind search and move to task-definition branch."
        )
        print(f"[MATRIX] {skip_reason}")
    else:
        run4_candidates = _build_run4_candidates(prefix, args.run4_variant, run_all=bool(args.run4_grid_search))
        for run in run4_candidates:
            run_or_fail(run)

    best = max(results, key=_rank_key)
    output = {
        "run_prefix": prefix,
        "rank_rule": "best_val_macro_f1 desc, then best_val_acc desc",
        "min_run3_gain": float(args.min_run3_gain),
        "run3_gain": float(run3_gain),
        "run4_skipped": bool(run4_skipped),
        "skip_reason": skip_reason,
        "best_run": best,
        "results": results,
    }
    summary_json = matrix_dir / "db5_pretrain_matrix_summary.json"
    summary_csv = matrix_dir / "db5_pretrain_matrix_summary.csv"
    referee_card = matrix_dir / "db5_referee_repro_card.md"
    summary_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv, results)
    referee_card.write_text(_build_referee_card(best, output, str(args.data_dir)), encoding="utf-8")
    if not referee_card.exists():
        failures.append(
            {
                "stage": "matrix_referee_card",
                "matrix_run_id": str(best.get("matrix_run_id", "")),
                "root_cause": f"missing matrix referee card: {referee_card}",
                "next_command": f"re-run matrix script: {Path(__file__).name}",
            }
        )
        _write_failure_report(matrix_dir, failures)
        raise RuntimeError(f"missing matrix referee card: {referee_card}")

    _write_failure_report(matrix_dir, failures)
    print(
        "[MATRIX] best="
        f"{best.get('matrix_run_id')} "
        f"val_f1={best.get('best_val_macro_f1')} "
        f"val_acc={best.get('best_val_acc')}"
    )
    print(f"[MATRIX] summary_json={summary_json}")
    print(f"[MATRIX] summary_csv={summary_csv}")
    print(f"[MATRIX] referee_card={referee_card}")


if __name__ == "__main__":
    main()
