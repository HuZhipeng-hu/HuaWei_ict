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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DB5 pretraining emergency matrix (Run-0..Run-4).")
    parser.add_argument("--config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default="../data_ninaproDB5")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--run_prefix", default="db5_sprint")
    parser.add_argument("--run4_variant", choices=sorted(RUN4_VARIANTS.keys()), default="lr5e4_wd3e4")
    parser.add_argument(
        "--run4_grid_search",
        action="store_true",
        help="Run all three Run-4 candidates instead of one recommended variant.",
    )
    parser.add_argument("--min_run3_gain", type=float, default=0.03)
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


def _run_pretrain(args: argparse.Namespace, run: MatrixRun) -> dict:
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
        "--foundation_dir",
        str(args.foundation_dir),
    ]
    for key, value in run.overrides.items():
        cmd.extend([f"--{key}", str(value)])
    cmd_str = " ".join(shlex.quote(item) for item in cmd)
    print(f"[MATRIX] {run.name} -> {cmd_str}")
    subprocess.run(cmd, cwd=str(CODE_ROOT), check=True)
    summary_path = Path(args.run_root) / run.run_id / "offline_summary.json"
    summary = _read_summary(summary_path)
    summary["matrix_name"] = run.name
    summary["matrix_run_id"] = run.run_id
    summary["matrix_command"] = cmd_str
    summary["overrides"] = dict(run.overrides)
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


def main() -> None:
    args = _parse_args()
    prefix = str(args.run_prefix).strip() or "db5_sprint"

    results: list[dict] = []
    core_runs = _build_core_runs(prefix)
    run0_summary = _run_pretrain(args, core_runs[0])
    run1_summary = _run_pretrain(args, core_runs[1])
    run2_summary = _run_pretrain(args, core_runs[2])
    run3_summary = _run_pretrain(args, core_runs[3])
    results.extend([run0_summary, run1_summary, run2_summary, run3_summary])

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
            results.append(_run_pretrain(args, run))

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
    matrix_dir = Path(args.run_root) / f"{prefix}_summary"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    summary_json = matrix_dir / "db5_pretrain_matrix_summary.json"
    summary_csv = matrix_dir / "db5_pretrain_matrix_summary.csv"
    referee_card = matrix_dir / "db5_referee_repro_card.md"
    summary_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv, results)
    referee_card.write_text(_build_referee_card(best, output, str(args.data_dir)), encoding="utf-8")
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
