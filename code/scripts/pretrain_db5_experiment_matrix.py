"""Run a fixed DB5 pretraining experiment matrix and pick the best foundation run."""

from __future__ import annotations

import argparse
import csv
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DB5 pretraining experiment matrix (A/B/C/D).")
    parser.add_argument("--config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default="../data_ninaproDB5")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--run_prefix", default="db5_matrix")
    return parser.parse_args()


def _read_summary(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary must be a JSON object: {summary_path}")
    return payload


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
    print(f"[MATRIX] {run.name} -> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(CODE_ROOT), check=True)
    summary_path = Path(args.run_root) / run.run_id / "offline_summary.json"
    summary = _read_summary(summary_path)
    summary["matrix_name"] = run.name
    summary["matrix_run_id"] = run.run_id
    summary["overrides"] = dict(run.overrides)
    return summary


def _rank_key(summary: dict) -> tuple[float, float]:
    return (
        float(summary.get("best_val_macro_f1", 0.0) or 0.0),
        float(summary.get("best_val_acc", 0.0) or 0.0),
    )


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
        "checkpoint_path",
        "foundation_version",
        "num_classes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = _parse_args()
    prefix = str(args.run_prefix).strip() or "db5_matrix"
    runs: list[MatrixRun] = [
        MatrixRun(name="run_a_baseline", run_id=f"{prefix}_run_a", overrides={}),
        MatrixRun(
            name="run_b_signal",
            run_id=f"{prefix}_run_b",
            overrides={
                "include_rest_class": "false",
                "use_first_myo_only": "false",
                "first_myo_channel_count": "16",
                "lowcut_hz": "20",
                "highcut_hz": "180",
            },
        ),
        MatrixRun(
            name="run_c_quality",
            run_id=f"{prefix}_run_c",
            overrides={
                "include_rest_class": "false",
                "use_first_myo_only": "false",
                "first_myo_channel_count": "16",
                "lowcut_hz": "20",
                "highcut_hz": "180",
                "energy_min": "0.25",
                "static_std_min": "0.08",
                "clip_ratio_max": "0.08",
                "saturation_abs": "126",
            },
        ),
    ]
    lr_candidates = [0.0003, 0.0005, 0.0008]
    for lr in lr_candidates:
        suffix = str(lr).replace(".", "")
        runs.append(
            MatrixRun(
                name=f"run_d_capacity_lr_{lr}",
                run_id=f"{prefix}_run_d_lr_{suffix}",
                overrides={
                    "include_rest_class": "false",
                    "use_first_myo_only": "false",
                    "first_myo_channel_count": "16",
                    "lowcut_hz": "20",
                    "highcut_hz": "180",
                    "energy_min": "0.25",
                    "static_std_min": "0.08",
                    "clip_ratio_max": "0.08",
                    "saturation_abs": "126",
                    "base_channels": "32",
                    "classifier_hidden_dim": "128",
                    "epochs": "80",
                    "early_stopping_patience": "12",
                    "learning_rate": str(lr),
                },
            )
        )

    results: list[dict] = []
    for run in runs:
        results.append(_run_pretrain(args, run))

    best = max(results, key=_rank_key)
    output = {
        "run_prefix": prefix,
        "rank_rule": "best_val_macro_f1 desc, then best_val_acc desc",
        "best_run": best,
        "results": results,
    }
    matrix_dir = Path(args.run_root) / f"{prefix}_summary"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    summary_json = matrix_dir / "db5_pretrain_matrix_summary.json"
    summary_csv = matrix_dir / "db5_pretrain_matrix_summary.csv"
    summary_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv, results)
    print(
        "[MATRIX] best="
        f"{best.get('matrix_run_id')} "
        f"val_f1={best.get('best_val_macro_f1')} "
        f"val_acc={best.get('best_val_acc')}"
    )
    print(f"[MATRIX] summary_json={summary_json}")
    print(f"[MATRIX] summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
