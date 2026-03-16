"""Run a fixed 6-run quick sweep for two-stage algorithm backend."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 6-run quick sweep for algo v2")
    parser.add_argument("--config", default="configs/training_event_onset_demo_p0.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default="s2_train_manifest_event_clean.csv")
    parser.add_argument("--split_manifest", default="artifacts/splits/s2_stable4_relax12_seed42.json")
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="s2_algo_v2_quick")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--rule_confidence", type=float, default=0.94)
    return parser


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload is not object: {path}")
    return payload


def _rank_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row.get("test_command_success_rate", 0.0)),
        -float(row.get("test_false_trigger_rate", 1.0)),
        -float(row.get("test_false_release_rate", 1.0)),
        float(row.get("event_action_accuracy", 0.0)),
        float(row.get("event_action_macro_f1", 0.0)),
    )


def _build_presets() -> list[dict]:
    return [
        {
            "preset": "baseline_v1",
            "algo_mode": "v1_single",
            "gate_action_threshold": 0.55,
            "gate_margin_threshold": 0.05,
            "wrist_rule_min_delta": 0.00,
            "wrist_rule_margin_delta": 0.00,
            "release_emg_min_delta": 0.00,
            "release_imu_max_delta": 0.00,
        },
        {
            "preset": "v2_default",
            "algo_mode": "v2_two_stage",
            "gate_action_threshold": 0.55,
            "gate_margin_threshold": 0.05,
            "wrist_rule_min_delta": 0.00,
            "wrist_rule_margin_delta": 0.00,
            "release_emg_min_delta": 0.00,
            "release_imu_max_delta": 0.00,
        },
        {
            "preset": "v2_loose_gate",
            "algo_mode": "v2_two_stage",
            "gate_action_threshold": 0.50,
            "gate_margin_threshold": 0.02,
            "wrist_rule_min_delta": 0.00,
            "wrist_rule_margin_delta": 0.00,
            "release_emg_min_delta": 0.00,
            "release_imu_max_delta": 0.00,
        },
        {
            "preset": "v2_strict_gate",
            "algo_mode": "v2_two_stage",
            "gate_action_threshold": 0.62,
            "gate_margin_threshold": 0.10,
            "wrist_rule_min_delta": 0.00,
            "wrist_rule_margin_delta": 0.00,
            "release_emg_min_delta": 0.00,
            "release_imu_max_delta": 0.00,
        },
        {
            "preset": "v2_strong_wrist_rule",
            "algo_mode": "v2_two_stage",
            "gate_action_threshold": 0.55,
            "gate_margin_threshold": 0.05,
            "wrist_rule_min_delta": 0.08,
            "wrist_rule_margin_delta": 0.08,
            "release_emg_min_delta": 0.00,
            "release_imu_max_delta": 0.00,
        },
        {
            "preset": "v2_strong_release_rule",
            "algo_mode": "v2_two_stage",
            "gate_action_threshold": 0.55,
            "gate_margin_threshold": 0.05,
            "wrist_rule_min_delta": 0.00,
            "wrist_rule_margin_delta": 0.00,
            "release_emg_min_delta": -0.08,
            "release_imu_max_delta": 0.30,
        },
    ]


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parent.parent
    run_root = Path(args.run_root)
    summary_dir = run_root / f"{args.run_prefix}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for idx, preset in enumerate(_build_presets(), 1):
        run_id = f"{args.run_prefix}_{idx:02d}_{preset['preset']}"
        cmd = [
            sys.executable,
            "scripts/train_event_algo_baseline.py",
            "--config",
            str(args.config),
            "--runtime_config",
            str(args.runtime_config),
            "--data_dir",
            str(args.data_dir),
            "--recordings_manifest",
            str(args.recordings_manifest),
            "--split_manifest",
            str(args.split_manifest),
            "--target_db5_keys",
            str(args.target_db5_keys),
            "--temperature",
            str(float(args.temperature)),
            "--rule_confidence",
            str(float(args.rule_confidence)),
            "--algo_mode",
            str(preset["algo_mode"]),
            "--gate_action_threshold",
            str(float(preset["gate_action_threshold"])),
            "--gate_margin_threshold",
            str(float(preset["gate_margin_threshold"])),
            "--wrist_rule_min_delta",
            str(float(preset["wrist_rule_min_delta"])),
            "--wrist_rule_margin_delta",
            str(float(preset["wrist_rule_margin_delta"])),
            "--release_emg_min_delta",
            str(float(preset["release_emg_min_delta"])),
            "--release_imu_max_delta",
            str(float(preset["release_imu_max_delta"])),
            "--rule_auto_calibrate",
            "true",
            "--run_root",
            str(args.run_root),
            "--run_id",
            str(run_id),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(code_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        run_dir = run_root / run_id
        offline_path = run_dir / "offline_summary.json"
        metrics_path = run_dir / "evaluation" / "test_metrics.json"
        if proc.returncode != 0 or not offline_path.exists():
            rows.append(
                {
                    "run_id": run_id,
                    "preset": str(preset["preset"]),
                    "status": "failed",
                    "rc": int(proc.returncode),
                    "message": str(proc.stdout).strip()[-1000:],
                }
            )
            continue

        offline = _load_json(offline_path)
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        row = {
            "run_id": run_id,
            "preset": str(preset["preset"]),
            "status": "ok",
            "rc": int(proc.returncode),
            "algo_mode": str(offline.get("algo_mode", preset["algo_mode"])),
            "test_accuracy": float(offline.get("test_accuracy", 0.0)),
            "test_macro_f1": float(offline.get("test_macro_f1", 0.0)),
            "event_action_accuracy": float(metrics.get("event_action_accuracy", 0.0)),
            "event_action_macro_f1": float(metrics.get("event_action_macro_f1", 0.0)),
            "test_command_success_rate": float(offline.get("test_command_success_rate", 0.0)),
            "test_false_trigger_rate": float(offline.get("test_false_trigger_rate", 0.0)),
            "test_false_release_rate": float(offline.get("test_false_release_rate", 0.0)),
            "test_gate_accept_rate": float(offline.get("test_gate_accept_rate", 0.0)),
            "test_gate_action_recall": float(offline.get("test_gate_action_recall", 0.0)),
            "test_stage2_action_acc": float(offline.get("test_stage2_action_acc", 0.0)),
            "test_rule_hit_rate": float(offline.get("test_rule_hit_rate", 0.0)),
            "rule_calibration_status": str((offline.get("rule_calibration", {}) or {}).get("status", "")),
            "rule_thresholds_final": dict(offline.get("rule_thresholds_final", {}) or {}),
            "summary_path": str(offline_path),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best = sorted(ok_rows, key=_rank_key, reverse=True)[0] if ok_rows else None

    summary = {
        "status": "ok" if best is not None else "failed",
        "run_prefix": str(args.run_prefix),
        "rank_rule": (
            "command_success_rate desc, false_trigger_rate asc, false_release_rate asc, "
            "event_action_accuracy desc, event_action_macro_f1 desc"
        ),
        "rows": rows,
        "best_run": best,
    }

    summary_json = summary_dir / "algo_v2_quick_sweep_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "run_id",
        "preset",
        "status",
        "algo_mode",
        "test_command_success_rate",
        "test_false_trigger_rate",
        "test_false_release_rate",
        "event_action_accuracy",
        "event_action_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_gate_accept_rate",
        "test_gate_action_recall",
        "test_stage2_action_acc",
        "test_rule_hit_rate",
        "rule_calibration_status",
    ]
    summary_csv = summary_dir / "algo_v2_quick_sweep_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_md = summary_dir / "algo_v2_quick_sweep_summary.md"
    lines = [
        "# Algo V2 Quick Sweep Summary",
        "",
        f"- run_prefix: `{args.run_prefix}`",
        f"- rank_rule: `{summary['rank_rule']}`",
        f"- summary_json: `{summary_json}`",
        f"- summary_csv: `{summary_csv}`",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                "## Best Run",
                f"- run_id: `{best.get('run_id')}`",
                f"- preset: `{best.get('preset')}`",
                f"- command_success_rate: `{float(best.get('test_command_success_rate', 0.0)):.6f}`",
                f"- false_trigger_rate: `{float(best.get('test_false_trigger_rate', 0.0)):.6f}`",
                f"- false_release_rate: `{float(best.get('test_false_release_rate', 0.0)):.6f}`",
                f"- event_action_accuracy: `{float(best.get('event_action_accuracy', 0.0)):.6f}`",
                f"- event_action_macro_f1: `{float(best.get('event_action_macro_f1', 0.0)):.6f}`",
                f"- test_accuracy: `{float(best.get('test_accuracy', 0.0)):.6f}`",
                f"- test_macro_f1: `{float(best.get('test_macro_f1', 0.0)):.6f}`",
            ]
        )
    else:
        lines.extend(["", "## Best Run", "- no successful runs"])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ALGO-V2-SWEEP] summary_json={summary_json}")
    print(f"[ALGO-V2-SWEEP] summary_csv={summary_csv}")
    print(f"[ALGO-V2-SWEEP] summary_md={summary_md}")


if __name__ == "__main__":
    main()
