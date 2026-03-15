"""Run 4-action+RELAX demo training matrix and rank by test accuracy first."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


DEFAULT_CONFIGS = "configs/training_event_onset_demo_p0.yaml,configs/training_event_onset_demo_p1.yaml"
DEFAULT_TAGS = "demo_p0,demo_p1"
DEFAULT_SEEDS = "42,77,99"
DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_checked(stage: str, cmd: list[str]) -> str:
    cmd_str = _format_cmd(cmd)
    print(f"[DEMO-MATRIX] {stage} -> {cmd_str}", flush=True)
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {cmd_str}")
    return cmd_str


def _parse_csv_tokens(raw: str, *, name: str) -> list[str]:
    tokens = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not tokens:
        raise ValueError(f"{name} must contain at least one value")
    return tokens


def _parse_int_tokens(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for token in _parse_csv_tokens(raw, name=name):
        values.append(int(token))
    return values


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload is not object: {path}")
    return payload


def _compute_relax_action_confusion(top_pairs: list[dict], *, action_keys: set[str]) -> int:
    total = 0
    for item in top_pairs:
        pair = [str(v).strip().upper() for v in list(item.get("pair") or [])]
        if len(pair) != 2:
            continue
        if "RELAX" not in pair:
            continue
        other = pair[0] if pair[1] == "RELAX" else pair[1]
        if other in action_keys:
            total += int(item.get("count", 0) or 0)
    return int(total)


def _rank_key(row: dict) -> tuple[int, float, float, float]:
    return (
        int(bool(row.get("safety_ok", False))),
        float(row.get("test_accuracy", 0.0)),
        float(row.get("test_macro_f1", 0.0)),
        -float(row.get("relax_action_confusion", 0.0)),
    )


def _build_finetune_cmd(
    args: argparse.Namespace,
    *,
    config_path: str,
    split_seed: int,
    run_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/finetune_event_onset.py",
        "--config",
        str(config_path),
        "--data_dir",
        str(args.data_dir),
        "--recordings_manifest",
        str(args.recordings_manifest),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--budget_per_class",
        str(int(args.budget_per_class)),
        "--budget_seed",
        str(int(args.budget_seed)),
        "--split_seed",
        str(int(split_seed)),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
    ]
    if str(args.pretrained_emg_checkpoint or "").strip():
        cmd.extend(["--pretrained_emg_checkpoint", str(args.pretrained_emg_checkpoint)])
    return cmd


def _load_run_metrics(run_root: Path, run_id: str) -> tuple[dict, dict]:
    run_dir = run_root / run_id
    offline = _load_json(run_dir / "offline_summary.json")
    metrics = _load_json(run_dir / "evaluation" / "test_metrics.json")
    return offline, metrics


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "run_id",
        "config_tag",
        "config_path",
        "split_seed",
        "test_accuracy",
        "test_macro_f1",
        "best_val_acc",
        "best_val_f1",
        "relax_action_confusion",
        "safety_ok",
        "checkpoint_path",
        "summary_path",
        "metrics_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run demo latch matrix (2 configs x 3 seeds)")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="Comma-separated config yaml paths")
    parser.add_argument("--config_tags", default=DEFAULT_TAGS, help="Comma-separated tags matching --configs")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated split seeds")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default="s2_train_manifest_relax12.csv")
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--budget_per_class", type=int, default=0)
    parser.add_argument("--budget_seed", type=int, default=42)
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--pretrained_emg_checkpoint", default="")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="s2_demo_latch_matrix_v1")
    parser.add_argument("--relax_action_confusion_limit", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    configs = _parse_csv_tokens(args.configs, name="--configs")
    tags = _parse_csv_tokens(args.config_tags, name="--config_tags")
    seeds = _parse_int_tokens(args.seeds, name="--seeds")
    if len(configs) != len(tags):
        raise ValueError("--configs and --config_tags size mismatch")

    action_keys = {item.strip().upper() for item in str(args.target_db5_keys).split(",") if item.strip()}

    rows: list[dict] = []
    last_stage = "init"
    last_cmd = ""

    try:
        for config_path, config_tag in zip(configs, tags):
            for seed in seeds:
                run_id = f"{args.run_prefix}_{config_tag}_s{seed}"
                stage = f"{config_tag}_seed{seed}"
                cmd = _build_finetune_cmd(args, config_path=config_path, split_seed=seed, run_id=run_id)
                last_stage = stage
                last_cmd = _format_cmd(cmd)
                _run_checked(stage, cmd)

                offline, metrics = _load_run_metrics(run_root, run_id)
                top_pairs = list(metrics.get("top_confusion_pairs") or [])
                relax_action_confusion = _compute_relax_action_confusion(top_pairs, action_keys=action_keys)
                safety_ok = relax_action_confusion <= int(args.relax_action_confusion_limit)

                row = {
                    "run_id": run_id,
                    "config_tag": config_tag,
                    "config_path": str(config_path),
                    "split_seed": int(seed),
                    "test_accuracy": float(metrics.get("accuracy", offline.get("test_accuracy", 0.0)) or 0.0),
                    "test_macro_f1": float(metrics.get("macro_f1", offline.get("test_macro_f1", 0.0)) or 0.0),
                    "best_val_acc": float(offline.get("best_val_acc", 0.0) or 0.0),
                    "best_val_f1": float(offline.get("best_val_f1", offline.get("best_val_macro_f1", 0.0)) or 0.0),
                    "relax_action_confusion": int(relax_action_confusion),
                    "safety_ok": bool(safety_ok),
                    "top_confusion_pairs": top_pairs[:10],
                    "checkpoint_path": str(offline.get("checkpoint_path", "")),
                    "summary_path": str((run_root / run_id / "offline_summary.json")),
                    "metrics_path": str((run_root / run_id / "evaluation" / "test_metrics.json")),
                }
                rows.append(row)
    except Exception as exc:
        failure = {
            "status": "failed",
            "stage": last_stage,
            "root_cause": str(exc),
            "next_command": last_cmd,
        }
        out_dir = run_root / f"{args.run_prefix}_summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "demo_latch_matrix_failure_report.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise

    if not rows:
        raise RuntimeError("No matrix rows generated")

    ranked = sorted(rows, key=_rank_key, reverse=True)
    best = dict(ranked[0])
    best["selected"] = True
    for row in rows:
        row["selected"] = row["run_id"] == best["run_id"]

    summary = {
        "run_prefix": str(args.run_prefix),
        "target_db5_keys": sorted(action_keys),
        "configs": [{"tag": tag, "path": path} for path, tag in zip(configs, tags)],
        "seeds": [int(seed) for seed in seeds],
        "rows": rows,
        "best_run": best,
        "rank_rule": "safety_ok desc, test_accuracy desc, test_macro_f1 desc, relax_action_confusion asc",
        "safety": {
            "relax_action_confusion_limit": int(args.relax_action_confusion_limit),
            "safety_fail_count": int(sum(1 for row in rows if not bool(row.get("safety_ok", False)))),
        },
    }

    out_dir = run_root / f"{args.run_prefix}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "demo_latch_matrix_summary.json"
    summary_csv = out_dir / "demo_latch_matrix_summary.csv"
    summary_md = out_dir / "demo_latch_matrix_summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, rows)

    lines = [
        "# Demo Latch Matrix Summary",
        "",
        f"- run_prefix: `{args.run_prefix}`",
        f"- target_db5_keys: `{','.join(sorted(action_keys))}`",
        f"- rank_rule: `{summary['rank_rule']}`",
        "",
        "## Best Run",
        f"- run_id: `{best.get('run_id')}`",
        f"- config_tag: `{best.get('config_tag')}`",
        f"- split_seed: `{best.get('split_seed')}`",
        f"- test_accuracy: `{float(best.get('test_accuracy', 0.0)):.6f}`",
        f"- test_macro_f1: `{float(best.get('test_macro_f1', 0.0)):.6f}`",
        f"- relax_action_confusion: `{int(best.get('relax_action_confusion', 0))}`",
        f"- safety_ok: `{bool(best.get('safety_ok', False))}`",
        "",
        "## Artifacts",
        f"- summary_json: `{summary_json}`",
        f"- summary_csv: `{summary_csv}`",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DEMO-MATRIX] summary_json={summary_json}")
    print(f"[DEMO-MATRIX] summary_csv={summary_csv}")
    print(f"[DEMO-MATRIX] summary_md={summary_md}")


if __name__ == "__main__":
    main()
