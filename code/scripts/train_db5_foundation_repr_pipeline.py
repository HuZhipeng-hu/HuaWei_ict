"""One-click DB5 representation foundation pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.run_utils import dump_json, ensure_run_dir


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _run_checked(stage: str, cmd: list[str]) -> str:
    cmd_str = _format_cmd(cmd)
    print(f"[REPR_PIPELINE] {stage} -> {cmd_str}")
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {cmd_str}")
    return cmd_str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DB5 representation pretrain + few-shot transfer curve.")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--db5_data_dir", default="../data_ninaproDB5")
    parser.add_argument("--wearer_data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--pretrain_config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--fewshot_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--target_db5_keys", default="E1_G01,E1_G02,E1_G03,E1_G04")
    parser.add_argument("--budgets", default="10,20,35,60")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--ms_mode", default="graph", choices=["graph", "pynative"])
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--knn_k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="db5_repr_pipeline")
    run_dir = Path(run_dir)

    pretrain_run_id = f"{run_id}_repr"
    fewshot_run_id = f"{run_id}_fewshot"

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
        "--projection_dim",
        str(int(args.projection_dim)),
        "--temperature",
        str(float(args.temperature)),
        "--knn_k",
        str(int(args.knn_k)),
    ]
    pretrain_cmd_str = _run_checked("repr_pretrain", pretrain_cmd)
    pretrain_summary = _load_json(Path(args.run_root) / pretrain_run_id / "offline_summary.json")
    encoder_ckpt = str(pretrain_summary.get("checkpoint_path", "")).strip()
    if not encoder_ckpt:
        raise RuntimeError("Representation pretrain summary missing checkpoint_path.")

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
    ]
    if str(args.recordings_manifest or "").strip():
        fewshot_cmd.extend(["--recordings_manifest", str(args.recordings_manifest).strip()])
    fewshot_cmd_str = _run_checked("fewshot_eval", fewshot_cmd)
    fewshot_report = _load_json(Path(args.run_root) / fewshot_run_id / "downstream_fewshot_report.json")

    payload = {
        "run_id": run_id,
        "repr_pretrain_run_id": pretrain_run_id,
        "fewshot_run_id": fewshot_run_id,
        "repr_pretrain_command": pretrain_cmd_str,
        "fewshot_command": fewshot_cmd_str,
        "repr_summary_path": str(Path(args.run_root) / pretrain_run_id / "offline_summary.json"),
        "fewshot_report_path": str(Path(args.run_root) / fewshot_run_id / "downstream_fewshot_report.json"),
        "encoder_checkpoint_path": encoder_ckpt,
        "fewshot_best_budget": fewshot_report.get("best_budget"),
        "fewshot_best_macro_f1_mean": (fewshot_report.get("best_budget_row") or {}).get("macro_f1_mean"),
        "fewshot_best_acc_mean": (fewshot_report.get("best_budget_row") or {}).get("acc_mean"),
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    dump_json(run_dir / "db5_repr_pipeline_summary.json", payload)

    card = "\n".join(
        [
            "# DB5 Representation Foundation Repro Card",
            "",
            "## Commands",
            "```bash",
            pretrain_cmd_str,
            fewshot_cmd_str,
            "```",
            "",
            "## Key Result",
            f"- encoder_checkpoint: `{encoder_ckpt}`",
            f"- fewshot_best_budget: `{payload['fewshot_best_budget']}`",
            f"- fewshot_best_macro_f1_mean: `{payload['fewshot_best_macro_f1_mean']}`",
            f"- fewshot_best_acc_mean: `{payload['fewshot_best_acc_mean']}`",
            "- no personal calibration data required",
            "",
        ]
    )
    (run_dir / "referee_repro_card.md").write_text(card, encoding="utf-8")
    print(f"[REPR_PIPELINE] summary={run_dir / 'db5_repr_pipeline_summary.json'}")
    print(f"[REPR_PIPELINE] card={run_dir / 'referee_repro_card.md'}")


if __name__ == "__main__":
    main()
