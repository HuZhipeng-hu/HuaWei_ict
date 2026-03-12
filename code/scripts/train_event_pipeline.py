"""One-click pipeline: aligned DB5 pretrain -> A/B finetune -> convert -> benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.run_utils import dump_json, ensure_run_dir


@dataclass(frozen=True)
class FinetuneSelection:
    variant: str
    checkpoint_path: str
    reason: str
    scratch_summary: dict
    pretrained_summary: dict


def _load_json(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Expected JSON artifact missing: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return payload


def _resolve_path_under_code_root(path_value: str | Path) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (CODE_ROOT / raw).resolve()


def _run_step(step_name: str, command: list[str], *, cwd: Path) -> None:
    logger = logging.getLogger("event_pipeline")
    logger.info("[%s] %s", step_name, " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _as_float(payload: dict, key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def select_best_finetune(scratch_summary: dict, pretrained_summary: dict) -> FinetuneSelection:
    scratch_f1 = _as_float(scratch_summary, "test_macro_f1")
    scratch_acc = _as_float(scratch_summary, "test_accuracy")
    pre_f1 = _as_float(pretrained_summary, "test_macro_f1")
    pre_acc = _as_float(pretrained_summary, "test_accuracy")
    eps = 1e-9

    if pre_f1 > scratch_f1 + eps:
        return FinetuneSelection(
            variant="pretrained",
            checkpoint_path=str(pretrained_summary["checkpoint_path"]),
            reason="pretrained.test_macro_f1 > scratch.test_macro_f1",
            scratch_summary=scratch_summary,
            pretrained_summary=pretrained_summary,
        )
    if abs(pre_f1 - scratch_f1) <= eps and pre_acc > scratch_acc + eps:
        return FinetuneSelection(
            variant="pretrained",
            checkpoint_path=str(pretrained_summary["checkpoint_path"]),
            reason="macro_f1 tie, pretrained.test_accuracy > scratch.test_accuracy",
            scratch_summary=scratch_summary,
            pretrained_summary=pretrained_summary,
        )
    return FinetuneSelection(
        variant="scratch",
        checkpoint_path=str(scratch_summary["checkpoint_path"]),
        reason="pretrained did not strictly outperform scratch under A/B tie-break rules",
        scratch_summary=scratch_summary,
        pretrained_summary=pretrained_summary,
    )


def _gate_result(benchmark_payload: dict) -> tuple[bool, dict]:
    gate = benchmark_payload.get("merge_gate")
    if not isinstance(gate, dict):
        raise ValueError("Benchmark output missing merge_gate result.")
    passed = bool(gate.get("passed", False))
    checks = gate.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    return passed, checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click event pipeline: pretrain -> finetune A/B -> convert -> benchmark -> gate"
    )
    parser.add_argument("--db5_data_dir", default="../data_ninaproDB5")
    parser.add_argument("--wearer_data_dir", default="../data")
    parser.add_argument("--budget_per_class", type=int, default=60)
    parser.add_argument("--budget_seed", type=int, default=42)
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--allow_legacy_fallback", action="store_true")
    parser.add_argument("--pretrain_config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--finetune_config", default="configs/training_event_onset.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset.yaml")
    parser.add_argument("--conversion_config", default="configs/conversion_event_onset.yaml")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_pipeline")
    args = build_parser().parse_args()

    run_root_path = _resolve_path_under_code_root(args.run_root)
    run_id, run_dir = ensure_run_dir(run_root_path, args.run_id, default_tag="event_pipeline")
    mode = "ascend" if str(args.device_target).lower() == "ascend" else "local"

    preflight_cmd = [
        sys.executable,
        "scripts/preflight.py",
        "--mode",
        mode,
        "--db5_data_dir",
        str(args.db5_data_dir),
        "--wearer_data_dir",
        str(args.wearer_data_dir),
        "--budget_per_class",
        str(int(args.budget_per_class)),
    ]
    _run_step("preflight", preflight_cmd, cwd=CODE_ROOT)

    pretrain_mode = "aligned3"
    pretrain_fallback_used = False
    pretrain_run_id = f"{run_id}_db5_pretrain"
    pretrain_cmd = [
        sys.executable,
        "scripts/pretrain_ninapro_db5.py",
        "--config",
        str(args.pretrain_config),
        "--data_dir",
        str(args.db5_data_dir),
        "--pretrain_mode",
        pretrain_mode,
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--run_root",
        str(run_root_path),
        "--run_id",
        pretrain_run_id,
    ]
    try:
        _run_step("pretrain_aligned3", pretrain_cmd, cwd=CODE_ROOT)
    except subprocess.CalledProcessError:
        if not args.allow_legacy_fallback:
            raise
        pretrain_mode = "legacy53"
        pretrain_fallback_used = True
        pretrain_run_id = f"{run_id}_db5_pretrain_legacy53"
        pretrain_cmd[pretrain_cmd.index("--pretrain_mode") + 1] = "legacy53"
        pretrain_cmd[pretrain_cmd.index("--run_id") + 1] = pretrain_run_id
        _run_step("pretrain_legacy53_fallback", pretrain_cmd, cwd=CODE_ROOT)

    pretrain_summary = _load_json(run_root_path / pretrain_run_id / "offline_summary.json")
    pretrain_ckpt = str(pretrain_summary["checkpoint_path"])

    shared_manifest = run_dir / "manifests" / "ab_shared_split_manifest.json"
    scratch_run_id = f"{run_id}_finetune_scratch"
    pretrained_run_id = f"{run_id}_finetune_pretrained"

    scratch_cmd = [
        sys.executable,
        "scripts/finetune_event_onset.py",
        "--config",
        str(args.finetune_config),
        "--data_dir",
        str(args.wearer_data_dir),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--run_root",
        str(run_root_path),
        "--run_id",
        scratch_run_id,
        "--budget_per_class",
        str(int(args.budget_per_class)),
        "--budget_seed",
        str(int(args.budget_seed)),
        "--split_manifest_out",
        str(shared_manifest),
    ]
    _run_step("finetune_scratch", scratch_cmd, cwd=CODE_ROOT)
    scratch_summary = _load_json(run_root_path / scratch_run_id / "offline_summary.json")

    pretrained_cmd = [
        sys.executable,
        "scripts/finetune_event_onset.py",
        "--config",
        str(args.finetune_config),
        "--data_dir",
        str(args.wearer_data_dir),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--run_root",
        str(run_root_path),
        "--run_id",
        pretrained_run_id,
        "--budget_per_class",
        str(int(args.budget_per_class)),
        "--budget_seed",
        str(int(args.budget_seed)),
        "--split_manifest_in",
        str(shared_manifest),
        "--pretrained_emg_checkpoint",
        pretrain_ckpt,
    ]
    _run_step("finetune_pretrained", pretrained_cmd, cwd=CODE_ROOT)
    pretrained_summary = _load_json(run_root_path / pretrained_run_id / "offline_summary.json")

    selection = select_best_finetune(scratch_summary, pretrained_summary)

    conversion_run_id = f"{run_id}_convert"
    mindir_path = run_dir / "models" / "event_onset_selected.mindir"
    metadata_path = run_dir / "models" / "event_onset_selected.model_metadata.json"
    convert_cmd = [
        sys.executable,
        "scripts/convert_event_onset.py",
        "--config",
        str(args.conversion_config),
        "--training_config",
        str(args.finetune_config),
        "--checkpoint",
        selection.checkpoint_path,
        "--output",
        str(mindir_path),
        "--metadata_output",
        str(metadata_path),
        "--device_target",
        str(args.device_target),
        "--run_root",
        str(run_root_path),
        "--run_id",
        conversion_run_id,
    ]
    _run_step("convert", convert_cmd, cwd=CODE_ROOT)
    conversion_summary = _load_json(run_root_path / conversion_run_id / "conversion" / "event_conversion_summary.json")

    benchmark_path = run_dir / "benchmark" / "event_runtime_benchmark.json"
    benchmark_cmd = [
        sys.executable,
        "scripts/benchmark_event_runtime_ckpt.py",
        "--training_config",
        str(args.finetune_config),
        "--runtime_config",
        str(args.runtime_config),
        "--data_dir",
        str(args.wearer_data_dir),
        "--split_manifest",
        str(shared_manifest),
        "--checkpoint",
        selection.checkpoint_path,
        "--model_path",
        str(mindir_path),
        "--model_metadata",
        str(metadata_path),
        "--output",
        str(benchmark_path),
        "--device_target",
        str(args.device_target),
        "--backend",
        "both",
    ]
    _run_step("benchmark", benchmark_cmd, cwd=CODE_ROOT)
    benchmark_summary = _load_json(benchmark_path)
    gate_passed, gate_checks = _gate_result(benchmark_summary)

    final_selection = {
        "run_id": run_id,
        "pretrain_mode": pretrain_mode,
        "pretrain_fallback_used": bool(pretrain_fallback_used),
        "budget_per_class": int(args.budget_per_class),
        "budget_seed": int(args.budget_seed),
        "selected_finetune_variant": selection.variant,
        "selected_checkpoint_path": selection.checkpoint_path,
        "selection_reason": selection.reason,
        "scratch_summary": selection.scratch_summary,
        "pretrained_summary": selection.pretrained_summary,
    }
    final_artifacts = {
        "run_id": run_id,
        "pretrain_summary_path": str(run_root_path / pretrain_run_id / "offline_summary.json"),
        "scratch_summary_path": str(run_root_path / scratch_run_id / "offline_summary.json"),
        "pretrained_summary_path": str(run_root_path / pretrained_run_id / "offline_summary.json"),
        "selected_checkpoint_path": selection.checkpoint_path,
        "shared_split_manifest": str(shared_manifest),
        "mindir_path": str(mindir_path),
        "model_metadata_path": str(metadata_path),
        "conversion_summary_path": str(run_root_path / conversion_run_id / "conversion" / "event_conversion_summary.json"),
        "benchmark_path": str(benchmark_path),
        "merge_gate_passed": bool(gate_passed),
        "merge_gate_checks": gate_checks,
    }
    dump_json(run_dir / "final_selection.json", final_selection)
    dump_json(run_dir / "final_artifacts.json", final_artifacts)

    if not gate_passed:
        raise RuntimeError(f"Merge gate failed. checks={gate_checks}")

    logger.info("Pipeline completed. final_selection=%s", run_dir / "final_selection.json")
    logger.info("Pipeline completed. final_artifacts=%s", run_dir / "final_artifacts.json")


if __name__ == "__main__":
    main()
