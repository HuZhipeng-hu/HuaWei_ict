"""Generate structured experiment commands for the current dual-branch protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import yaml


def _bool_flag(value: bool) -> str:
    return "true" if value else "false"


def _load_training_defaults(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_default_structure(config: Dict, args: argparse.Namespace) -> Dict[str, object]:
    model_cfg = config.get("model", {})
    return {
        "model_type": args.champion_model_type or model_cfg.get("model_type", "standard"),
        "base_channels": args.champion_base_channels or model_cfg.get("base_channels", 16),
        "use_se": model_cfg.get("use_se", True) if args.champion_use_se is None else args.champion_use_se,
    }


def _train_command(args: argparse.Namespace, run_id: str, overrides: Dict[str, object]) -> str:
    parts = [
        args.python,
        "-m training.train",
        f"--config {args.config}",
        f"--data_dir {args.data_dir}",
        f"--split_manifest_in {args.manifest}",
        f"--run_root {args.run_root}",
        f"--run_id {run_id}",
        "--manifest_strategy v2",
    ]
    for key, value in overrides.items():
        if value is None:
            continue
        parts.append(f"--{key} {value}")
    return " ".join(parts)


def _eval_command(args: argparse.Namespace, run_id: str) -> str:
    checkpoint = f"{args.run_root}/{run_id}/checkpoints/neurogrip_best.ckpt"
    return " ".join(
        [
            args.python,
            "scripts/evaluate_ckpt.py",
            f"--config {args.config}",
            f"--data_dir {args.data_dir}",
            f"--checkpoint {checkpoint}",
            f"--split_manifest {args.manifest}",
            f"--run_root {args.run_root}",
            f"--run_id {run_id}",
        ]
    )


def _benchmark_command(args: argparse.Namespace, run_id: str) -> str:
    checkpoint = f"{args.run_root}/{run_id}/checkpoints/neurogrip_best.ckpt"
    return " ".join(
        [
            args.python,
            "scripts/benchmark_realtime_ckpt.py",
            f"--training_config {args.config}",
            "--runtime_config configs/runtime.yaml",
            f"--data_dir {args.data_dir}",
            f"--checkpoint {checkpoint}",
            f"--split_manifest {args.manifest}",
            f"--run_root {args.run_root}",
            f"--run_id {run_id}",
        ]
    )


def _stage_a_commands(args: argparse.Namespace) -> List[Dict[str, str]]:
    experiments = []
    for model_type in ["standard", "lite"]:
        for base_channels in [16, 24, 32]:
            use_se_values = [True, False] if model_type == "standard" else [False]
            for use_se in use_se_values:
                tag = f"a_{model_type}_b{base_channels}_{'se' if use_se else 'nose'}"
                overrides = {
                    "model_type": model_type,
                    "base_channels": base_channels,
                    "use_se": _bool_flag(use_se),
                }
                experiments.append(
                    {
                        "stage": "A",
                        "run_id": tag,
                        "overrides": str(overrides),
                        "train": _train_command(args, tag, overrides),
                        "eval": _eval_command(args, tag),
                        "benchmark": _benchmark_command(args, tag),
                    }
                )
    return experiments


def _stage_bc_commands(args: argparse.Namespace, structure: Dict[str, object]) -> List[Dict[str, str]]:
    experiments = []
    combos = []
    for loss_type in ["focal", "cb_focal"]:
        for hard_mining_ratio in [0.0, 0.3, 0.5]:
            for augment_factor in [2, 3]:
                for use_mixup in [False, True]:
                    combos.append((loss_type, hard_mining_ratio, augment_factor, use_mixup))
    for index, (loss_type, hard_mining_ratio, augment_factor, use_mixup) in enumerate(combos, start=1):
        tag = f"bc_{index:02d}_{loss_type}_hm{str(hard_mining_ratio).replace('.', '')}_a{augment_factor}_{'mix' if use_mixup else 'nomix'}"
        overrides = {
            "model_type": structure["model_type"],
            "base_channels": structure["base_channels"],
            "use_se": _bool_flag(bool(structure["use_se"])),
            "loss_type": loss_type,
            "hard_mining_ratio": hard_mining_ratio,
            "augment_factor": augment_factor,
            "use_mixup": _bool_flag(use_mixup),
            "augmentation_enabled": "true",
        }
        experiments.append(
            {
                "stage": "B/C",
                "run_id": tag,
                "overrides": str(overrides),
                "train": _train_command(args, tag, overrides),
                "eval": _eval_command(args, tag),
                "benchmark": _benchmark_command(args, tag),
            }
        )
    return experiments


def _stage_d_templates(args: argparse.Namespace, structure: Dict[str, object]) -> List[Dict[str, str]]:
    templates = []
    for label in ["top1", "top2"]:
        for seed in [901, 902, 951, 952]:
            tag = f"d_{label}_seed{seed}"
            overrides = {
                "model_type": structure["model_type"],
                "base_channels": structure["base_channels"],
                "use_se": _bool_flag(bool(structure["use_se"])),
                "split_seed": seed,
            }
            templates.append(
                {
                    "stage": "D",
                    "run_id": tag,
                    "overrides": str(overrides),
                    "train": _train_command(args, tag, overrides),
                    "eval": _eval_command(args, tag),
                    "benchmark": _benchmark_command(args, tag),
                }
            )
    return templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured experiment commands")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--manifest", default="artifacts/splits/default_split_manifest.json")
    parser.add_argument("--python", default="python")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--output", default="artifacts/experiment_matrix.md")
    parser.add_argument("--champion_model_type", default=None, choices=["standard", "lite"])
    parser.add_argument("--champion_base_channels", type=int, default=None)
    parser.add_argument("--champion_use_se", type=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = _load_training_defaults(Path(args.config))
    structure = _resolve_default_structure(defaults, args)

    all_rows = []
    all_rows.extend(_stage_a_commands(args))
    all_rows.extend(_stage_bc_commands(args, structure))
    all_rows.extend(_stage_d_templates(args, structure))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# NeuroGrip Experiment Matrix",
        "",
        f"- Training config: `{args.config}`",
        f"- Data dir: `{args.data_dir}`",
        f"- Locked manifest: `{args.manifest}`",
        f"- Run root: `{args.run_root}`",
        "",
        "## Aggregate CSV Headers",
        "",
        "```text",
        "run_id,manifest_path,checkpoint_path,model_type,base_channels,use_se,loss_type,hard_mining_ratio,augment_enabled,augment_factor,use_mixup,test_accuracy,test_macro_f1,test_macro_recall,top_confusion_pair,hit_rate,false_trigger_rate,latency_p50_ms,latency_p95_ms",
        "```",
        "",
    ]

    for row in all_rows:
        lines.extend(
            [
                f"## Stage {row['stage']} / {row['run_id']}",
                "",
                f"- Overrides: `{row['overrides']}`",
                f"- Expected run dir: `{args.run_root}/{row['run_id']}`",
                "",
                "```bash",
                row["train"],
                row["eval"],
                row["benchmark"],
                "```",
                "",
            ]
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote experiment matrix to {output}")


if __name__ == "__main__":
    main()
