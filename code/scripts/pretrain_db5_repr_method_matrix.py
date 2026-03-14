"""Run DB5 representation pretraining stage-2 matrix (pretrain-only, 4 rounds)."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ninapro_db5.config import load_db5_pretrain_config
from ninapro_db5.dataset import DB5PretrainDatasetLoader
from ninapro_db5.evaluate import _set_device
from ninapro_db5.model import build_db5_encoder_model
from scripts import pretrain_ninapro_db5_repr as repr_script
from shared.run_utils import dump_json
from training.data.split_strategy import build_manifest


PUBLIC_RANK_RULE = "best_val_macro_f1 desc, best_val_acc desc, test_macro_f1 desc"
FEWSHOT_RANK_RULE = "recommended_budget asc, macro_f1_mean desc, acc_mean desc, macro_f1_std asc"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_checked(stage: str, cmd: list[str]) -> str:
    cmd_str = _format_cmd(cmd)
    print(f"[REPR-MATRIX] {stage} -> {cmd_str}")
    completed = subprocess.run(cmd, cwd=str(CODE_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with rc={completed.returncode}: {cmd_str}")
    return cmd_str


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


# Backward-compatible helpers retained for existing tests and downstream imports.
def _compute_recommended_budget(curve_rows: list[dict], tolerance: float) -> tuple[int, dict]:
    if not curve_rows:
        raise RuntimeError("few-shot curve rows are empty.")
    sorted_rows = sorted(curve_rows, key=lambda row: int(row["budget"]))
    best_f1 = max(float(row.get("macro_f1_mean", 0.0)) for row in sorted_rows)
    floor = float(best_f1 - float(tolerance))
    for row in sorted_rows:
        if float(row.get("macro_f1_mean", 0.0)) >= floor:
            return int(row["budget"]), row
    return int(sorted_rows[-1]["budget"]), sorted_rows[-1]


def _resolve_fewshot_plan(*, mode: str, recordings_manifest: str | None) -> tuple[bool, str, str]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"off", "auto", "on"}:
        raise ValueError(f"Invalid fewshot_mode={mode!r}, expected one of: off/auto/on")

    raw_manifest = str(recordings_manifest or "").strip()
    if normalized_mode == "off":
        return False, "skipped", "fewshot_mode=off"

    if not raw_manifest:
        if normalized_mode == "auto":
            return False, "skipped", "recordings_manifest not provided"
        raise RuntimeError("fewshot_mode=on requires --recordings_manifest <path>")

    manifest_path = Path(raw_manifest)
    if not manifest_path.exists() or not manifest_path.is_file():
        if normalized_mode == "auto":
            return False, "skipped", f"recordings_manifest not found: {manifest_path}"
        raise RuntimeError(f"fewshot_mode=on requires existing recordings_manifest, got: {manifest_path}")

    return True, "enabled", ""


def _update_plateau_streak(*, previous_budget: int | None, current_budget: int, current_streak: int) -> int:
    if previous_budget is None:
        return 0
    if int(current_budget) < int(previous_budget):
        return 0
    return int(current_streak) + 1


def _rank_key(row: dict) -> tuple[float, float, float, float]:
    rec_budget = int(row.get("recommended_budget", 10**9))
    rec_row = dict(row.get("recommended_budget_row", {}) or {})
    macro_f1 = float(rec_row.get("macro_f1_mean", 0.0))
    acc = float(rec_row.get("acc_mean", 0.0))
    std = float(rec_row.get("macro_f1_std", 1.0))
    return (float(rec_budget), -macro_f1, -acc, std)


def _pretrain_rank_key(row: dict) -> tuple[float, float, float]:
    val_f1 = float(row.get("pretrain_best_val_macro_f1", 0.0))
    val_acc = float(row.get("pretrain_best_val_acc", 0.0))
    test_f1 = float(row.get("pretrain_test_macro_f1", 0.0))
    return (-val_f1, -val_acc, -test_f1)


def _parse_float_grid(raw: str, *, name: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"{name} must contain at least one numeric value")
    return values


def _parse_int_grid(raw: str, *, name: str) -> list[int]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{name} must contain at least one integer value")
    return values


def _safe_float_token(value: float) -> str:
    token = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _with_pretrain_aliases(row: dict) -> dict:
    """Attach backward-compatible aliases for downstream readers.

    Historical consumers may read fields using one of two schemas:
    - new: phase/candidate/run_id/checkpoint_path
    - old: round/name/pretrain_run_id/encoder_checkpoint_path
    """
    payload = dict(row)
    payload["round"] = payload.get("round") or payload.get("phase")
    payload["name"] = payload.get("name") or payload.get("candidate")
    payload["pretrain_run_id"] = payload.get("pretrain_run_id") or payload.get("run_id")
    payload["encoder_checkpoint_path"] = payload.get("encoder_checkpoint_path") or payload.get("checkpoint_path")
    return payload


def _as_pretrain_row(*, phase: str, candidate: str, run_id: str, summary: dict, temperature: float, projection_dim: int, knn_k: int) -> dict:
    return _with_pretrain_aliases(
        {
        "phase": phase,
        "candidate": candidate,
        "run_id": str(run_id),
        "temperature": float(temperature),
        "projection_dim": int(projection_dim),
        "knn_k": int(knn_k),
        "repr_objective": str(summary.get("repr_objective", "supcon")),
        "sampler_mode": str(summary.get("sampler_mode", "class_source_balanced")),
        "augmentation_profile": str(summary.get("augmentation_profile", "strong")),
        "learning_rate": float(summary.get("learning_rate", 0.0) or 0.0),
        "weight_decay": float(summary.get("weight_decay", 0.0) or 0.0),
        "pretrain_best_val_macro_f1": float(summary.get("best_val_macro_f1", 0.0) or 0.0),
        "pretrain_best_val_acc": float(summary.get("best_val_acc", 0.0) or 0.0),
        "pretrain_test_macro_f1": float(summary.get("test_macro_f1", 0.0) or 0.0),
        "pretrain_test_accuracy": float(summary.get("test_accuracy", 0.0) or 0.0),
        "checkpoint_path": str(summary.get("checkpoint_path", "")),
        "pretrain_summary_path": str(summary.get("offline_summary_path", "")),
        "selected": False,
        }
    )


def _build_pretrain_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    temperature: float,
    projection_dim: int,
    knn_k: int,
    repr_objective: str,
    augmentation_profile: str,
    pairing_mode: str,
    quality_sampling_mode: str,
    ce_weight: float,
    contrastive_weight: float,
    temporal_weight: float,
    recon_weight: float,
    learning_rate: float,
    weight_decay: float,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/pretrain_ninapro_db5_repr.py",
        "--config",
        str(args.pretrain_config),
        "--data_dir",
        str(args.db5_data_dir),
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--foundation_dir",
        str(args.foundation_dir),
        "--ms_mode",
        str(args.ms_mode),
        "--repr_objective",
        str(repr_objective),
        "--sampler_mode",
        "class_source_balanced",
        "--augmentation_profile",
        str(augmentation_profile),
        "--pairing_mode",
        str(pairing_mode),
        "--quality_sampling_mode",
        str(quality_sampling_mode),
        "--ce_weight",
        str(float(ce_weight)),
        "--contrastive_weight",
        str(float(contrastive_weight)),
        "--temporal_weight",
        str(float(temporal_weight)),
        "--recon_weight",
        str(float(recon_weight)),
        "--label_smoothing",
        str(float(args.label_smoothing)),
        "--learning_rate",
        str(float(learning_rate)),
        "--weight_decay",
        str(float(weight_decay)),
        "--epochs",
        str(int(args.epochs)),
        "--early_stopping_patience",
        str(int(args.early_stopping_patience)),
        "--temperature",
        str(float(temperature)),
        "--projection_dim",
        str(int(projection_dim)),
        "--knn_k",
        str(int(knn_k)),
        "--run_downstream_fewshot",
        "false",
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(int(args.batch_size))])
    if args.manifest_use_source_metadata is not None:
        cmd.extend(["--manifest_use_source_metadata", str(args.manifest_use_source_metadata).lower()])
    return cmd


def _run_pretrain_candidate(
    *,
    args: argparse.Namespace,
    phase: str,
    candidate: str,
    run_id: str,
    temperature: float,
    projection_dim: int,
    knn_k: int,
    repr_objective: str,
    augmentation_profile: str,
    pairing_mode: str,
    quality_sampling_mode: str,
    ce_weight: float,
    contrastive_weight: float,
    temporal_weight: float,
    recon_weight: float,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict, str]:
    stage = f"{phase}_{candidate}"
    cmd = _build_pretrain_cmd(
        args,
        run_id=run_id,
        temperature=temperature,
        projection_dim=projection_dim,
        knn_k=knn_k,
        repr_objective=repr_objective,
        augmentation_profile=augmentation_profile,
        pairing_mode=pairing_mode,
        quality_sampling_mode=quality_sampling_mode,
        ce_weight=ce_weight,
        contrastive_weight=contrastive_weight,
        temporal_weight=temporal_weight,
        recon_weight=recon_weight,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    cmd_str = _run_checked(stage, cmd)
    summary_path = Path(args.run_root) / run_id / "offline_summary.json"
    summary = _load_json(summary_path)
    summary["offline_summary_path"] = str(summary_path)
    row = _as_pretrain_row(
        phase=phase,
        candidate=candidate,
        run_id=run_id,
        summary=summary,
        temperature=temperature,
        projection_dim=projection_dim,
        knn_k=knn_k,
    )
    if not str(row.get("checkpoint_path", "")).strip():
        raise RuntimeError(f"missing checkpoint_path in {summary_path}")
    return row, cmd_str


def _pick_best_pretrain(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("no pretrain rows available for ranking")
    return sorted(rows, key=_pretrain_rank_key)[0]


def _build_bottleneck_profile(*, run_root: Path, run_id: str, epochs: int) -> dict:
    run_dir = Path(run_root) / str(run_id)
    offline = _load_json(run_dir / "offline_summary.json")
    window_diag = _load_json(run_dir / "db5_window_diagnostics.json")
    totals = dict(window_diag.get("totals", {}) or {})
    raw = float(totals.get("raw_candidates", 0) or 0)
    selected = float(totals.get("selected", 0) or 0)
    window_util = float(selected / raw) if raw > 0 else 0.0
    best_epoch = int(offline.get("best_val_epoch", -1) or -1)
    best_val_f1 = float(offline.get("best_val_macro_f1", 0.0) or 0.0)
    early_peak = bool(best_epoch > 0 and best_epoch <= max(8, int(epochs * 0.12)))
    weak_signal = bool(best_val_f1 < 0.05)
    low_window_util = bool(window_util < 0.35)
    signature = f"early_peak={int(early_peak)}|weak_signal={int(weak_signal)}|low_util={int(low_window_util)}"
    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "window_utilization": window_util,
        "early_peak": early_peak,
        "weak_signal": weak_signal,
        "low_window_util": low_window_util,
        "signature": signature,
    }


def _run4_knn_robustness(
    *,
    args: argparse.Namespace,
    checkpoint_path: str,
    knn_grid: list[int],
    run_dir: Path,
) -> dict:
    config = load_db5_pretrain_config(str(args.pretrain_config))
    config.data_dir = str(args.db5_data_dir)
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.manifest_use_source_metadata is not None:
        config.manifest_use_source_metadata = bool(args.manifest_use_source_metadata)

    loader = DB5PretrainDatasetLoader(str(config.data_dir), config)
    samples, labels, source_ids, source_metadata = loader.load_all_with_sources(return_metadata=True)
    class_names = loader.get_class_names()
    if not class_names:
        raise RuntimeError("DB5 class_names is empty in run4 evaluation.")

    manifest = build_manifest(
        labels,
        source_ids,
        seed=config.split_seed,
        split_mode="grouped_file",
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_classes=len(class_names),
        class_names=class_names,
        manifest_strategy="v2",
        source_metadata=source_metadata if config.manifest_use_source_metadata else None,
    )
    if repr_script._has_group_leakage(manifest):
        raise RuntimeError("Group leakage detected in run4 kNN robustness manifest.")

    train_idx = manifest.train_indices
    val_idx = manifest.val_indices
    test_idx = manifest.test_indices
    train_x, train_y = samples[train_idx], labels[train_idx]
    val_x, val_y = samples[val_idx], labels[val_idx]
    test_x, test_y = samples[test_idx], labels[test_idx]

    _set_device(mode=str(args.ms_mode), target=str(args.device_target), device_id=int(args.device_id))
    encoder = build_db5_encoder_model(config)

    try:
        from mindspore import load_checkpoint, load_param_into_net
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("MindSpore is required for run4 kNN robustness evaluation") from exc

    ckpt = str(checkpoint_path).strip()
    if not ckpt:
        raise RuntimeError("run4 robustness evaluation requires non-empty checkpoint path")
    params = load_checkpoint(ckpt)
    load_param_into_net(encoder, params)
    encoder.set_train(False)

    grid_rows: list[dict] = []
    for k in knn_grid:
        val_report = repr_script._evaluate_repr_knn(
            encoder,
            train_x=train_x,
            train_y=train_y,
            eval_x=val_x,
            eval_y=val_y,
            class_names=class_names,
            batch_size=int(config.training.batch_size),
            knn_k=int(k),
        )
        test_report = repr_script._evaluate_repr_knn(
            encoder,
            train_x=train_x,
            train_y=train_y,
            eval_x=test_x,
            eval_y=test_y,
            class_names=class_names,
            batch_size=int(config.training.batch_size),
            knn_k=int(k),
        )
        grid_rows.append(
            {
                "knn_k": int(k),
                "val_macro_f1": float(val_report.get("macro_f1", 0.0)),
                "val_acc": float(val_report.get("accuracy", 0.0)),
                "test_macro_f1": float(test_report.get("macro_f1", 0.0)),
                "test_accuracy": float(test_report.get("accuracy", 0.0)),
            }
        )

    def _grid_rank_key(row: dict) -> tuple[float, float, float]:
        return (
            -float(row.get("val_macro_f1", 0.0)),
            -float(row.get("val_acc", 0.0)),
            -float(row.get("test_macro_f1", 0.0)),
        )

    best_row = sorted(grid_rows, key=_grid_rank_key)[0]
    payload = {
        "status": "completed",
        "checkpoint_path": ckpt,
        "rank_rule": PUBLIC_RANK_RULE,
        "rows": grid_rows,
        "best": best_row,
    }
    dump_json(run_dir / "run4_knn_robustness.json", payload)
    _write_csv(
        run_dir / "run4_knn_robustness.csv",
        grid_rows,
        fields=["knn_k", "val_macro_f1", "val_acc", "test_macro_f1", "test_accuracy"],
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DB5 representation method-driven matrix: "
            "Stage-A method blocks (A1-A6) then Stage-B focused parameter polish (B1-B4)."
        )
    )
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="db5_repr_stage2_v1")
    parser.add_argument("--pretrain_config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--db5_data_dir", default="../data_ninaproDB5")
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--ms_mode", default="graph", choices=["graph", "pynative"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--baseline_temperature", type=float, default=0.07)
    parser.add_argument("--baseline_projection_dim", type=int, default=128)
    parser.add_argument("--baseline_knn_k", type=int, default=5)
    parser.add_argument("--baseline_repr_objective", default="supcon")
    parser.add_argument("--temperature_grid", default="0.05,0.07,0.10")
    parser.add_argument("--projection_dim_grid", default="128,192,256")
    parser.add_argument("--knn_k_grid", default="3,5,11")
    parser.add_argument("--b1_learning_rate", type=float, default=3e-4)
    parser.add_argument("--b2_temperature", type=float, default=0.05)
    parser.add_argument("--b3_projection_dim", type=int, default=192)
    parser.add_argument("--b4_ce_weight", type=float, default=0.10)
    parser.add_argument("--method_gain_eps", type=float, default=0.002)
    parser.add_argument("--method_plateau_rounds", type=int, default=2)
    parser.add_argument("--fewshot_mode", default="off", choices=["off", "auto", "on"])
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--manifest_use_source_metadata", choices=["true", "false"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_ts = time.time()
    run_root = Path(args.run_root)
    run_dir = run_root / f"{args.run_prefix}_summary"
    run_dir.mkdir(parents=True, exist_ok=True)

    knn_grid = _parse_int_grid(args.knn_k_grid, name="knn_k_grid")

    rows: list[dict] = []
    commands: list[dict] = []
    last_stage = "setup"
    last_cmd = ""
    stop_reason = ""
    method_plateau_detected = False
    method_plateau_streak = 0
    run4_payload: dict | None = None
    stage_a_rows: list[dict] = []
    stage_b_rows: list[dict] = []
    profile_history: list[dict] = []

    fewshot_mode = str(args.fewshot_mode).strip().lower()
    fewshot_status = "skipped"
    fewshot_skip_reason = "fewshot_mode=off" if fewshot_mode == "off" else "method matrix is pretrain-only"

    stage_a_specs = [
        {
            "phase": "A1",
            "candidate": "baseline_supcon",
            "temperature": float(args.baseline_temperature),
            "projection_dim": int(args.baseline_projection_dim),
            "knn_k": int(args.baseline_knn_k),
            "repr_objective": "supcon",
            "augmentation_profile": "strong",
            "pairing_mode": "same_window_aug",
            "quality_sampling_mode": "uniform",
            "ce_weight": 0.0,
            "contrastive_weight": 1.0,
            "temporal_weight": 0.0,
            "recon_weight": 0.0,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        {
            "phase": "A2",
            "candidate": "quality_aware_pairing",
            "temperature": float(args.baseline_temperature),
            "projection_dim": int(args.baseline_projection_dim),
            "knn_k": int(args.baseline_knn_k),
            "repr_objective": "supcon",
            "augmentation_profile": "strong",
            "pairing_mode": "quality_mixed",
            "quality_sampling_mode": "quality",
            "ce_weight": 0.0,
            "contrastive_weight": 1.0,
            "temporal_weight": 0.0,
            "recon_weight": 0.0,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        {
            "phase": "A3",
            "candidate": "curriculum_augmentation",
            "temperature": float(args.baseline_temperature),
            "projection_dim": int(args.baseline_projection_dim),
            "knn_k": int(args.baseline_knn_k),
            "repr_objective": "supcon",
            "augmentation_profile": "curriculum",
            "pairing_mode": "quality_mixed",
            "quality_sampling_mode": "quality",
            "ce_weight": 0.0,
            "contrastive_weight": 1.0,
            "temporal_weight": 0.0,
            "recon_weight": 0.0,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        {
            "phase": "A4",
            "candidate": "multitask_repr_only",
            "temperature": float(args.baseline_temperature),
            "projection_dim": int(args.baseline_projection_dim),
            "knn_k": int(args.baseline_knn_k),
            "repr_objective": "multitask_repr",
            "augmentation_profile": "strong",
            "pairing_mode": "same_window_aug",
            "quality_sampling_mode": "uniform",
            "ce_weight": 0.0,
            "contrastive_weight": 1.0,
            "temporal_weight": 0.3,
            "recon_weight": 0.2,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        {
            "phase": "A5",
            "candidate": "multitask_plus_quality",
            "temperature": float(args.baseline_temperature),
            "projection_dim": int(args.baseline_projection_dim),
            "knn_k": int(args.baseline_knn_k),
            "repr_objective": "multitask_repr",
            "augmentation_profile": "strong",
            "pairing_mode": "quality_mixed",
            "quality_sampling_mode": "quality",
            "ce_weight": 0.0,
            "contrastive_weight": 1.0,
            "temporal_weight": 0.3,
            "recon_weight": 0.2,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        {
            "phase": "A6",
            "candidate": "multitask_quality_curriculum",
            "temperature": float(args.baseline_temperature),
            "projection_dim": int(args.baseline_projection_dim),
            "knn_k": int(args.baseline_knn_k),
            "repr_objective": "multitask_repr",
            "augmentation_profile": "curriculum",
            "pairing_mode": "quality_mixed",
            "quality_sampling_mode": "quality",
            "ce_weight": 0.0,
            "contrastive_weight": 1.0,
            "temporal_weight": 0.3,
            "recon_weight": 0.2,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
    ]

    try:
        previous_row: dict | None = None
        previous_profile: dict | None = None

        for spec in stage_a_specs:
            run_id = f"{args.run_prefix}_{spec['phase'].lower()}"
            last_stage = f"{spec['phase']}_{spec['candidate']}"
            row, cmd_str = _run_pretrain_candidate(
                args=args,
                phase=str(spec["phase"]),
                candidate=str(spec["candidate"]),
                run_id=run_id,
                temperature=float(spec["temperature"]),
                projection_dim=int(spec["projection_dim"]),
                knn_k=int(spec["knn_k"]),
                repr_objective=str(spec["repr_objective"]),
                augmentation_profile=str(spec["augmentation_profile"]),
                pairing_mode=str(spec["pairing_mode"]),
                quality_sampling_mode=str(spec["quality_sampling_mode"]),
                ce_weight=float(spec["ce_weight"]),
                contrastive_weight=float(spec["contrastive_weight"]),
                temporal_weight=float(spec["temporal_weight"]),
                recon_weight=float(spec["recon_weight"]),
                learning_rate=float(spec["learning_rate"]),
                weight_decay=float(spec["weight_decay"]),
            )
            profile = _build_bottleneck_profile(run_root=run_root, run_id=run_id, epochs=int(args.epochs))
            row["method_settings"] = dict(spec)
            row["bottleneck_profile"] = dict(profile)
            row["selected"] = True if spec["phase"] == "A1" else False
            if previous_row is not None and previous_profile is not None:
                val_gain = float(row["pretrain_best_val_macro_f1"]) - float(previous_row["pretrain_best_val_macro_f1"])
                same_signature = str(profile.get("signature", "")) == str(previous_profile.get("signature", ""))
                if same_signature and val_gain < float(args.method_gain_eps):
                    method_plateau_streak += 1
                else:
                    method_plateau_streak = 0
                row["method_gain_vs_prev"] = float(val_gain)
                row["profile_changed_vs_prev"] = bool(not same_signature)
            else:
                row["method_gain_vs_prev"] = None
                row["profile_changed_vs_prev"] = None

            last_cmd = cmd_str
            rows.append(row)
            stage_a_rows.append(row)
            profile_history.append(profile)
            commands.append({"stage": last_stage, "command": cmd_str})
            previous_row = row
            previous_profile = profile

            if int(method_plateau_streak) >= int(args.method_plateau_rounds):
                method_plateau_detected = True
                stop_reason = (
                    f"Stage-A method yield plateaued: streak={method_plateau_streak}, "
                    f"eps={float(args.method_gain_eps):.4f}"
                )
                break

        if stage_a_rows:
            stage_a_rows[0]["selected"] = True
            for row in stage_a_rows[1:]:
                row["selected"] = False

        if stage_a_rows and not method_plateau_detected:
            base = dict(stage_a_rows[-1].get("method_settings", {}))
            base_phase = "B"
            stage_b_specs = [
                {
                    **base,
                    "phase": "B1",
                    "candidate": "lr_polish",
                    "learning_rate": float(args.b1_learning_rate),
                },
                {
                    **base,
                    "phase": "B2",
                    "candidate": "temperature_polish",
                    "temperature": float(args.b2_temperature),
                },
                {
                    **base,
                    "phase": "B3",
                    "candidate": "projection_polish",
                    "projection_dim": int(args.b3_projection_dim),
                },
                {
                    **base,
                    "phase": "B4",
                    "candidate": "ce_weight_probe",
                    "ce_weight": float(args.b4_ce_weight),
                },
            ]
            for spec in stage_b_specs:
                run_id = f"{args.run_prefix}_{spec['phase'].lower()}"
                last_stage = f"{spec['phase']}_{spec['candidate']}"
                row, cmd_str = _run_pretrain_candidate(
                    args=args,
                    phase=str(spec["phase"]),
                    candidate=str(spec["candidate"]),
                    run_id=run_id,
                    temperature=float(spec["temperature"]),
                    projection_dim=int(spec["projection_dim"]),
                    knn_k=int(spec["knn_k"]),
                    repr_objective=str(spec["repr_objective"]),
                    augmentation_profile=str(spec["augmentation_profile"]),
                    pairing_mode=str(spec["pairing_mode"]),
                    quality_sampling_mode=str(spec["quality_sampling_mode"]),
                    ce_weight=float(spec["ce_weight"]),
                    contrastive_weight=float(spec["contrastive_weight"]),
                    temporal_weight=float(spec["temporal_weight"]),
                    recon_weight=float(spec["recon_weight"]),
                    learning_rate=float(spec["learning_rate"]),
                    weight_decay=float(spec["weight_decay"]),
                )
                profile = _build_bottleneck_profile(run_root=run_root, run_id=run_id, epochs=int(args.epochs))
                row["method_settings"] = dict(spec)
                row["bottleneck_profile"] = dict(profile)
                row["selected"] = False
                last_cmd = cmd_str
                rows.append(row)
                stage_b_rows.append(row)
                profile_history.append(profile)
                commands.append({"stage": last_stage, "command": cmd_str, "stage_group": base_phase})

    except Exception as exc:
        default_cmd = (
            "python scripts/pretrain_db5_repr_method_matrix.py "
            f"--run_prefix {args.run_prefix} --db5_data_dir {args.db5_data_dir} --fewshot_mode off"
        )
        dump_json(
            run_dir / "repr_method_matrix_failure_report.json",
            {
                "status": "failed",
                "stage": str(last_stage),
                "root_cause": str(exc),
                "next_command": str(last_cmd or default_cmd),
                "generated_at_unix": int(time.time()),
            },
        )
        raise

    if not rows:
        raise RuntimeError("no successful method-matrix pretrain rows")

    best_pretrain = _pick_best_pretrain(rows)
    for row in rows:
        row["selected"] = bool(str(row.get("run_id", "")) == str(best_pretrain.get("run_id", "")))

    # kNN robustness on best pretrain checkpoint.
    checkpoint_path = str(best_pretrain.get("checkpoint_path", "")).strip()
    if checkpoint_path:
        last_stage = "run_robust_knn_eval"
        run4_payload = _run4_knn_robustness(
            args=args,
            checkpoint_path=checkpoint_path,
            knn_grid=knn_grid,
            run_dir=run_dir,
        )
        commands.append(
            {
                "stage": last_stage,
                "command": "internal_eval:run_robust_knn_eval",
                "checkpoint_path": checkpoint_path,
                "knn_grid": knn_grid,
            }
        )

    elapsed_minutes = float((time.time() - start_ts) / 60.0)

    matrix_payload = {
        "run_prefix": str(args.run_prefix),
        "fewshot_mode": fewshot_mode,
        "fewshot_status": fewshot_status,
        "fewshot_skip_reason": fewshot_skip_reason,
        "public_rank_rule": PUBLIC_RANK_RULE,
        "fewshot_rank_rule": FEWSHOT_RANK_RULE,
        "matrix_strategy": "method_first_then_parameter_polish",
        "stage_a_rows": stage_a_rows,
        "stage_b_rows": stage_b_rows,
        "method_gain_eps": float(args.method_gain_eps),
        "method_plateau_rounds": int(args.method_plateau_rounds),
        "method_plateau_detected": bool(method_plateau_detected),
        "method_plateau_streak": int(method_plateau_streak),
        "profile_history": profile_history,
        "stop_reason": str(stop_reason),
        "run_robust_knn_eval": run4_payload,
        "best_pretrain_run": best_pretrain,
        "best_pretrain_run_id": str(best_pretrain.get("run_id", "")),
        "best_pretrain_round": str(best_pretrain.get("phase", "")),
        "best_pretrain_name": str(best_pretrain.get("candidate", "")),
        "best_pretrain_checkpoint_path": str(best_pretrain.get("checkpoint_path", "")),
        "rows": rows,
        "commands": commands,
        "elapsed_minutes": elapsed_minutes,
    }
    dump_json(run_dir / "db5_repr_method_matrix_summary.json", matrix_payload)
    _write_csv(
        run_dir / "db5_repr_method_matrix_summary.csv",
        rows,
        fields=[
            "phase",
            "candidate",
            "selected",
            "run_id",
            "temperature",
            "projection_dim",
            "knn_k",
            "repr_objective",
            "method_gain_vs_prev",
            "profile_changed_vs_prev",
            "sampler_mode",
            "augmentation_profile",
            "learning_rate",
            "weight_decay",
            "pretrain_best_val_macro_f1",
            "pretrain_best_val_acc",
            "pretrain_test_macro_f1",
            "pretrain_test_accuracy",
            "checkpoint_path",
            "pretrain_summary_path",
        ],
    )

    card = "\n".join(
        [
            "# DB5 Repr Method Matrix Repro Card",
            "",
            "## Goal",
            "- strengthen DB5 foundation representation pretraining",
            "- method-first optimization for migration-ready representations",
            "- keep public reproducibility track pretrain-only",
            "",
            "## Stage-A Method Blocks",
            "- A1 baseline_supcon",
            "- A2 quality_aware_pairing",
            "- A3 curriculum_augmentation",
            "- A4 multitask_repr_only",
            "- A5 multitask_plus_quality",
            "- A6 multitask_quality_curriculum",
            "",
            "## Stage Results",
            f"- total_runs: `{len(rows)}`",
            f"- stage_a_runs: `{len(stage_a_rows)}`",
            f"- stage_b_runs: `{len(stage_b_rows)}`",
            f"- method_plateau_detected: `{bool(method_plateau_detected)}`",
            f"- stop_reason: `{stop_reason}`",
            f"- robust_knn_eval: `{'completed' if run4_payload else 'skipped'}`",
            f"- robust_knn_best_k: `{int(run4_payload['best']['knn_k']) if run4_payload else ''}`",
            "",
            "## Best Pretrain",
            f"- run_id: `{best_pretrain['run_id']}`",
            f"- best_val_macro_f1: `{float(best_pretrain['pretrain_best_val_macro_f1']):.4f}`",
            f"- best_val_acc: `{float(best_pretrain['pretrain_best_val_acc']):.4f}`",
            f"- test_macro_f1: `{float(best_pretrain['pretrain_test_macro_f1']):.4f}`",
            f"- checkpoint: `{best_pretrain['checkpoint_path']}`",
            f"- public_rank_rule: `{PUBLIC_RANK_RULE}`",
            "",
            "## Commands",
            "- public_pretrain_only: `python scripts/pretrain_db5_repr_method_matrix.py --fewshot_mode off --db5_data_dir ../data_ninaproDB5 --run_prefix <run_id>`",
            "",
            "## Artifacts",
            f"- summary_json: `{run_dir / 'db5_repr_method_matrix_summary.json'}`",
            f"- summary_csv: `{run_dir / 'db5_repr_method_matrix_summary.csv'}`",
            f"- robust_knn_json: `{run_dir / 'run4_knn_robustness.json'}`",
            f"- robust_knn_csv: `{run_dir / 'run4_knn_robustness.csv'}`",
        ]
    )
    (run_dir / "referee_repro_card.md").write_text(card, encoding="utf-8")
    print(f"[REPR-MATRIX] summary={run_dir / 'db5_repr_method_matrix_summary.json'}")
    print(f"[REPR-MATRIX] card={run_dir / 'referee_repro_card.md'}")


if __name__ == "__main__":
    main()
