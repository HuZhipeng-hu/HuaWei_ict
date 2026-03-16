"""Audit model-line pipeline to rule out design/implementation/parameter issues."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.config import load_event_runtime_config, load_event_training_config
from shared.label_modes import get_label_mode_spec
from training.data.split_strategy import load_manifest


def _parse_tokens(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_int_tokens(raw: str) -> list[int]:
    return [int(item) for item in _parse_tokens(raw)]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be object: {path}")
    return payload


def _resolve_code_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (CODE_ROOT / path).resolve()


def _is_finite_prob(value: Any) -> bool:
    try:
        number = float(value)
    except Exception:
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


@dataclass
class CheckResult:
    name: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.passed = False
        self.issues.append(str(message))


def _check_design_contract(args: argparse.Namespace) -> CheckResult:
    result = CheckResult(name="design_contract")
    model_cfg, train_data_cfg, _, _ = load_event_training_config(str(args.training_config))
    runtime_cfg = load_event_runtime_config(str(args.runtime_config))

    train_keys = [str(item).strip().upper() for item in train_data_cfg.target_db5_keys]
    runtime_keys = [str(item).strip().upper() for item in runtime_cfg.data.target_db5_keys]
    if train_keys != runtime_keys:
        result.fail(f"target_db5_keys mismatch: training={train_keys}, runtime={runtime_keys}")

    label_spec = get_label_mode_spec(train_data_cfg.label_mode, train_data_cfg.target_db5_keys)
    expected_class_names = [str(item).strip().upper() for item in label_spec.class_names]
    model_cfg.num_classes = int(len(expected_class_names))
    label_to_state, mapping_by_name = load_and_validate_actuation_map(
        runtime_cfg.actuation_mapping_path,
        class_names=expected_class_names,
    )
    mapping_keys = sorted(str(item).strip().upper() for item in mapping_by_name.keys())
    if mapping_keys != sorted(expected_class_names):
        result.fail(
            "actuation mapping keys mismatch expected classes: "
            f"mapping={mapping_keys}, expected={sorted(expected_class_names)}"
        )

    relax_state = label_to_state.get(0)
    release_labels = sorted(
        int(label)
        for label, state in label_to_state.items()
        if int(label) != 0 and state == relax_state
    )
    release_mode = str(runtime_cfg.runtime.release_mode).strip().lower()
    if release_mode == "command_only" and not release_labels:
        result.fail("release_mode=command_only but no non-RELAX label is mapped to RELAX actuator state")

    split_manifest_raw = str(args.split_manifest or "").strip() or str(train_data_cfg.split_manifest_path)
    split_manifest_path = _resolve_code_path(split_manifest_raw)
    if not split_manifest_path.exists():
        result.fail(f"split manifest missing: {split_manifest_path}")
    else:
        manifest = load_manifest(str(split_manifest_path))
        train_sources = set(manifest.train_sources)
        val_sources = set(manifest.val_sources)
        test_sources = set(manifest.test_sources)
        if not train_sources:
            result.fail("manifest has empty train_sources")
        if not val_sources:
            result.fail("manifest has empty val_sources")
        if not test_sources:
            result.fail("manifest has empty test_sources")
        if train_sources & val_sources:
            result.fail(f"manifest leakage: train/val overlap count={len(train_sources & val_sources)}")
        if train_sources & test_sources:
            result.fail(f"manifest leakage: train/test overlap count={len(train_sources & test_sources)}")
        if val_sources & test_sources:
            result.fail(f"manifest leakage: val/test overlap count={len(val_sources & test_sources)}")
        result.details["split_counts"] = {
            "train_sources": len(train_sources),
            "val_sources": len(val_sources),
            "test_sources": len(test_sources),
            "train_indices": len(manifest.train_indices),
            "val_indices": len(manifest.val_indices),
            "test_indices": len(manifest.test_indices),
        }

    result.details.update(
        {
            "training_config": str(args.training_config),
            "runtime_config": str(args.runtime_config),
            "split_manifest": str(split_manifest_path),
            "expected_class_names": expected_class_names,
            "release_mode": release_mode,
            "release_command_labels": release_labels,
        }
    )
    return result


def _check_run_artifacts(
    *,
    run_root: Path,
    run_ids: list[str],
    expected_num_classes: int,
) -> CheckResult:
    result = CheckResult(name="implementation_artifacts")
    result.details["run_count"] = len(run_ids)
    result.details["runs_checked"] = list(run_ids)

    if not run_ids:
        result.fail("no runs found from screen/longrun summaries")
        return result

    for run_id in run_ids:
        run_dir = run_root / str(run_id)
        if not run_dir.exists():
            result.fail(f"missing run dir: {run_dir}")
            continue
        offline_path = run_dir / "offline_summary.json"
        test_metrics_path = run_dir / "evaluation" / "test_metrics.json"
        control_eval_path = run_dir / "evaluation" / "control_eval_summary.json"
        overrides_path = run_dir / "config_snapshots" / "effective_overrides.yaml"

        if not offline_path.exists():
            result.fail(f"{run_id}: missing offline_summary.json")
            continue
        if not test_metrics_path.exists():
            result.fail(f"{run_id}: missing evaluation/test_metrics.json")
            continue

        offline = _load_json(offline_path)
        test_metrics = _load_json(test_metrics_path)
        event_acc_offline = float(offline.get("event_action_accuracy", 0.0) or 0.0)
        event_acc_eval = float(test_metrics.get("event_action_accuracy", 0.0) or 0.0)
        if abs(event_acc_offline - event_acc_eval) > 1e-6:
            result.fail(
                f"{run_id}: event_action_accuracy mismatch offline({event_acc_offline:.6f}) "
                f"vs eval({event_acc_eval:.6f})"
            )

        for metric_name in [
            "accuracy",
            "macro_f1",
            "macro_recall",
            "event_action_accuracy",
            "event_action_macro_f1",
        ]:
            if metric_name in test_metrics and not _is_finite_prob(test_metrics.get(metric_name)):
                result.fail(f"{run_id}: invalid metric {metric_name}={test_metrics.get(metric_name)!r}")

        checkpoint_path = Path(str(offline.get("checkpoint_path", "")).strip())
        if checkpoint_path and not checkpoint_path.is_absolute():
            checkpoint_path = (CODE_ROOT / checkpoint_path).resolve()
        if not checkpoint_path or not checkpoint_path.exists():
            result.fail(f"{run_id}: checkpoint missing -> {checkpoint_path}")

        if control_eval_path.exists():
            control = _load_json(control_eval_path)
            for metric_name in ["command_success_rate", "false_trigger_rate", "false_release_rate"]:
                if metric_name in control and not _is_finite_prob(control.get(metric_name)):
                    result.fail(f"{run_id}: invalid control metric {metric_name}={control.get(metric_name)!r}")
            total_count = int(control.get("total_clip_count", 0) or 0)
            action_count = int(control.get("action_clip_count", 0) or 0)
            relax_count = int(control.get("relax_clip_count", 0) or 0)
            release_count = int(control.get("release_command_clip_count", 0) or 0)
            if total_count > 0 and action_count + relax_count + release_count != total_count:
                result.fail(
                    f"{run_id}: clip counts inconsistent "
                    f"(action={action_count}, relax={relax_count}, release={release_count}, total={total_count})"
                )
        else:
            result.fail(f"{run_id}: missing evaluation/control_eval_summary.json")

        if not overrides_path.exists():
            result.fail(f"{run_id}: missing config_snapshots/effective_overrides.yaml")
        else:
            text = overrides_path.read_text(encoding="utf-8")
            if "num_classes" not in text:
                result.fail(f"{run_id}: overrides missing model.num_classes")
            elif f"num_classes: {int(expected_num_classes)}" not in text:
                result.fail(
                    f"{run_id}: overrides num_classes not {expected_num_classes} (check effective_overrides.yaml)"
                )

    return result


def _check_param_coverage(args: argparse.Namespace, *, screen_summary: dict, longrun_summary: dict) -> CheckResult:
    result = CheckResult(name="param_coverage")
    screen_rows = list(screen_summary.get("rows") or [])
    if not screen_rows:
        result.fail("screen summary has no rows")
        return result

    expected_loss = _parse_tokens(args.screen_loss_types)
    expected_base = _parse_int_tokens(args.screen_base_channels)
    expected_freeze = _parse_int_tokens(args.screen_freeze_emg_epochs)
    expected_elr = [float(item) for item in _parse_tokens(args.screen_encoder_lr_ratios)]
    expected_pt = _parse_tokens(args.screen_pretrained_modes)

    observed_pt = sorted({str(row.get("pretrained_mode", "")).strip().lower() for row in screen_rows})
    filtered_pt = [item for item in expected_pt if item.lower() in observed_pt]
    if not filtered_pt:
        filtered_pt = observed_pt

    expected_total = len(expected_loss) * len(expected_base) * len(expected_freeze) * len(expected_elr) * len(filtered_pt)
    result.details["screen_expected_rows"] = int(expected_total)
    result.details["screen_actual_rows"] = int(len(screen_rows))
    if len(screen_rows) != expected_total:
        result.fail(f"screen row count mismatch: expected={expected_total}, actual={len(screen_rows)}")

    tuples_seen = set()
    for row in screen_rows:
        key = (
            str(row.get("loss_type")),
            int(row.get("base_channels")),
            int(row.get("freeze_emg_epochs")),
            float(row.get("encoder_lr_ratio")),
            str(row.get("pretrained_mode")),
        )
        if key in tuples_seen:
            result.fail(f"duplicate screen candidate: {key}")
        tuples_seen.add(key)

    longrun_rows = list(longrun_summary.get("rows") or [])
    if not longrun_rows:
        result.fail("longrun summary has no rows")
        return result
    expected_seeds = _parse_int_tokens(args.longrun_seeds)
    candidate_ranks = sorted({int(row.get("candidate_rank")) for row in longrun_rows})
    result.details["longrun_candidate_ranks"] = candidate_ranks
    result.details["longrun_expected_seeds"] = expected_seeds

    for rank in candidate_ranks:
        rows = [row for row in longrun_rows if int(row.get("candidate_rank")) == rank]
        seeds = sorted(int(row.get("split_seed")) for row in rows)
        if seeds != sorted(expected_seeds):
            result.fail(f"candidate_rank={rank} seed coverage mismatch: expected={sorted(expected_seeds)}, got={seeds}")

    best_run_id = str(longrun_summary.get("best_run_id", "") or "")
    if best_run_id and best_run_id not in {str(row.get("run_id")) for row in longrun_rows}:
        result.fail(f"best_run_id not found in longrun rows: {best_run_id}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit model-line sprint pipeline")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="s2_model90")
    parser.add_argument("--training_config", default="configs/training_event_onset_demo_p0.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--split_manifest", default=None)

    parser.add_argument("--screen_loss_types", default="cross_entropy,cb_focal")
    parser.add_argument("--screen_base_channels", default="16,24")
    parser.add_argument("--screen_freeze_emg_epochs", default="5,8")
    parser.add_argument("--screen_encoder_lr_ratios", default="0.3,0.2")
    parser.add_argument("--screen_pretrained_modes", default="off,on")
    parser.add_argument("--longrun_seeds", default="42,52,62")

    parser.add_argument("--output_json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = _resolve_code_path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    screen_path = run_root / f"{args.run_prefix}_screen_summary.json"
    longrun_path = run_root / f"{args.run_prefix}_longrun_summary.json"

    report: dict[str, Any] = {
        "status": "ok",
        "run_root": str(run_root),
        "run_prefix": str(args.run_prefix),
        "checks": {},
    }

    design_result = _check_design_contract(args)
    report["checks"]["design"] = asdict(design_result)

    expected_num_classes = len(design_result.details.get("expected_class_names") or [])
    screen_summary = _load_json(screen_path) if screen_path.exists() else {}
    longrun_summary = _load_json(longrun_path) if longrun_path.exists() else {}

    run_ids = sorted(
        {
            str(row.get("run_id"))
            for row in (screen_summary.get("rows") or []) + (longrun_summary.get("rows") or [])
            if str(row.get("run_id", "")).strip()
        }
    )
    implementation_result = _check_run_artifacts(
        run_root=run_root,
        run_ids=run_ids,
        expected_num_classes=int(expected_num_classes) if expected_num_classes else 0,
    )
    report["checks"]["implementation"] = asdict(implementation_result)

    if not screen_summary:
        param_result = CheckResult(name="param_coverage", passed=False, issues=[f"missing {screen_path}"])
    elif not longrun_summary:
        param_result = CheckResult(name="param_coverage", passed=False, issues=[f"missing {longrun_path}"])
    else:
        param_result = _check_param_coverage(args, screen_summary=screen_summary, longrun_summary=longrun_summary)
    report["checks"]["params"] = asdict(param_result)

    blocking_issues: list[str] = []
    for check_name in ["design", "implementation", "params"]:
        payload = report["checks"][check_name]
        if not bool(payload.get("passed", False)):
            for issue in payload.get("issues", []):
                blocking_issues.append(f"{check_name}: {issue}")

    report["blocking_issue_count"] = len(blocking_issues)
    report["blocking_issues"] = blocking_issues
    report["data_only_ready"] = len(blocking_issues) == 0
    if blocking_issues:
        report["status"] = "failed"

    output_path = (
        _resolve_code_path(args.output_json)
        if str(args.output_json or "").strip()
        else (run_root / f"{args.run_prefix}_audit_report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

