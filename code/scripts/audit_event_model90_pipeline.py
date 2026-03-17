
"""Audit model-line pipeline to rule out design/implementation/parameter issues."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.config import load_event_runtime_config, load_event_training_config
from shared.event_labels import normalize_event_label_input, public_event_labels
from shared.label_modes import get_label_mode_spec
from training.data.split_strategy import load_manifest

TARGET_CLASS_ORDER = ["CONTINUE", "TENSE_OPEN", "THUMB_UP", "WRIST_CW"]


def _parse_tokens(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_int_tokens(raw: str) -> list[int]:
    return [int(item) for item in _parse_tokens(raw)]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be object: {path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.exists() else {}


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


def _metric_close(lhs: float, rhs: float, *, atol: float = 1e-6) -> bool:
    return abs(float(lhs) - float(rhs)) <= float(atol)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str) and not value.strip():
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _format_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _rank_row(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(row.get("event_action_accuracy", 0.0)),
        float(row.get("event_action_macro_f1", 0.0)),
        float(row.get("command_success_rate", 0.0)),
        -float(row.get("false_trigger_rate", 1.0)),
        -float(row.get("false_release_rate", 1.0)),
        float(row.get("test_accuracy", 0.0)),
    )


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
    expected_class_names = [normalize_event_label_input(item) for item in label_spec.class_names]
    model_cfg.num_classes = int(len(expected_class_names))
    if public_event_labels(expected_class_names) != TARGET_CLASS_ORDER:
        result.fail(
            f"label contract mismatch: expected={TARGET_CLASS_ORDER}, actual={public_event_labels(expected_class_names)}"
        )

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
        result.fail("release_mode=command_only but no non-CONTINUE label maps to CONTINUE/RELAX actuator state")

    tense_open_idx = expected_class_names.index("TENSE_OPEN") if "TENSE_OPEN" in expected_class_names else -1
    if release_mode == "command_only" and tense_open_idx >= 0 and tense_open_idx not in release_labels:
        result.fail("release_mode=command_only expects TENSE_OPEN to map to CONTINUE/RELAX state")

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
            "expected_class_names": public_event_labels(expected_class_names),
            "release_mode": release_mode,
            "release_command_labels": release_labels,
        }
    )
    return result


def _extract_run_ids(*summaries: dict[str, Any]) -> list[str]:
    run_ids: set[str] = set()
    for summary in summaries:
        for row in list(summary.get("rows") or []):
            run_id = str(row.get("run_id", "")).strip()
            if run_id:
                run_ids.add(run_id)
    return sorted(run_ids)

def _check_run_artifacts(
    *,
    run_root: Path,
    run_ids: list[str],
    expected_num_classes: int,
    metric_tolerance: float,
) -> CheckResult:
    result = CheckResult(name="implementation_artifacts")
    result.details["run_count"] = len(run_ids)
    result.details["runs_checked"] = list(run_ids)

    if not run_ids:
        result.fail("no runs found from screen/longrun/neighbor summaries")
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
        run_metadata_path = run_dir / "run_metadata.json"

        if not offline_path.exists():
            result.fail(f"{run_id}: missing offline_summary.json")
            continue
        if not test_metrics_path.exists():
            result.fail(f"{run_id}: missing evaluation/test_metrics.json")
            continue

        offline = _load_json(offline_path)
        test_metrics = _load_json(test_metrics_path)

        metric_pairs = [
            ("test_accuracy", "accuracy"),
            ("test_macro_f1", "macro_f1"),
            ("event_action_accuracy", "event_action_accuracy"),
            ("event_action_macro_f1", "event_action_macro_f1"),
        ]
        for left_key, right_key in metric_pairs:
            if left_key not in offline or right_key not in test_metrics:
                continue
            left = float(offline.get(left_key, 0.0) or 0.0)
            right = float(test_metrics.get(right_key, 0.0) or 0.0)
            if not _metric_close(left, right, atol=metric_tolerance):
                result.fail(
                    f"{run_id}: metric mismatch {left_key}={left:.6f} vs {right_key}={right:.6f}"
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
            continue_count = int(control.get("continue_clip_count", control.get("relax_clip_count", 0)) or 0)
            release_count = int(control.get("release_command_clip_count", 0) or 0)
            if total_count > 0 and action_count + continue_count + release_count != total_count:
                result.fail(
                    f"{run_id}: clip counts inconsistent "
                    f"(action={action_count}, continue={continue_count}, release={release_count}, total={total_count})"
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
            if "device_target" not in text:
                result.fail(f"{run_id}: overrides missing device_target evidence")
            if "device_id" not in text:
                result.fail(f"{run_id}: overrides missing device_id evidence")

        if not run_metadata_path.exists():
            result.fail(f"{run_id}: missing run_metadata.json")
        else:
            run_metadata = _load_json(run_metadata_path)
            device_info = dict(run_metadata.get("training_device") or {})
            if not str(device_info.get("target", "")).strip():
                result.fail(f"{run_id}: run_metadata missing training_device.target")
            if "id" not in device_info:
                result.fail(f"{run_id}: run_metadata missing training_device.id")
            recordings_manifest_path = str(run_metadata.get("recordings_manifest_path", "")).strip()
            if not recordings_manifest_path:
                result.fail(f"{run_id}: run_metadata missing recordings_manifest_path")
            else:
                resolved_recordings_manifest = _resolve_code_path(recordings_manifest_path)
                if not resolved_recordings_manifest.exists():
                    result.fail(
                        f"{run_id}: recordings_manifest_path missing -> {resolved_recordings_manifest}"
                    )

            split_manifest_path = str(run_metadata.get("split_manifest_path", "")).strip()
            offline_manifest_path = str(offline.get("manifest_path", "")).strip()
            if not split_manifest_path:
                result.fail(f"{run_id}: run_metadata missing split_manifest_path")
            else:
                resolved_split_manifest = _resolve_code_path(split_manifest_path)
                if not resolved_split_manifest.exists():
                    result.fail(f"{run_id}: split_manifest_path missing -> {resolved_split_manifest}")
                if offline_manifest_path:
                    resolved_offline_manifest = _resolve_code_path(offline_manifest_path)
                    if resolved_split_manifest != resolved_offline_manifest:
                        result.fail(
                            f"{run_id}: split manifest mismatch "
                            f"run_metadata={resolved_split_manifest} vs offline={resolved_offline_manifest}"
                        )

            quality_report_path = str(run_metadata.get("quality_report", "")).strip()
            if not quality_report_path:
                result.fail(f"{run_id}: run_metadata missing quality_report")
            else:
                resolved_quality_report = _resolve_code_path(quality_report_path)
                if not resolved_quality_report.exists():
                    result.fail(f"{run_id}: quality_report missing -> {resolved_quality_report}")

            evaluation_outputs = dict(run_metadata.get("evaluation_outputs") or {})
            if not evaluation_outputs:
                result.fail(f"{run_id}: run_metadata missing evaluation_outputs")
            else:
                for name, raw_path in evaluation_outputs.items():
                    resolved_output = _resolve_code_path(str(raw_path))
                    if not resolved_output.exists():
                        result.fail(f"{run_id}: evaluation output missing {name} -> {resolved_output}")

            class_names = list(run_metadata.get("class_names") or [])
            if not class_names:
                result.fail(f"{run_id}: run_metadata missing class_names")
            elif expected_num_classes > 0 and len(class_names) != int(expected_num_classes):
                result.fail(
                    f"{run_id}: class_names count mismatch "
                    f"(expected={expected_num_classes}, actual={len(class_names)})"
                )
            elif len({str(item) for item in class_names}) != len(class_names):
                result.fail(f"{run_id}: class_names contains duplicates -> {class_names}")

            model_variant = str(run_metadata.get("model_variant", "")).strip()
            if model_variant == "event_onset_two_stage_demo3":
                public_class_names = list(run_metadata.get("public_class_names") or [])
                gate_classes = list(run_metadata.get("gate_classes") or [])
                command_classes = list(run_metadata.get("command_classes") or [])
                if public_class_names != TARGET_CLASS_ORDER:
                    result.fail(
                        f"{run_id}: public_class_names mismatch -> {public_class_names} vs {TARGET_CLASS_ORDER}"
                    )
                if gate_classes != ["CONTINUE", "COMMAND"]:
                    result.fail(f"{run_id}: gate_classes mismatch -> {gate_classes}")
                if command_classes != ["TENSE_OPEN", "THUMB_UP", "WRIST_CW"]:
                    result.fail(f"{run_id}: command_classes mismatch -> {command_classes}")

    return result


def _check_param_coverage(
    args: argparse.Namespace,
    *,
    screen_summary: dict[str, Any],
    longrun_summary: dict[str, Any],
    neighbor_summary: dict[str, Any],
) -> CheckResult:
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

    if not neighbor_summary:
        result.fail("missing neighbor summary")
        return result

    neighbor_rows = list(neighbor_summary.get("rows") or [])
    result.details["neighbor_row_count"] = int(len(neighbor_rows))
    if not neighbor_rows:
        result.fail("neighbor summary has no rows")
        return result

    significant = bool(neighbor_summary.get("significant_improvement_found", False))
    result.details["neighbor_significant_improvement"] = significant
    result.details["neighbor_event_gain"] = float(neighbor_summary.get("event_action_accuracy_gain", 0.0) or 0.0)
    result.details["neighbor_command_gain"] = float(neighbor_summary.get("command_success_rate_gain", 0.0) or 0.0)
    if significant:
        result.fail("neighbor tuning found significant improvement; parameter space is not exhausted")

    return result

def _choose_stability_run_id(
    *,
    args: argparse.Namespace,
    screen_summary: dict[str, Any],
    longrun_summary: dict[str, Any],
) -> str:
    explicit = str(args.stability_run_id or "").strip()
    if explicit:
        return explicit
    best_longrun = str(longrun_summary.get("best_run_id", "")).strip()
    if best_longrun:
        return best_longrun
    screen_rows = list(screen_summary.get("rows") or [])
    if not screen_rows:
        return ""
    return str(sorted(screen_rows, key=_rank_row, reverse=True)[0].get("run_id", "")).strip()


def _run_control_eval_once(
    *,
    args: argparse.Namespace,
    run_id: str,
    split_manifest: str,
    recordings_manifest: str,
    output_json: Path,
) -> tuple[dict[str, Any], str]:
    cmd = [
        sys.executable,
        "scripts/evaluate_event_demo_control.py",
        "--run_root",
        str(args.run_root),
        "--run_id",
        str(run_id),
        "--training_config",
        str(args.training_config),
        "--runtime_config",
        str(args.runtime_config),
        "--data_dir",
        str(args.data_dir),
        "--split_manifest",
        str(split_manifest),
        "--target_db5_keys",
        str(args.target_db5_keys),
        "--backend",
        str(args.control_backend),
        "--device_target",
        str(args.device_target),
        "--output_json",
        str(output_json),
    ]
    if str(recordings_manifest).strip():
        cmd.extend(["--recordings_manifest", str(recordings_manifest)])

    completed = subprocess.run(
        cmd,
        cwd=str(CODE_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr[-1500:] if completed.stderr else ""
        raise RuntimeError(
            f"control eval rerun failed (run_id={run_id}, rc={completed.returncode}): {stderr}"
        )

    payload = _load_json(output_json)
    return payload, _format_cmd(cmd)


def _check_eval_stability(
    args: argparse.Namespace,
    *,
    run_root: Path,
    screen_summary: dict[str, Any],
    longrun_summary: dict[str, Any],
) -> CheckResult:
    result = CheckResult(name="evaluation_stability")
    repeats = max(1, int(args.stability_repeats))
    tolerance = float(args.stability_tolerance)

    if bool(args.skip_stability_check):
        result.details["skipped"] = True
        result.details["reason"] = "--skip_stability_check"
        return result

    if repeats <= 1:
        result.details["skipped"] = True
        result.details["reason"] = "stability_repeats<=1"
        return result

    run_id = _choose_stability_run_id(args=args, screen_summary=screen_summary, longrun_summary=longrun_summary)
    if not run_id:
        result.fail("cannot determine stability run_id")
        return result

    run_dir = run_root / run_id
    if not run_dir.exists():
        result.fail(f"stability run dir missing: {run_dir}")
        return result

    control_eval_path = run_dir / "evaluation" / "control_eval_summary.json"
    offline_summary_path = run_dir / "offline_summary.json"
    control_eval = _load_optional_json(control_eval_path)
    offline_summary = _load_optional_json(offline_summary_path)

    split_manifest = str(control_eval.get("split_manifest", "")).strip()
    if not split_manifest:
        split_manifest = str(offline_summary.get("manifest_path", "")).strip()
    if not split_manifest:
        result.fail(f"{run_id}: missing split manifest for stability rerun")
        return result

    recordings_manifest = str(control_eval.get("recordings_manifest", "")).strip()
    if not recordings_manifest:
        recordings_manifest = str(args.recordings_manifest or "").strip()

    stability_dir = run_dir / "evaluation" / "stability_reruns"
    stability_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_repeat: list[dict[str, float]] = []
    command_text = ""
    for idx in range(1, repeats + 1):
        output_json = stability_dir / f"control_eval_repeat_{idx:02d}.json"
        payload, command_text = _run_control_eval_once(
            args=args,
            run_id=run_id,
            split_manifest=split_manifest,
            recordings_manifest=recordings_manifest,
            output_json=output_json,
        )
        metrics_by_repeat.append(
            {
                "command_success_rate": float(payload.get("command_success_rate", 0.0) or 0.0),
                "false_trigger_rate": float(payload.get("false_trigger_rate", 0.0) or 0.0),
                "false_release_rate": float(payload.get("false_release_rate", 0.0) or 0.0),
            }
        )

    spans: dict[str, float] = {}
    for key in ["command_success_rate", "false_trigger_rate", "false_release_rate"]:
        values = [row[key] for row in metrics_by_repeat]
        span = max(values) - min(values)
        spans[key] = float(span)
        if span > tolerance:
            result.fail(
                f"{run_id}: stability drift on {key} exceeds tolerance {tolerance} (span={span:.6f})"
            )

    result.details.update(
        {
            "run_id": run_id,
            "repeats": repeats,
            "tolerance": tolerance,
            "metrics_by_repeat": metrics_by_repeat,
            "metric_spans": spans,
            "command": command_text,
        }
    )
    return result


def _choose_best_reference_row(*summaries: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(list(summary.get("rows") or []))
    if not rows:
        return {}
    return dict(sorted(rows, key=_rank_row, reverse=True)[0])


def _load_runtime_tuned_command_metrics(args: argparse.Namespace) -> dict[str, Any]:
    runtime_path = _resolve_code_path(args.runtime_tuning_summary)
    tune_summary_path = _resolve_code_path(args.tune_summary)
    runtime_summary = _load_optional_json(runtime_path)
    if not runtime_summary:
        return {}

    best_row = dict(runtime_summary.get("best") or {})
    if not best_row:
        return {}

    best_run_id = ""
    tune_summary = _load_optional_json(tune_summary_path)
    if tune_summary:
        best_run_id = str(tune_summary.get("best_run_id", "")).strip()

    return {
        "run_id": best_run_id,
        "command_success_rate": _float_or_default(best_row.get("command_success_rate"), 0.0),
        "false_trigger_rate": _float_or_default(best_row.get("false_trigger_rate"), 1.0),
        "false_release_rate": _float_or_default(best_row.get("false_release_rate"), 1.0),
        "source": str(runtime_path),
    }


def _assess_goal_and_conclusion(
    args: argparse.Namespace,
    *,
    data_only_ready: bool,
    blocking_issues: list[str],
    screen_summary: dict[str, Any],
    longrun_summary: dict[str, Any],
    neighbor_summary: dict[str, Any],
) -> dict[str, Any]:
    reference = _choose_best_reference_row(longrun_summary, neighbor_summary, screen_summary)
    runtime_tuned = _load_runtime_tuned_command_metrics(args)

    event_action_accuracy = _float_or_default(reference.get("event_action_accuracy"), 0.0)
    event_action_macro_f1 = _float_or_default(reference.get("event_action_macro_f1"), 0.0)

    command_success_rate = _float_or_default(reference.get("command_success_rate"), 0.0)
    false_trigger_rate = _float_or_default(reference.get("false_trigger_rate"), 1.0)
    false_release_rate = _float_or_default(reference.get("false_release_rate"), 1.0)
    command_metric_source = f"run:{reference.get('run_id', '')}"

    if runtime_tuned:
        command_success_rate = float(runtime_tuned["command_success_rate"])
        false_trigger_rate = float(runtime_tuned["false_trigger_rate"])
        false_release_rate = float(runtime_tuned["false_release_rate"])
        command_metric_source = f"runtime_tuned:{runtime_tuned.get('source', '')}"

    dev_pass = bool(
        event_action_accuracy >= float(args.target_event_action_accuracy)
        and event_action_macro_f1 >= float(args.target_event_action_macro_f1)
    )
    demo_pass = bool(
        command_success_rate >= float(args.target_command_success_rate)
        and false_trigger_rate <= float(args.max_false_trigger_rate)
        and false_release_rate <= float(args.max_false_release_rate)
    )

    if blocking_issues:
        conclusion = "engineering_gates_not_cleared"
    elif dev_pass and demo_pass:
        conclusion = "target_metrics_achieved"
    else:
        conclusion = "data_bottleneck_only"

    return {
        "conclusion": conclusion,
        "data_only_bottleneck": bool(data_only_ready and not (dev_pass and demo_pass)),
        "development_gate": {
            "event_action_accuracy": event_action_accuracy,
            "event_action_macro_f1": event_action_macro_f1,
            "target_event_action_accuracy": float(args.target_event_action_accuracy),
            "target_event_action_macro_f1": float(args.target_event_action_macro_f1),
            "passed": dev_pass,
            "source_run_id": str(reference.get("run_id", "")),
        },
        "demo_gate": {
            "command_success_rate": command_success_rate,
            "false_trigger_rate": false_trigger_rate,
            "false_release_rate": false_release_rate,
            "target_command_success_rate": float(args.target_command_success_rate),
            "max_false_trigger_rate": float(args.max_false_trigger_rate),
            "max_false_release_rate": float(args.max_false_release_rate),
            "passed": demo_pass,
            "source": command_metric_source,
        },
    }


def _categorize_param_failure(param_result: CheckResult) -> tuple[str, str]:
    issues = [str(item).strip() for item in param_result.issues if str(item).strip()]
    lowered = [item.lower() for item in issues]
    if any("missing " in item or "no rows" in item for item in lowered):
        return (
            "implementation_bug",
            "parameter coverage evidence is incomplete, so the pipeline cannot prove the search ran correctly",
        )
    return (
        "hyperparameter_underfit",
        "the constrained search space is not exhausted, so model quality cannot yet be blamed on data alone",
    )


def _categorize_root_cause(
    *,
    design_result: CheckResult,
    implementation_result: CheckResult,
    param_result: CheckResult,
    stability_result: CheckResult,
    goal_assessment: dict[str, Any],
) -> dict[str, Any]:
    if not design_result.passed:
        return {
            "root_cause_category": "artifact_contract_bug",
            "root_cause_summary": "design contract failed before model quality could be evaluated",
            "root_cause_check": design_result.name,
            "root_cause_evidence": list(design_result.issues),
        }

    if not implementation_result.passed:
        return {
            "root_cause_category": "implementation_bug",
            "root_cause_summary": "run artifacts are incomplete or internally inconsistent",
            "root_cause_check": implementation_result.name,
            "root_cause_evidence": list(implementation_result.issues),
        }

    if not stability_result.passed:
        return {
            "root_cause_category": "implementation_bug",
            "root_cause_summary": "evaluation reruns are not stable enough to trust the reported metrics",
            "root_cause_check": stability_result.name,
            "root_cause_evidence": list(stability_result.issues),
        }

    if not param_result.passed:
        category, summary = _categorize_param_failure(param_result)
        return {
            "root_cause_category": category,
            "root_cause_summary": summary,
            "root_cause_check": param_result.name,
            "root_cause_evidence": list(param_result.issues),
        }

    conclusion = str(goal_assessment.get("conclusion", "")).strip().lower()
    if conclusion == "target_metrics_achieved":
        return {
            "root_cause_category": "none",
            "root_cause_summary": "target metrics achieved; no blocking root cause remains",
            "root_cause_check": "",
            "root_cause_evidence": [],
        }

    return {
        "root_cause_category": "data_bottleneck",
        "root_cause_summary": (
            "engineering gates are clear and the remaining gap is primarily data quality, "
            "coverage, or label separability"
        ),
        "root_cause_check": "goal_assessment",
        "root_cause_evidence": [
            f"development_gate.passed={bool(goal_assessment.get('development_gate', {}).get('passed', False))}",
            f"demo_gate.passed={bool(goal_assessment.get('demo_gate', {}).get('passed', False))}",
            f"conclusion={goal_assessment.get('conclusion', '')}",
        ],
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit model-line sprint pipeline")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_prefix", default="s2_model90")
    parser.add_argument("--training_config", default="configs/training_event_onset_demo3_two_stage.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo3_latch.yaml")
    parser.add_argument("--split_manifest", default=None)

    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default="")
    parser.add_argument("--target_db5_keys", default="TENSE_OPEN,THUMB_UP,WRIST_CW")
    parser.add_argument("--control_backend", default="ckpt", choices=["ckpt", "lite"])
    parser.add_argument("--device_target", default="GPU", choices=["CPU", "GPU", "Ascend"])

    parser.add_argument("--screen_loss_types", default="cross_entropy,cb_focal")
    parser.add_argument("--screen_base_channels", default="16,24")
    parser.add_argument("--screen_freeze_emg_epochs", default="6,8,10")
    parser.add_argument("--screen_encoder_lr_ratios", default="0.24,0.3,0.36")
    parser.add_argument("--screen_pretrained_modes", default="off")
    parser.add_argument("--longrun_seeds", default="42,52,62")

    parser.add_argument("--neighbor_summary", default=None)
    parser.add_argument("--runtime_tuning_summary", default=None)
    parser.add_argument("--tune_summary", default=None)

    parser.add_argument("--metric_tolerance", type=float, default=1e-6)
    parser.add_argument("--stability_repeats", type=int, default=2)
    parser.add_argument("--stability_tolerance", type=float, default=1e-6)
    parser.add_argument("--stability_run_id", default="")
    parser.add_argument("--skip_stability_check", action="store_true")

    parser.add_argument("--target_event_action_accuracy", type=float, default=0.90)
    parser.add_argument("--target_event_action_macro_f1", type=float, default=0.88)
    parser.add_argument("--target_command_success_rate", type=float, default=0.90)
    parser.add_argument("--max_false_trigger_rate", type=float, default=0.05)
    parser.add_argument("--max_false_release_rate", type=float, default=0.05)

    parser.add_argument("--output_json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = _resolve_code_path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    screen_path = run_root / f"{args.run_prefix}_screen_summary.json"
    longrun_path = run_root / f"{args.run_prefix}_longrun_summary.json"
    neighbor_path = (
        _resolve_code_path(args.neighbor_summary)
        if str(args.neighbor_summary or "").strip()
        else (run_root / f"{args.run_prefix}_neighbor_summary.json")
    )

    if not str(args.runtime_tuning_summary or "").strip():
        args.runtime_tuning_summary = str(run_root / f"{args.run_prefix}_runtime_threshold_tuning_summary.json")
    if not str(args.tune_summary or "").strip():
        args.tune_summary = str(run_root / f"{args.run_prefix}_tune_summary.json")

    report: dict[str, Any] = {
        "status": "ok",
        "run_root": str(run_root),
        "run_prefix": str(args.run_prefix),
        "checks": {},
    }

    screen_summary = _load_optional_json(screen_path)
    longrun_summary = _load_optional_json(longrun_path)
    neighbor_summary = _load_optional_json(neighbor_path)

    design_result = _check_design_contract(args)
    report["checks"]["design"] = asdict(design_result)

    expected_num_classes = len(design_result.details.get("expected_class_names") or [])
    run_ids = _extract_run_ids(screen_summary, longrun_summary, neighbor_summary)
    implementation_result = _check_run_artifacts(
        run_root=run_root,
        run_ids=run_ids,
        expected_num_classes=int(expected_num_classes) if expected_num_classes else 0,
        metric_tolerance=float(args.metric_tolerance),
    )
    report["checks"]["implementation"] = asdict(implementation_result)

    if not screen_summary:
        param_result = CheckResult(name="param_coverage", passed=False, issues=[f"missing {screen_path}"])
    elif not longrun_summary:
        param_result = CheckResult(name="param_coverage", passed=False, issues=[f"missing {longrun_path}"])
    elif not neighbor_summary:
        param_result = CheckResult(name="param_coverage", passed=False, issues=[f"missing {neighbor_path}"])
    else:
        param_result = _check_param_coverage(
            args,
            screen_summary=screen_summary,
            longrun_summary=longrun_summary,
            neighbor_summary=neighbor_summary,
        )
    report["checks"]["params"] = asdict(param_result)

    stability_result = _check_eval_stability(
        args,
        run_root=run_root,
        screen_summary=screen_summary,
        longrun_summary=longrun_summary,
    )
    report["checks"]["stability"] = asdict(stability_result)

    blocking_issues: list[str] = []
    for check_name in ["design", "implementation", "params", "stability"]:
        payload = report["checks"][check_name]
        if not bool(payload.get("passed", False)):
            for issue in payload.get("issues", []):
                blocking_issues.append(f"{check_name}: {issue}")

    report["blocking_issue_count"] = len(blocking_issues)
    report["blocking_issues"] = blocking_issues
    report["data_only_ready"] = len(blocking_issues) == 0

    goal_assessment = _assess_goal_and_conclusion(
        args,
        data_only_ready=bool(report["data_only_ready"]),
        blocking_issues=blocking_issues,
        screen_summary=screen_summary,
        longrun_summary=longrun_summary,
        neighbor_summary=neighbor_summary,
    )
    report["goal_assessment"] = goal_assessment
    report["data_only_bottleneck"] = bool(goal_assessment.get("data_only_bottleneck", False))
    report["conclusion"] = str(goal_assessment.get("conclusion", "engineering_gates_not_cleared"))
    report.update(
        _categorize_root_cause(
            design_result=design_result,
            implementation_result=implementation_result,
            param_result=param_result,
            stability_result=stability_result,
            goal_assessment=goal_assessment,
        )
    )

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
