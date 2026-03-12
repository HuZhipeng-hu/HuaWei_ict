"""Strict quality gate for DB5 pretraining matrix runs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


REQUIRED_SMOKE_FILES = (
    "db5_manifest.json",
    "db5_split_diagnostics.json",
    "db5_window_diagnostics.json",
)

PYTEST_SUITES: list[tuple[str, list[str]]] = [
    ("pytest_split", ["tests/test_split_strategy.py", "tests/test_split_strategy_v2.py"]),
    ("pytest_db5", ["tests/test_db5_dataset.py", "tests/test_db5_pretrain_matrix.py"]),
    ("pytest_training_stability", ["tests/test_trainer_learning_rate.py", "tests/test_trainer_balanced_sampler.py"]),
]


@dataclass
class StageResult:
    stage: str
    passed: bool
    return_code: int
    warning_count: int
    error_count: int
    elapsed_seconds: float
    command: str
    log_path: str
    message: str
    next_command: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict DB5 pretraining quality gate.")
    parser.add_argument("--config", default="configs/pretrain_ninapro_db5.yaml")
    parser.add_argument("--data_dir", default="../data_ninaproDB5")
    parser.add_argument("--wearer_data_dir", default="../data")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--foundation_dir", default="artifacts/foundation/db5_full53")
    parser.add_argument("--run_prefix", default="db5_qgate")
    parser.add_argument("--pytest_basetemp_root", default=".tmp_pytest_runcheck")
    parser.add_argument("--skip_db5_probe", action="store_true")
    parser.add_argument("--skip_budget_probe", action="store_true")
    parser.add_argument(
        "--fail_on_warning",
        choices=["true", "false"],
        default="true",
        help="Fail gate if any stage output contains [WARN].",
    )
    return parser.parse_args()


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _count_log_markers(output: str) -> tuple[int, int]:
    warns = len(re.findall(r"^\[WARN\]", output, flags=re.MULTILINE))
    errs = len(re.findall(r"^\[ERROR\]", output, flags=re.MULTILINE))
    return warns, errs


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_command(stage: str, cmd: Sequence[str], *, logs_dir: Path, cwd: Path) -> StageResult:
    started = time.time()
    command = _format_cmd(cmd)
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started
    output = (completed.stdout or "") + ("\n" if completed.stderr else "") + (completed.stderr or "")
    warns, errs = _count_log_markers(output)
    log_path = logs_dir / f"{stage}.log"
    _write_text(log_path, output)
    return StageResult(
        stage=stage,
        passed=bool(completed.returncode == 0 and errs == 0),
        return_code=int(completed.returncode),
        warning_count=int(warns),
        error_count=int(errs),
        elapsed_seconds=float(elapsed),
        command=command,
        log_path=str(log_path),
        message="",
        next_command=command,
    )


def _group_leakage_present(manifest: dict) -> bool:
    train = set(manifest.get("group_keys_train", []) or [])
    val = set(manifest.get("group_keys_val", []) or [])
    test = set(manifest.get("group_keys_test", []) or [])
    return bool((train & val) or (train & test) or (val & test))


def _validate_smoke_outputs(smoke_dir: Path) -> list[str]:
    issues: list[str] = []
    for name in REQUIRED_SMOKE_FILES:
        if not (smoke_dir / name).exists():
            issues.append(f"missing artifact: {smoke_dir / name}")
    if issues:
        return issues

    manifest = json.loads((smoke_dir / "db5_manifest.json").read_text(encoding="utf-8"))
    split_diag = json.loads((smoke_dir / "db5_split_diagnostics.json").read_text(encoding="utf-8"))
    if _group_leakage_present(manifest):
        issues.append("group leakage detected in manifest")

    overall = dict(split_diag.get("overall", {}) or {})
    if bool(overall.get("has_any_empty_classes", False)):
        issues.append("split diagnostics reports empty classes")

    by_split = dict(split_diag.get("by_split", {}) or {})
    for split_name in ("train", "val", "test"):
        payload = dict(by_split.get(split_name, {}) or {})
        class_counts = dict(payload.get("class_counts", {}) or {})
        empty = [name for name, value in class_counts.items() if int(value) <= 0]
        if empty:
            issues.append(f"{split_name} has empty classes: {','.join(empty)}")
    return issues


def _stage_table(results: list[StageResult]) -> str:
    lines = [
        "| stage | status | rc | warnings | errors | seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| "
            f"{item.stage} | {'PASS' if item.passed else 'FAIL'} | {item.return_code} | "
            f"{item.warning_count} | {item.error_count} | {item.elapsed_seconds:.2f} |"
        )
    return "\n".join(lines)


def _write_reports(
    *,
    report_dir: Path,
    args: argparse.Namespace,
    results: list[StageResult],
    status: str,
    failure: StageResult | None,
) -> None:
    payload = {
        "status": status,
        "generated_at_unix": int(time.time()),
        "code_root": str(CODE_ROOT),
        "run_root": str(args.run_root),
        "run_prefix": str(args.run_prefix),
        "fail_on_warning": bool(str(args.fail_on_warning).strip().lower() == "true"),
        "stages": [asdict(item) for item in results],
        "failure": asdict(failure) if failure is not None else None,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "db5_quality_gate_report.json"
    report_md = report_dir / "db5_quality_gate_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DB5 Pretrain Quality Gate Report",
        "",
        f"- status: `{status}`",
        f"- fail_on_warning: `{payload['fail_on_warning']}`",
        "",
        "## Stage Results",
        _stage_table(results),
        "",
    ]
    if failure is not None:
        lines.extend(
            [
                "## Failure",
                f"- stage: `{failure.stage}`",
                f"- root_cause: `{failure.message}`",
                f"- next_command: `{failure.next_command}`",
                f"- log: `{failure.log_path}`",
                "",
            ]
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[QGATE] report_json={report_json}")
    print(f"[QGATE] report_md={report_md}")


def _enforce_stage(result: StageResult, *, fail_on_warning: bool) -> StageResult:
    if result.return_code != 0:
        result.passed = False
        result.message = f"subprocess failed with code {result.return_code}"
        return result
    if result.error_count > 0:
        result.passed = False
        result.message = f"log contains {result.error_count} [ERROR] marker(s)"
        return result
    if fail_on_warning and result.warning_count > 0:
        result.passed = False
        result.message = f"log contains {result.warning_count} [WARN] marker(s)"
        return result
    result.passed = True
    result.message = "ok"
    return result


def main() -> int:
    args = _parse_args()
    fail_on_warning = bool(str(args.fail_on_warning).strip().lower() == "true")
    report_dir = Path(args.run_root) / f"{args.run_prefix}_quality_gate"
    logs_dir = report_dir / "logs"
    smoke_run_id = f"{args.run_prefix}_manifest_smoke"
    results: list[StageResult] = []

    def run_checked(stage: str, cmd: list[str]) -> None:
        result = _run_command(stage, cmd, logs_dir=logs_dir, cwd=CODE_ROOT)
        result = _enforce_stage(result, fail_on_warning=fail_on_warning)
        results.append(result)
        if not result.passed:
            _write_reports(report_dir=report_dir, args=args, results=results, status="failed", failure=result)
            raise SystemExit(1)

    preflight_base = [
        "--db5_data_dir",
        str(args.data_dir),
        "--wearer_data_dir",
        str(args.wearer_data_dir),
    ]
    if args.skip_db5_probe:
        preflight_base.append("--skip_db5_probe")
    if args.skip_budget_probe:
        preflight_base.append("--skip_budget_probe")

    run_checked(
        "preflight_local",
        [sys.executable, "scripts/preflight.py", "--mode", "local", *preflight_base],
    )
    run_checked(
        "preflight_ascend",
        [sys.executable, "scripts/preflight.py", "--mode", "ascend", *preflight_base],
    )

    for stage_name, files in PYTEST_SUITES:
        basetemp = f"{args.pytest_basetemp_root}_{stage_name}"
        run_checked(
            stage_name,
            [sys.executable, "-m", "pytest", *files, "-q", "--basetemp", basetemp],
        )

    smoke_cmd: list[str] = [
        sys.executable,
        "scripts/pretrain_ninapro_db5.py",
        "--config",
        str(args.config),
        "--data_dir",
        str(args.data_dir),
        "--run_root",
        str(args.run_root),
        "--run_id",
        smoke_run_id,
        "--device_target",
        str(args.device_target),
        "--device_id",
        str(int(args.device_id)),
        "--foundation_dir",
        str(args.foundation_dir),
        "--include_rest_class",
        "false",
        "--use_first_myo_only",
        "false",
        "--first_myo_channel_count",
        "16",
        "--lowcut_hz",
        "20",
        "--highcut_hz",
        "180",
        "--energy_min",
        "0.25",
        "--static_std_min",
        "0.08",
        "--clip_ratio_max",
        "0.08",
        "--saturation_abs",
        "126",
        "--manifest_use_source_metadata",
        "true",
        "--smoke_manifest_only",
    ]
    run_checked("manifest_smoke", smoke_cmd)

    smoke_dir = Path(args.run_root) / smoke_run_id
    issues = _validate_smoke_outputs(smoke_dir)
    artifact_stage = StageResult(
        stage="artifact_gate",
        passed=not issues,
        return_code=0,
        warning_count=0,
        error_count=int(len(issues)),
        elapsed_seconds=0.0,
        command=f"validate artifacts in {smoke_dir}",
        log_path=str(logs_dir / "artifact_gate.log"),
        message="ok" if not issues else "; ".join(issues),
        next_command=_format_cmd(smoke_cmd),
    )
    _write_text(Path(artifact_stage.log_path), artifact_stage.message)
    results.append(artifact_stage)
    if issues:
        _write_reports(report_dir=report_dir, args=args, results=results, status="failed", failure=artifact_stage)
        return 1

    _write_reports(report_dir=report_dir, args=args, results=results, status="passed", failure=None)
    print("[QGATE] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
