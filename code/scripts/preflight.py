"""Preflight checks for the event-onset production pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_runtime_config, load_event_training_config
from shared.config import load_config


@dataclass
class Check:
    level: str
    name: str
    detail: str


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_under_root(code_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return code_root / path


def collect_dependency_checks(mode: str) -> list[Check]:
    checks: list[Check] = []
    base_required = ("numpy", "scipy", "yaml")
    ascend_required = ("mindspore", "mindspore_lite")
    optional = ("pyserial", "smbus2", "pytest")

    for mod in base_required:
        checks.append(Check("INFO" if has_module(mod) else "ERROR", f"module:{mod}", "installed" if has_module(mod) else "missing"))

    for mod in ascend_required:
        if mode == "ascend":
            checks.append(Check("INFO" if has_module(mod) else "ERROR", f"module:{mod}", "installed" if has_module(mod) else "missing"))
        else:
            checks.append(Check("INFO" if has_module(mod) else "WARN", f"module:{mod}", "installed" if has_module(mod) else "missing (acceptable for local mode)"))

    for mod in optional:
        checks.append(Check("INFO" if has_module(mod) else "WARN", f"module:{mod}", "installed" if has_module(mod) else "missing (optional)"))

    return checks


def collect_file_checks(code_root: Path, data_root: Path) -> list[Check]:
    checks: list[Check] = []
    required_files = [
        code_root / "configs" / "pretrain_ninapro_db5.yaml",
        code_root / "configs" / "training_event_onset.yaml",
        code_root / "configs" / "conversion_event_onset.yaml",
        code_root / "configs" / "runtime_event_onset.yaml",
        code_root / "scripts" / "pretrain_ninapro_db5.py",
        code_root / "scripts" / "finetune_event_onset.py",
        code_root / "scripts" / "convert_event_onset.py",
        code_root / "scripts" / "run_event_runtime.py",
    ]
    for path in required_files:
        checks.append(Check("INFO" if path.exists() else "ERROR", f"file:{path.relative_to(code_root)}", "exists" if path.exists() else "missing"))

    checks.append(Check("INFO" if data_root.exists() else "WARN", f"dir:{data_root}", "exists" if data_root.exists() else "missing (set --data_dir when running training)"))
    return checks


def collect_config_checks(code_root: Path) -> list[Check]:
    checks: list[Check] = []
    training_cfg_path = code_root / "configs" / "training_event_onset.yaml"
    runtime_cfg_path = code_root / "configs" / "runtime_event_onset.yaml"
    conversion_cfg_path = code_root / "configs" / "conversion_event_onset.yaml"

    model_cfg, data_cfg, _, _ = load_event_training_config(training_cfg_path)
    runtime_cfg = load_event_runtime_config(runtime_cfg_path)
    conversion_cfg = load_config(conversion_cfg_path)

    expected_emg_shape = (
        1,
        int(model_cfg.emg_in_channels),
        int(model_cfg.emg_freq_bins),
        int(model_cfg.emg_time_frames),
    )
    expected_imu_shape = (
        1,
        int(model_cfg.imu_input_dim),
        int(model_cfg.imu_num_steps),
    )
    checks.append(Check("INFO", "event.expected_emg_shape", str(expected_emg_shape)))
    checks.append(Check("INFO", "event.expected_imu_shape", str(expected_imu_shape)))

    if data_cfg.label_mode != "event_onset":
        checks.append(Check("ERROR", "training.label_mode", f"expected event_onset, got {data_cfg.label_mode}"))
    else:
        checks.append(Check("INFO", "training.label_mode", "event_onset"))

    conv_inputs = conversion_cfg.get("inputs", {}) or {}
    conv_emg = tuple(int(x) for x in conv_inputs.get("emg_shape", expected_emg_shape))
    conv_imu = tuple(int(x) for x in conv_inputs.get("imu_shape", expected_imu_shape))
    checks.append(Check("INFO", "conversion.emg_shape", str(conv_emg)))
    checks.append(Check("INFO", "conversion.imu_shape", str(conv_imu)))
    if conv_emg != expected_emg_shape:
        checks.append(Check("ERROR", "conversion.emg_shape", f"expected {expected_emg_shape}, got {conv_emg}"))
    if conv_imu != expected_imu_shape:
        checks.append(Check("ERROR", "conversion.imu_shape", f"expected {expected_imu_shape}, got {conv_imu}"))

    runtime_ckpt = _resolve_under_root(code_root, runtime_cfg.checkpoint_path)
    runtime_mindir = _resolve_under_root(code_root, runtime_cfg.model_path)
    runtime_meta = _resolve_under_root(code_root, runtime_cfg.model_metadata_path)
    checks.append(Check("INFO" if runtime_ckpt and runtime_ckpt.exists() else "WARN", "runtime.checkpoint_path", f"{runtime_ckpt}" if runtime_ckpt else "unset"))
    checks.append(Check("INFO" if runtime_mindir and runtime_mindir.exists() else "WARN", "runtime.model_path", f"{runtime_mindir}" if runtime_mindir else "unset"))
    checks.append(Check("INFO" if runtime_meta and runtime_meta.exists() else "WARN", "runtime.model_metadata_path", f"{runtime_meta}" if runtime_meta else "unset"))

    if runtime_meta and runtime_meta.exists():
        try:
            payload = json.loads(runtime_meta.read_text(encoding="utf-8"))
            inputs = {str(item.get("name")): tuple(int(x) for x in item.get("shape", [])) for item in payload.get("inputs", [])}
            if inputs.get("emg") != expected_emg_shape:
                checks.append(Check("ERROR", "runtime.metadata.emg", f"expected {expected_emg_shape}, got {inputs.get('emg')}"))
            if inputs.get("imu") != expected_imu_shape:
                checks.append(Check("ERROR", "runtime.metadata.imu", f"expected {expected_imu_shape}, got {inputs.get('imu')}"))
        except Exception as exc:
            checks.append(Check("ERROR", "runtime.metadata.parse", str(exc)))

    return checks


def print_checks(checks: Iterable[Check]) -> None:
    for item in checks:
        print(f"[{item.level}] {item.name}: {item.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Event-onset preflight checks")
    parser.add_argument("--mode", choices=("local", "ascend"), default="local")
    args = parser.parse_args()

    checks: list[Check] = []
    py = sys.version_info
    checks.append(Check("INFO", "python.version", f"{py.major}.{py.minor}.{py.micro}"))
    if (py.major, py.minor) < (3, 8):
        checks.append(Check("ERROR", "python.version", "requires >=3.8"))

    code_root = CODE_ROOT
    data_root = code_root.parent / "data"
    checks.extend(collect_dependency_checks(args.mode))
    checks.extend(collect_file_checks(code_root, data_root))
    checks.extend(collect_config_checks(code_root))

    print("=" * 72)
    print(f"Event-onset preflight mode={args.mode}")
    print(f"code_root={code_root}")
    print(f"data_root={data_root}")
    print("=" * 72)
    print_checks(checks)

    errors = [c for c in checks if c.level == "ERROR"]
    warns = [c for c in checks if c.level == "WARN"]
    print("=" * 72)
    print(f"Summary: {len(errors)} error(s), {len(warns)} warning(s), {len(checks)} checks")
    if errors:
        print("Preflight: FAILED")
        return 1
    print("Preflight: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
