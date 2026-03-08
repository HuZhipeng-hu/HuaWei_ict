"""
Preflight checks for NeuroGrip handoff.

Usage:
    python scripts/preflight.py --mode local
    python scripts/preflight.py --mode ascend
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - script exits early if missing.
    yaml = None


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


@dataclass
class Check:
    level: str  # "ERROR" | "WARN" | "INFO"
    name: str
    detail: str


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def normalize_runtime_device(raw_device: str) -> str:
    aliases = {
        "CPU": "CPU",
        "GPU": "GPU",
        "ASCEND": "ASCEND",
        "NPU": "ASCEND",
    }
    key = str(raw_device).strip().upper()
    return aliases.get(key, key)


def _resolve_under_root(code_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return code_root / path


def check_mindir_loadability(model_path: Path, device: str, strict: bool) -> Check:
    if not has_module("mindspore_lite"):
        return Check(
            "WARN",
            "runtime.model_load_check",
            "mindspore_lite not installed; skipped build_from_file loadability check",
        )

    try:
        import mindspore_lite as mslite
    except Exception as exc:  # pragma: no cover - environment dependent
        return Check(
            "WARN",
            "runtime.model_load_check",
            f"mindspore_lite import failed ({exc}); skipped loadability check",
        )

    target = device.lower()
    context = mslite.Context()
    context.target = [target]
    if device == "CPU":
        context.cpu.thread_num = 1

    model = mslite.Model()
    try:
        model.build_from_file(str(model_path), mslite.ModelType.MINDIR, context)
    except Exception as exc:
        level = "ERROR" if strict else "WARN"
        return Check(
            level,
            "runtime.model_load_check",
            (
                f"build_from_file failed for {model_path} on device={device}. "
                f"Details: {exc}. This often indicates unsupported Lite ops "
                "(e.g. AdaptiveAvgPool2D on CPU) or model/runtime mismatch."
            ),
        )
    return Check(
        "INFO",
        "runtime.model_load_check",
        f"build_from_file passed for {model_path} on device={device}",
    )


def collect_dependency_checks(mode: str) -> list[Check]:
    checks: list[Check] = []
    base_required = ("numpy", "scipy", "yaml")
    ascend_required = ("mindspore", "mindspore_lite")
    optional = ("pyserial", "smbus2", "pytest")

    for mod in base_required:
        if has_module(mod):
            checks.append(Check("INFO", f"module:{mod}", "installed"))
        else:
            checks.append(Check("ERROR", f"module:{mod}", "missing"))

    if mode == "ascend":
        for mod in ascend_required:
            if has_module(mod):
                checks.append(Check("INFO", f"module:{mod}", "installed"))
            else:
                checks.append(Check("ERROR", f"module:{mod}", "missing"))
    else:
        for mod in ascend_required:
            if has_module(mod):
                checks.append(Check("INFO", f"module:{mod}", "installed"))
            else:
                checks.append(Check("WARN", f"module:{mod}", "missing (acceptable for local mode)"))

    for mod in optional:
        if has_module(mod):
            checks.append(Check("INFO", f"module:{mod}", "installed"))
        else:
            checks.append(Check("WARN", f"module:{mod}", "missing (optional)"))

    return checks


def collect_file_checks(code_root: Path, data_root: Path) -> list[Check]:
    checks: list[Check] = []
    required_files = [
        code_root / "configs" / "training.yaml",
        code_root / "configs" / "conversion.yaml",
        code_root / "configs" / "runtime.yaml",
        code_root / "training" / "train.py",
        code_root / "conversion" / "convert.py",
        code_root / "runtime" / "run.py",
    ]

    for path in required_files:
        if path.exists():
            checks.append(Check("INFO", f"file:{path.relative_to(code_root)}", "exists"))
        else:
            checks.append(Check("ERROR", f"file:{path.relative_to(code_root)}", "missing"))

    if data_root.exists():
        checks.append(Check("INFO", f"dir:{data_root}", "exists"))
    else:
        checks.append(Check("ERROR", f"dir:{data_root}", "missing"))

    legacy_snapshot = code_root / "code"
    if legacy_snapshot.exists():
        checks.append(
            Check(
                "WARN",
                f"dir:{legacy_snapshot.relative_to(code_root)}",
                "legacy snapshot exists; use only main code root for development",
            )
        )

    return checks


def _derive_expected_input_shape(preprocess_config) -> tuple[int, int, int, int]:
    from shared.config import get_protocol_input_shape

    return get_protocol_input_shape(preprocess_config)


def _append_data_checks(checks: list[Check], data_root: Path) -> None:
    from shared.gestures import FOLDER_TO_GESTURE

    if not data_root.exists():
        return

    dirs = {p.name.lower(): p for p in data_root.iterdir() if p.is_dir()}
    for gesture in FOLDER_TO_GESTURE:
        folder = dirs.get(gesture)
        if folder is None:
            checks.append(Check("ERROR", f"data.{gesture}", "missing folder"))
            continue
        csv_count = len(list(folder.glob("*.csv")))
        if csv_count == 0:
            checks.append(Check("ERROR", f"data.{gesture}", "no csv files"))
        else:
            checks.append(Check("INFO", f"data.{gesture}", f"{csv_count} csv files"))


def collect_config_checks(code_root: Path, data_root: Path, mode: str) -> list[Check]:
    from shared.config import load_conversion_config, load_runtime_config, load_training_config

    checks: list[Check] = []

    runtime_cfg = load_runtime_config(code_root / "configs" / "runtime.yaml")
    conversion_cfg = load_conversion_config(code_root / "configs" / "conversion.yaml")
    train_model_cfg, train_pp_cfg, _, _ = load_training_config(code_root / "configs" / "training.yaml")

    runtime_device_raw = runtime_cfg.device.target
    runtime_device = normalize_runtime_device(runtime_device_raw)

    if runtime_device not in {"CPU", "GPU", "ASCEND"}:
        checks.append(
            Check(
                "ERROR",
                "runtime.device.target",
                (
                    f"unsupported value {runtime_device_raw!r}. "
                    "Expected one of CPU/GPU/Ascend (alias NPU)."
                ),
            )
        )
    else:
        checks.append(Check("INFO", "runtime.device.target", runtime_device))
        if mode == "ascend" and runtime_device == "CPU":
            checks.append(
                Check(
                    "ERROR",
                    "runtime.device.target",
                    "mode=ascend does not allow runtime.device.target=CPU. "
                    "Set configs/runtime.yaml device.target to Ascend (or use --device).",
                )
            )

    runtime_model_path = _resolve_under_root(code_root, runtime_cfg.model_path)
    conversion_output_path = _resolve_under_root(code_root, conversion_cfg.output_path)
    checkpoint_path = _resolve_under_root(code_root, conversion_cfg.checkpoint_path)

    if runtime_model_path.exists():
        checks.append(Check("INFO", "runtime.model_path", f"found: {runtime_model_path}"))
        if runtime_device in {"CPU", "GPU", "ASCEND"}:
            checks.append(
                check_mindir_loadability(
                    model_path=runtime_model_path,
                    device=runtime_device,
                    strict=(mode == "ascend"),
                )
            )
    else:
        checks.append(
            Check(
                "WARN",
                "runtime.model_path",
                f"not found: {runtime_model_path} (expected before runtime deployment)",
            )
        )

    if runtime_model_path == conversion_output_path:
        checks.append(Check("INFO", "runtime.model_alignment", "runtime model path matches conversion output"))
    else:
        checks.append(
            Check(
                "WARN",
                "runtime.model_alignment",
                f"runtime model path {runtime_model_path} != conversion output {conversion_output_path}",
            )
        )

    if checkpoint_path.exists():
        checks.append(Check("INFO", "conversion.checkpoint_path", f"found: {checkpoint_path}"))
    else:
        checks.append(
            Check(
                "WARN",
                "conversion.checkpoint_path",
                f"not found: {checkpoint_path} (expected before training/export)",
            )
        )

    try:
        runtime_expected_shape = _derive_expected_input_shape(runtime_cfg.preprocess)
        checks.append(Check("INFO", "runtime.expected_input_shape", str(runtime_expected_shape)))
    except Exception as exc:
        runtime_expected_shape = None
        checks.append(Check("ERROR", "runtime.expected_input_shape", f"failed to derive shape: {exc}"))

    try:
        training_expected_shape = _derive_expected_input_shape(train_pp_cfg)
        checks.append(Check("INFO", "training.expected_input_shape", str(training_expected_shape)))
    except Exception as exc:
        training_expected_shape = None
        checks.append(Check("ERROR", "training.expected_input_shape", f"failed to derive shape: {exc}"))

    conversion_input_shape = tuple(int(x) for x in conversion_cfg.input_shape)
    checks.append(Check("INFO", "conversion.input_shape", str(conversion_input_shape)))

    if runtime_expected_shape is not None and training_expected_shape is not None:
        if runtime_expected_shape == training_expected_shape:
            checks.append(Check("INFO", "runtime.training_shape_consistency", "runtime and training shapes match"))
        else:
            checks.append(
                Check(
                    "ERROR",
                    "runtime.training_shape_consistency",
                    f"runtime {runtime_expected_shape} != training {training_expected_shape}",
                )
            )

    if runtime_expected_shape is not None:
        if runtime_expected_shape == conversion_input_shape:
            checks.append(Check("INFO", "runtime.conversion_shape_consistency", "runtime and conversion shapes match"))
        else:
            checks.append(
                Check(
                    "ERROR",
                    "runtime.conversion_shape_consistency",
                    f"runtime {runtime_expected_shape} != conversion {conversion_input_shape}",
                )
            )

        runtime_channels = runtime_expected_shape[1]
        if train_model_cfg.in_channels == runtime_channels:
            checks.append(Check("INFO", "training.model_channels", "model.in_channels matches preprocess output"))
        else:
            checks.append(
                Check(
                    "ERROR",
                    "training.model_channels",
                    f"training.model.in_channels={train_model_cfg.in_channels} != preprocess output channels={runtime_channels}",
                )
            )

    _append_data_checks(checks, data_root)
    return checks


def print_checks(checks: Iterable[Check]) -> None:
    for c in checks:
        print(f"[{c.level}] {c.name}: {c.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroGrip preflight checks")
    parser.add_argument(
        "--mode",
        choices=("local", "ascend"),
        default="local",
        help="local: no MindSpore required; ascend: require MindSpore stack.",
    )
    args = parser.parse_args()

    code_root = CODE_ROOT
    data_root = code_root.parent / "data"

    checks: list[Check] = []

    py = sys.version_info
    if (py.major, py.minor) < (3, 8):
        checks.append(Check("ERROR", "python.version", f"requires >=3.8, current={py.major}.{py.minor}.{py.micro}"))
    else:
        checks.append(Check("INFO", "python.version", f"{py.major}.{py.minor}.{py.micro}"))
    if args.mode == "ascend" and (py.major, py.minor) >= (3, 12):
        checks.append(
            Check(
                "WARN",
                "python.version",
                "MindSpore 2.7.1 environments often use Python <=3.11; verify target machine compatibility.",
            )
        )

    if yaml is None:
        checks.append(Check("ERROR", "module:yaml", "PyYAML not installed"))
        print_checks(checks)
        return 1

    checks.extend(collect_dependency_checks(args.mode))
    checks.extend(collect_file_checks(code_root, data_root))
    checks.extend(collect_config_checks(code_root, data_root, args.mode))

    print("=" * 72)
    print(f"NeuroGrip preflight mode={args.mode}")
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
