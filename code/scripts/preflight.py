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


def read_yaml(path: Path):
    if yaml is None:
        raise RuntimeError("PyYAML is required for preflight checks.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def collect_config_checks(code_root: Path, data_root: Path, mode: str) -> list[Check]:
    checks: list[Check] = []

    runtime_cfg = read_yaml(code_root / "configs" / "runtime.yaml")
    conversion_cfg = read_yaml(code_root / "configs" / "conversion.yaml")
    inference_cfg = runtime_cfg.get("inference", {})
    runtime_device_raw = inference_cfg.get("device", "CPU")
    runtime_device = normalize_runtime_device(runtime_device_raw)

    if runtime_device not in {"CPU", "GPU", "ASCEND"}:
        checks.append(
            Check(
                "ERROR",
                "runtime.inference.device",
                (
                    f"unsupported value {runtime_device_raw!r}. "
                    "Expected one of CPU/GPU/Ascend (alias NPU)."
                ),
            )
        )
    else:
        checks.append(Check("INFO", "runtime.inference.device", f"{runtime_device}"))
        if mode == "ascend" and runtime_device == "CPU":
            checks.append(
                Check(
                    "ERROR",
                    "runtime.inference.device",
                    "mode=ascend does not allow runtime.inference.device=CPU. "
                    "Set configs/runtime.yaml inference.device to Ascend (or use --device).",
                )
            )

    model_path = code_root / inference_cfg.get("model_path", "models/neurogrip.mindir")
    if model_path.exists():
        checks.append(Check("INFO", "runtime.model_path", f"found: {model_path}"))
        if runtime_device in {"CPU", "GPU", "ASCEND"}:
            checks.append(
                check_mindir_loadability(
                    model_path=model_path,
                    device=runtime_device,
                    strict=(mode == "ascend"),
                )
            )
    else:
        checks.append(
            Check(
                "WARN",
                "runtime.model_path",
                f"not found: {model_path} (expected before conversion)",
            )
        )

    ckpt_path = code_root / conversion_cfg.get("export", {}).get("checkpoint", "checkpoints/neurogrip_best.ckpt")
    if ckpt_path.exists():
        checks.append(Check("INFO", "conversion.export.checkpoint", f"found: {ckpt_path}"))
    else:
        checks.append(
            Check(
                "WARN",
                "conversion.export.checkpoint",
                f"not found: {ckpt_path} (expected before training)",
            )
        )

    if data_root.exists():
        expected_gestures = ("relax", "fist", "pinch", "ok", "ye", "sidegrip")
        dirs = {p.name.lower(): p for p in data_root.iterdir() if p.is_dir()}
        for gesture in expected_gestures:
            folder = dirs.get(gesture)
            if folder is None:
                checks.append(Check("ERROR", f"data.{gesture}", "missing folder"))
                continue
            csv_count = len(list(folder.glob("*.csv")))
            if csv_count == 0:
                checks.append(Check("ERROR", f"data.{gesture}", "no csv files"))
            else:
                checks.append(Check("INFO", f"data.{gesture}", f"{csv_count} csv files"))

    pp = runtime_cfg.get("preprocess", {})
    model = runtime_cfg.get("model", {})
    expected_freq_bins = int(pp.get("stft_n_fft", 46)) // 2 + 1
    segment_length = int(pp.get("segment_length", 84))
    window_size = int(pp.get("stft_window_size", 24))
    hop_size = int(pp.get("stft_hop_size", 12))
    expected_time_frames = max(1, (segment_length - window_size) // hop_size + 1)
    expected_shape = (1, int(pp.get("num_channels", 6)), expected_freq_bins, expected_time_frames)
    checks.append(Check("INFO", "runtime.expected_input_shape", str(expected_shape)))

    model_in = int(model.get("in_channels", 6))
    pp_ch = int(pp.get("num_channels", 6))
    if model_in != pp_ch:
        checks.append(
            Check(
                "WARN",
                "runtime.channel_consistency",
                f"model.in_channels={model_in} != preprocess.num_channels={pp_ch}",
            )
        )
    else:
        checks.append(Check("INFO", "runtime.channel_consistency", "model and preprocess channels match"))

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

    code_root = Path(__file__).resolve().parents[1]
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
