"""Preflight checks for the event-onset production pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.manifest import load_event_manifest_rows
from shared.config import load_config
from shared.label_modes import get_label_mode_spec

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

def _resolve_data_arg(path_value: str | None, fallback: Path) -> Path:
    if path_value is None:
        return fallback
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (CODE_ROOT / raw).resolve()

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

def collect_file_checks(code_root: Path, wearer_data_root: Path, db5_data_root: Path) -> list[Check]:
    checks: list[Check] = []
    required_files = [
        code_root / "configs" / "pretrain_ninapro_db5.yaml",
        code_root / "configs" / "training_event_onset.yaml",
        code_root / "configs" / "conversion_event_onset.yaml",
        code_root / "configs" / "runtime_event_onset.yaml",
        code_root / "scripts" / "pretrain_db5_repr_method_matrix.py",
        code_root / "scripts" / "pretrain_ninapro_db5_repr.py",
        code_root / "scripts" / "finetune_event_onset.py",
        code_root / "scripts" / "convert_event_onset.py",
        code_root / "scripts" / "run_event_runtime.py",
        code_root / "scripts" / "evaluate_ckpt.py",
        code_root / "scripts" / "benchmark_event_runtime_ckpt.py",
    ]
    for path in required_files:
        checks.append(Check("INFO" if path.exists() else "ERROR", f"file:{path.relative_to(code_root)}", "exists" if path.exists() else "missing"))

    checks.append(
        Check(
            "INFO" if wearer_data_root.exists() else "WARN",
            f"dir:wearer_data={wearer_data_root}",
            "exists" if wearer_data_root.exists() else "missing (set --wearer_data_dir)",
        )
    )
    checks.append(
        Check(
            "INFO" if db5_data_root.exists() else "WARN",
            f"dir:db5_data={db5_data_root}",
            "exists" if db5_data_root.exists() else "missing (set --db5_data_dir)",
        )
    )
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
    checks.append(Check("INFO", "training.target_db5_keys", ",".join(data_cfg.target_db5_keys)))
    label_spec = get_label_mode_spec(data_cfg.label_mode, data_cfg.target_db5_keys)
    if int(model_cfg.num_classes) != len(label_spec.class_names):
        checks.append(
            Check(
                "ERROR",
                "training.model_num_classes",
                f"model.num_classes={model_cfg.num_classes}, expected={len(label_spec.class_names)}",
            )
        )
    else:
        checks.append(Check("INFO", "training.model_num_classes", str(model_cfg.num_classes)))

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
    runtime_map = _resolve_under_root(code_root, runtime_cfg.actuation_mapping_path)
    checks.append(Check("INFO" if runtime_ckpt and runtime_ckpt.exists() else "WARN", "runtime.checkpoint_path", f"{runtime_ckpt}" if runtime_ckpt else "unset"))
    checks.append(Check("INFO" if runtime_mindir and runtime_mindir.exists() else "WARN", "runtime.model_path", f"{runtime_mindir}" if runtime_mindir else "unset"))
    checks.append(Check("INFO" if runtime_meta and runtime_meta.exists() else "WARN", "runtime.model_metadata_path", f"{runtime_meta}" if runtime_meta else "unset"))
    checks.append(Check("INFO" if runtime_map and runtime_map.exists() else "WARN", "runtime.actuation_mapping_path", f"{runtime_map}" if runtime_map else "unset"))
    if runtime_map and runtime_map.exists():
        try:
            _, mapping_by_name = load_and_validate_actuation_map(runtime_map, class_names=label_spec.class_names)
            checks.append(Check("INFO", "runtime.actuation_mapping_keys", ",".join(sorted(mapping_by_name.keys()))))
        except Exception as exc:
            checks.append(Check("ERROR", "runtime.actuation_mapping", str(exc)))

    if runtime_meta and runtime_meta.exists():
        try:
            payload = json.loads(runtime_meta.read_text(encoding="utf-8"))
            inputs = {str(item.get("name")): tuple(int(x) for x in item.get("shape", [])) for item in payload.get("inputs", [])}
            if inputs.get("emg") != expected_emg_shape:
                checks.append(Check("ERROR", "runtime.metadata.emg", f"expected {expected_emg_shape}, got {inputs.get('emg')}"))
            if inputs.get("imu") != expected_imu_shape:
                checks.append(Check("ERROR", "runtime.metadata.imu", f"expected {expected_imu_shape}, got {inputs.get('imu')}"))
            metadata_class_names = [str(name).strip().upper() for name in payload.get("class_names", [])]
            expected_class_names = [str(name).strip().upper() for name in label_spec.class_names]
            if metadata_class_names and metadata_class_names != expected_class_names:
                checks.append(
                    Check(
                        "ERROR",
                        "runtime.metadata.class_names",
                        f"expected {expected_class_names}, got {metadata_class_names}",
                    )
                )
            elif not metadata_class_names:
                checks.append(Check("WARN", "runtime.metadata.class_names", "missing class_names"))
            else:
                checks.append(Check("INFO", "runtime.metadata.class_names", ",".join(metadata_class_names)))
        except Exception as exc:
            checks.append(Check("ERROR", "runtime.metadata.parse", str(exc)))

    return checks

def collect_db5_checks(code_root: Path, db5_data_root: Path, *, skip_probe: bool) -> list[Check]:
    checks: list[Check] = []
    try:
        import numpy as np
        import scipy.io as sio

        from ninapro_db5.config import load_db5_pretrain_config
    except Exception as exc:
        checks.append(Check("ERROR", "db5.import", f"failed to import DB5 modules: {exc}"))
        return checks

    cfg_path = code_root / "configs" / "pretrain_ninapro_db5.yaml"
    cfg = load_db5_pretrain_config(cfg_path)
    zip_files = sorted(db5_data_root.glob(cfg.zip_glob))
    subject_count = len(zip_files)
    checks.append(Check("INFO", "db5.subject_zip_count", str(subject_count)))
    if subject_count <= 0:
        checks.append(Check("ERROR", "db5.subject_zip_count", f"no files matched {cfg.zip_glob} under {db5_data_root}"))
        return checks
    if subject_count < 10:
        checks.append(Check("WARN", "db5.subject_zip_count", f"expected ~=10 subjects, found {subject_count}"))

    if skip_probe:
        checks.append(Check("WARN", "db5.full53_probe", "skipped by --skip_db5_probe"))
        return checks

    def _iter_segments(labels: np.ndarray):
        start = 0
        current = int(labels[0])
        for idx in range(1, labels.shape[0]):
            value = int(labels[idx])
            if value != current:
                yield current, start, idx
                start = idx
                current = value
        yield current, start, labels.shape[0]

    try:
        action_keys: set[str] = set()
        has_rest = False
        labels_key = "restimulus" if bool(cfg.use_restimulus) else "stimulus"
        for zip_path in zip_files:
            with zipfile.ZipFile(zip_path) as zf:
                members = sorted(
                    (item for item in zf.infolist() if item.filename.lower().endswith(".mat")),
                    key=lambda item: item.filename,
                )
                for member in members:
                    with zf.open(member) as handle:
                        mat = sio.loadmat(io.BytesIO(handle.read()))
                    labels = np.asarray(mat[labels_key]).reshape(-1).astype(np.int32)
                    exercise = int(np.asarray(mat["exercise"]).reshape(-1)[0])
                    if labels.size <= 0:
                        continue
                    for local_label, _, _ in _iter_segments(labels):
                        if int(local_label) == 0:
                            has_rest = True
                        else:
                            key = f"E{int(exercise)}_G{int(local_label):02d}"
                            action_keys.add(key)

        checks.append(Check("INFO", "db5.full53.action_key_count", str(len(action_keys))))
        checks.append(Check("INFO" if has_rest else "WARN", "db5.full53.has_rest", str(has_rest)))
        if len(action_keys) < 52:
            checks.append(
                Check(
                    "WARN",
                    "db5.full53.action_key_count",
                    f"expected around 52 action keys, found {len(action_keys)}",
                )
            )
    except Exception as exc:
        checks.append(Check("ERROR", "db5.full53_probe", str(exc)))

    return checks

def collect_budget_checks(
    code_root: Path,
    wearer_data_root: Path,
    *,
    budget_per_class: int,
    skip_probe: bool,
) -> list[Check]:
    checks: list[Check] = []
    model_cfg, data_cfg, _, _ = load_event_training_config(code_root / "configs" / "training_event_onset.yaml")
    del model_cfg
    checks.append(Check("INFO", "budget.per_class", str(int(budget_per_class))))

    manifest_path = Path(data_cfg.recordings_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = wearer_data_root / manifest_path
    if not manifest_path.exists():
        checks.append(Check("WARN", "budget.recordings_manifest", f"missing: {manifest_path}"))
        return checks

    rows = load_event_manifest_rows(manifest_path)
    if not rows:
        checks.append(Check("WARN", "budget.recordings_manifest", f"no rows: {manifest_path}"))
        return checks

    clip_counts: dict[str, int] = {}
    for row in rows.values():
        if str(row.get("capture_mode", "")) != data_cfg.capture_mode_filter:
            continue
        target_state = str(row.get("target_state", "")).strip().upper()
        if not target_state:
            continue
        clip_counts[target_state] = int(clip_counts.get(target_state, 0)) + 1

    if not clip_counts:
        checks.append(Check("WARN", "budget.filtered_clips", "0 clips after capture_mode_filter"))
        return checks

    checks.append(Check("INFO", "budget.filtered_clip_total", str(sum(clip_counts.values()))))
    if skip_probe:
        checks.append(Check("WARN", "budget.window_estimate", "skipped by --skip_budget_probe"))
        return checks

    target_classes = ["RELAX", *[str(key).strip().upper() for key in data_cfg.target_db5_keys]]
    for class_name in target_classes:
        clips = int(clip_counts.get(class_name, 0))
        per_clip = int(data_cfg.idle_top_k_windows_per_clip if class_name == "RELAX" else data_cfg.top_k_windows_per_clip)
        estimated_windows = clips * max(1, per_clip)
        selected = min(estimated_windows, int(max(0, budget_per_class)))
        if estimated_windows <= 0:
            level = "ERROR"
        elif estimated_windows < int(budget_per_class):
            level = "WARN"
        else:
            level = "INFO"
        checks.append(
            Check(
                level,
                f"budget.estimate.{class_name}",
                f"clips={clips}, est_windows={estimated_windows}, selected={selected}",
            )
        )

    return checks

def print_checks(checks: Iterable[Check]) -> None:
    for item in checks:
        print(f"[{item.level}] {item.name}: {item.detail}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Event-onset preflight checks")
    parser.add_argument("--mode", choices=("local", "ascend"), default="local")
    parser.add_argument("--db5_data_dir", default=None)
    parser.add_argument("--wearer_data_dir", default=None)
    parser.add_argument("--budget_per_class", type=int, default=60)
    parser.add_argument("--skip_db5_probe", action="store_true")
    parser.add_argument("--skip_budget_probe", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    py = sys.version_info
    checks.append(Check("INFO", "python.version", f"{py.major}.{py.minor}.{py.micro}"))
    if (py.major, py.minor) < (3, 8):
        checks.append(Check("ERROR", "python.version", "requires >=3.8"))

    code_root = CODE_ROOT
    wearer_data_root = _resolve_data_arg(args.wearer_data_dir, code_root.parent / "data")
    db5_data_root = _resolve_data_arg(args.db5_data_dir, code_root.parent / "data_ninaproDB5")

    checks.extend(collect_dependency_checks(args.mode))
    checks.extend(collect_file_checks(code_root, wearer_data_root, db5_data_root))
    checks.extend(collect_config_checks(code_root))
    checks.extend(collect_db5_checks(code_root, db5_data_root, skip_probe=bool(args.skip_db5_probe)))
    checks.extend(
        collect_budget_checks(
            code_root,
            wearer_data_root,
            budget_per_class=int(args.budget_per_class),
            skip_probe=bool(args.skip_budget_probe),
        )
    )

    print("=" * 72)
    print(f"Event-onset preflight mode={args.mode}")
    print(f"code_root={code_root}")
    print(f"db5_data_root={db5_data_root}")
    print(f"wearer_data_root={wearer_data_root}")
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
