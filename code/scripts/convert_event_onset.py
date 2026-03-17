"""Convert event-onset checkpoint to MindIR and emit runtime metadata."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.config import load_event_training_config
from event_onset.model import build_event_model
from shared.event_labels import public_event_labels
from shared.label_modes import get_label_mode_spec
from shared.run_utils import append_csv_row, copy_config_snapshot, dump_json, ensure_run_dir
from shared.config import load_config

try:
    import mindspore as ms
    from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore
    export = None  # type: ignore
    load_checkpoint = None  # type: ignore
    load_param_into_net = None  # type: ignore


SUMMARY_FIELDS = [
    "run_id",
    "training_config",
    "checkpoint_path",
    "output_path",
    "metadata_path",
    "emg_input_shape",
    "imu_input_shape",
    "device_target",
]


def _parse_shape(value: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return fallback
    raw = tuple(int(part.strip()) for part in value.split(","))
    if not raw:
        raise ValueError("input shape cannot be empty")
    return raw


def _build_model_metadata(
    *,
    training_config_path: str,
    checkpoint_path: str,
    model_path: str,
    emg_shape: tuple[int, ...],
    imu_shape: tuple[int, ...],
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "model_type": "event_onset",
        "training_config": training_config_path,
        "checkpoint_path": checkpoint_path,
        "model_path": model_path,
        "inputs": [
            {"name": "emg", "shape": list(emg_shape), "dtype": "float32"},
            {"name": "imu", "shape": list(imu_shape), "dtype": "float32"},
        ],
        "output": {"name": "logits", "dtype": "float32"},
        "class_names": public_event_labels(class_names),
    }


def _ensure_checkpoint_readable(path: str | Path) -> Path:
    ckpt = Path(path)
    if not ckpt.exists() or not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt}. "
            "Fix: pass --checkpoint <path/to/event_onset_best.ckpt>."
        )
    if ckpt.stat().st_size <= 0:
        raise ValueError(f"Checkpoint file is empty: {ckpt}")
    return ckpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert event-onset checkpoint to MindIR")
    parser.add_argument("--config", default="configs/conversion_event_onset.yaml")
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None, help="Output MindIR path")
    parser.add_argument("--metadata_output", default=None, help="Output metadata JSON path")
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--emg_input_shape", default=None, help="Override EMG shape like 1,8,24,3")
    parser.add_argument("--imu_input_shape", default=None, help="Override IMU shape like 1,6,16")
    parser.add_argument(
        "--target_db5_keys",
        default=None,
        help=(
            "Comma-separated action keys, "
            "e.g. TENSE_OPEN,V_SIGN,OK_SIGN,THUMB_UP,WRIST_CW,WRIST_CCW. CONTINUE/RELAX is implicit."
        ),
    )
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_root", default="artifacts/runs")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_onset.convert")
    args = build_parser().parse_args()

    if ms is None or Tensor is None or context is None or export is None:
        raise RuntimeError("MindSpore is required for event-onset conversion.")

    raw = load_config(args.config)
    run_id, run_dir = ensure_run_dir(args.run_root, args.run_id, default_tag="event_convert")
    copy_config_snapshot(args.config, run_dir / "config_snapshots" / Path(args.config).name)

    training_config = args.training_config or raw.get("training_config", "configs/training_event_onset.yaml")
    checkpoint_path = args.checkpoint or raw.get("checkpoint_path", "checkpoints/event_onset_best.ckpt")
    output_path = args.output or raw.get("output_path", "models/event_onset.mindir")
    metadata_output = (
        args.metadata_output
        or (raw.get("metadata", {}) or {}).get("output_json")
        or str(Path(output_path).with_suffix(".model_metadata.json"))
    )
    device_target = args.device_target or raw.get("device_target", "CPU")

    checkpoint_path = str(_ensure_checkpoint_readable(checkpoint_path))

    model_cfg, data_cfg, _, _ = load_event_training_config(training_config)
    if args.target_db5_keys:
        keys = [item.strip().upper() for item in str(args.target_db5_keys).split(",") if item.strip()]
        if not keys:
            raise ValueError("--target_db5_keys provided but no valid keys parsed.")
        data_cfg.target_db5_keys = keys
        model_cfg.num_classes = 1 + len(keys)
    model = build_event_model(model_cfg)

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    params = load_checkpoint(str(checkpoint_path))
    load_param_into_net(model, params)
    model.set_train(False)

    cfg_inputs = raw.get("inputs", {}) or {}
    default_emg_shape = (
        1,
        int(model_cfg.emg_in_channels),
        int(model_cfg.emg_freq_bins),
        int(model_cfg.emg_time_frames),
    )
    default_imu_shape = (
        1,
        int(model_cfg.imu_input_dim),
        int(model_cfg.imu_num_steps),
    )
    emg_shape = _parse_shape(
        args.emg_input_shape,
        tuple(int(x) for x in cfg_inputs.get("emg_shape", default_emg_shape)),
    )
    imu_shape = _parse_shape(
        args.imu_input_shape,
        tuple(int(x) for x in cfg_inputs.get("imu_shape", default_imu_shape)),
    )

    if len(emg_shape) != 4:
        raise ValueError(f"EMG input shape must be rank-4, got {emg_shape}")
    if len(imu_shape) != 3:
        raise ValueError(f"IMU input shape must be rank-3, got {imu_shape}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_stem = str(output_file.with_suffix(""))

    dummy_emg = Tensor(np.zeros(emg_shape, dtype=np.float32))
    dummy_imu = Tensor(np.zeros(imu_shape, dtype=np.float32))
    export(model, dummy_emg, dummy_imu, file_name=output_stem, file_format="MINDIR")
    model_path = str(Path(output_stem + ".mindir"))

    label_spec = get_label_mode_spec(data_cfg.label_mode, data_cfg.target_db5_keys)
    metadata = _build_model_metadata(
        training_config_path=str(training_config),
        checkpoint_path=str(checkpoint_path),
        model_path=model_path,
        emg_shape=emg_shape,
        imu_shape=imu_shape,
        class_names=label_spec.class_names,
    )
    metadata_path = Path(metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "training_config": str(training_config),
        "checkpoint_path": str(checkpoint_path),
        "output_path": model_path,
        "metadata_path": str(metadata_path),
        "emg_input_shape": list(emg_shape),
        "imu_input_shape": list(imu_shape),
        "device_target": str(device_target),
    }
    dump_json(run_dir / "conversion" / "event_conversion_summary.json", summary)
    append_csv_row(Path(args.run_root) / "event_conversion_results.csv", SUMMARY_FIELDS, summary)
    logger.info("Event conversion completed: %s", model_path)
    logger.info("Event metadata saved: %s", metadata_path)


if __name__ == "__main__":
    main()
