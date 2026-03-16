"""Grid-search runtime thresholds for demo latch control and export tuned config."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from event_onset.actuation_mapping import load_and_validate_actuation_map
from event_onset.config import load_event_runtime_config, load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.inference import EventPredictor
from event_onset.runtime import EventOnsetController
from shared.config import load_config
from shared.label_modes import get_label_mode_spec
from training.data.split_strategy import load_manifest


DEFAULT_TARGET_KEYS = "TENSE_OPEN,THUMB_UP,WRIST_CW,WRIST_CCW"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune runtime thresholds for event demo control")
    parser.add_argument("--run_root", default="artifacts/runs")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--training_config", default="configs/training_event_onset_demo_p0.yaml")
    parser.add_argument("--runtime_config", default="configs/runtime_event_onset_demo_latch.yaml")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--recordings_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--target_db5_keys", default=DEFAULT_TARGET_KEYS)
    parser.add_argument("--backend", default="ckpt", choices=["ckpt", "lite"])
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_metadata", default=None)
    parser.add_argument("--confidence_thresholds", default="0.82,0.86,0.90")
    parser.add_argument("--activation_margins", default="0.10,0.14,0.18")
    parser.add_argument("--vote_windows", default="3,5")
    parser.add_argument("--vote_min_counts", default="2,3")
    parser.add_argument("--switch_confidence_boosts", default="0.08,0.12")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_runtime_config", default=None)
    return parser


def _parse_keys(raw: str) -> list[str]:
    keys = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not keys:
        raise ValueError("target_db5_keys is empty")
    return keys


def _parse_float_tokens(raw: str, *, name: str) -> list[float]:
    values: list[float] = []
    for token in [item.strip() for item in str(raw).split(",") if item.strip()]:
        values.append(float(token))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _parse_int_tokens(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for token in [item.strip() for item in str(raw).split(",") if item.strip()]:
        values.append(int(token))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be object: {path}")
    return payload


def _resolve_split_manifest(*, args: argparse.Namespace, run_dir: Path, training_data_cfg) -> Path:
    if str(args.split_manifest or "").strip():
        return Path(args.split_manifest).resolve()

    offline_summary_path = run_dir / "offline_summary.json"
    if offline_summary_path.exists():
        offline = _load_json(offline_summary_path)
        candidate = str(offline.get("manifest_path", "")).strip()
        if candidate:
            return (CODE_ROOT / candidate).resolve()

    return (CODE_ROOT / str(training_data_cfg.split_manifest_path)).resolve()


def _validate_runtime_class_contract(
    *,
    backend: str,
    expected_class_names: list[str],
    model_num_classes: int,
    mapping_by_name: dict[str, str],
    metadata,
) -> None:
    normalized_expected = [str(name).strip().upper() for name in expected_class_names]
    if int(model_num_classes) != len(normalized_expected):
        raise ValueError(
            f"model.num_classes={model_num_classes} mismatches expected labels={len(normalized_expected)} "
            f"({normalized_expected})"
        )

    mapping_keys = sorted(str(key).strip().upper() for key in mapping_by_name.keys())
    if mapping_keys != sorted(normalized_expected):
        raise ValueError(
            f"Actuation mapping keys mismatch expected classes. mapping_keys={mapping_keys}, "
            f"expected={sorted(normalized_expected)}"
        )

    if metadata is None:
        if backend == "lite":
            raise ValueError("Lite backend requires model metadata with class_names for strict runtime validation.")
        return

    metadata_class_names = [str(name).strip().upper() for name in metadata.class_names]
    if not metadata_class_names:
        if backend == "lite":
            raise ValueError("Lite backend metadata must include non-empty class_names.")
        return
    if metadata_class_names != normalized_expected:
        raise ValueError(
            "Runtime class order mismatch between config and model metadata: "
            f"config={normalized_expected}, metadata={metadata_class_names}"
        )


def _collect_test_clips(
    *,
    loader: EventClipDatasetLoader,
    test_sources: set[str],
    class_to_idx: dict[str, int],
) -> list[tuple[int, int, np.ndarray]]:
    clips: list[tuple[int, int, np.ndarray]] = []
    for start_state, target_state, matrix, metadata in loader.iter_clips():
        source_id = str(metadata.get("relative_path", ""))
        if source_id not in test_sources:
            continue
        start_name = str(start_state).strip().upper()
        target_name = str(target_state).strip().upper()
        if start_name not in class_to_idx or target_name not in class_to_idx:
            continue
        clips.append(
            (
                int(class_to_idx[start_name]),
                int(class_to_idx[target_name]),
                np.asarray(matrix[:, :14], dtype=np.float32),
            )
        )
    if not clips:
        raise RuntimeError("No test clips available after split/label filtering.")
    return clips


def _evaluate_combo(
    *,
    clips: list[tuple[int, int, np.ndarray]],
    class_names: list[str],
    label_to_state: dict[int, object],
    data_cfg,
    runtime_cfg,
    predict_proba,
    params: dict[str, float | int],
) -> dict[str, float]:
    inference_cfg = copy.deepcopy(runtime_cfg.inference)
    inference_cfg.confidence_threshold = float(params["confidence_threshold"])
    inference_cfg.activation_margin_threshold = float(params["activation_margin_threshold"])
    inference_cfg.vote_window = int(params["vote_window"])
    inference_cfg.vote_min_count = int(params["vote_min_count"])
    inference_cfg.switch_confidence_boost = float(params["switch_confidence_boost"])

    relax_state = label_to_state[0]
    action_states = {label_to_state[idx] for idx in range(1, len(class_names))}
    total = 0
    action_total = 0
    command_success = 0
    false_release = 0
    false_trigger = 0

    for start_label, target_label, matrix in clips:
        controller = EventOnsetController(
            data_config=data_cfg,
            inference_config=inference_cfg,
            runtime_config=runtime_cfg.runtime,
            class_names=class_names,
            label_to_state=label_to_state,
            predict_proba=predict_proba,
            actuator=None,
        )
        expected_state = label_to_state[int(target_label)]
        controller.state_machine.current_label = int(start_label)
        controller.state_machine.current_state = label_to_state[int(start_label)]
        steps = controller.ingest_rows(matrix)
        transitions = [step for step in steps if bool(step.decision.changed)]
        total += 1

        if int(target_label) == 0:
            triggered_action = any(step.decision.state in action_states for step in transitions)
            success = (not triggered_action) and (controller.current_state == relax_state)
            if triggered_action:
                false_trigger += 1
        else:
            action_total += 1
            reached_idx = next(
                (idx for idx, step in enumerate(transitions) if step.decision.state == expected_state),
                None,
            )
            reached_target = reached_idx is not None or controller.current_state == expected_state
            wrong_action = any(
                (step.decision.state in action_states) and (step.decision.state != expected_state)
                for step in transitions
            )
            released_after_target = False
            if reached_idx is not None:
                released_after_target = any(
                    step.decision.state == relax_state for step in transitions[reached_idx + 1 :]
                )

            if released_after_target:
                false_release += 1
            if wrong_action:
                false_trigger += 1
            success = reached_target and (not wrong_action) and (not released_after_target)

        if success:
            command_success += 1

    return {
        "command_success_rate": float(command_success / total) if total else 0.0,
        "false_release_rate": float(false_release / action_total) if action_total else 0.0,
        "false_trigger_rate": float(false_trigger / total) if total else 0.0,
        "total_clip_count": int(total),
        "action_clip_count": int(action_total),
    }


def _rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("command_success_rate", 0.0)),
        -float(row.get("false_trigger_rate", 1.0)),
        -float(row.get("false_release_rate", 1.0)),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "confidence_threshold",
        "activation_margin_threshold",
        "vote_window",
        "vote_min_count",
        "switch_confidence_boost",
        "command_success_rate",
        "false_trigger_rate",
        "false_release_rate",
        "total_clip_count",
        "action_clip_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def _write_runtime_config(
    *,
    source_runtime_config: Path,
    best_row: dict,
    output_path: Path,
) -> None:
    payload = load_config(source_runtime_config)
    if "inference" not in payload or not isinstance(payload["inference"], dict):
        payload["inference"] = {}
    payload["inference"]["confidence_threshold"] = float(best_row["confidence_threshold"])
    payload["inference"]["activation_margin_threshold"] = float(best_row["activation_margin_threshold"])
    payload["inference"]["vote_window"] = int(best_row["vote_window"])
    payload["inference"]["vote_min_count"] = int(best_row["vote_min_count"])
    payload["inference"]["switch_confidence_boost"] = float(best_row["switch_confidence_boost"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("event_runtime_threshold_tuner")
    args = build_parser().parse_args()

    run_root = Path(args.run_root)
    run_dir = run_root / str(args.run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    model_cfg, data_cfg, _, _ = load_event_training_config(args.training_config)
    runtime_cfg = load_event_runtime_config(args.runtime_config)
    target_keys = _parse_keys(args.target_db5_keys)
    data_cfg.target_db5_keys = list(target_keys)
    runtime_cfg.data.target_db5_keys = list(target_keys)

    if str(args.recordings_manifest or "").strip():
        data_cfg.recordings_manifest_path = str(args.recordings_manifest)
        runtime_cfg.data.recordings_manifest_path = str(args.recordings_manifest)

    split_manifest = _resolve_split_manifest(args=args, run_dir=run_dir, training_data_cfg=data_cfg)
    if not split_manifest.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest}")

    if str(args.checkpoint or "").strip():
        runtime_cfg.checkpoint_path = str(args.checkpoint)
    else:
        runtime_cfg.checkpoint_path = str(run_dir / "checkpoints" / "event_onset_best.ckpt")
    if str(args.model_path or "").strip():
        runtime_cfg.model_path = str(args.model_path)
    if str(args.model_metadata or "").strip():
        runtime_cfg.model_metadata_path = str(args.model_metadata)

    label_spec = get_label_mode_spec(data_cfg.label_mode, data_cfg.target_db5_keys)
    model_cfg.num_classes = int(len(label_spec.class_names))
    label_to_state, mapping_by_name = load_and_validate_actuation_map(
        runtime_cfg.actuation_mapping_path,
        class_names=label_spec.class_names,
    )

    predictor = EventPredictor(
        backend=str(args.backend),
        model_config=model_cfg,
        device_target=str(args.device_target),
        checkpoint_path=runtime_cfg.checkpoint_path,
        model_path=runtime_cfg.model_path,
        model_metadata_path=runtime_cfg.model_metadata_path,
    )
    _validate_runtime_class_contract(
        backend=str(args.backend),
        expected_class_names=list(label_spec.class_names),
        model_num_classes=int(model_cfg.num_classes),
        mapping_by_name=mapping_by_name,
        metadata=predictor.metadata,
    )

    manifest = load_manifest(split_manifest)
    test_sources = set(manifest.test_sources)
    loader = EventClipDatasetLoader(
        str(args.data_dir),
        data_cfg,
        recordings_manifest_path=data_cfg.recordings_manifest_path,
    )
    class_names = list(label_spec.class_names)
    class_to_idx = {str(name).strip().upper(): int(idx) for idx, name in enumerate(class_names)}
    clips = _collect_test_clips(loader=loader, test_sources=test_sources, class_to_idx=class_to_idx)

    confs = _parse_float_tokens(args.confidence_thresholds, name="--confidence_thresholds")
    margins = _parse_float_tokens(args.activation_margins, name="--activation_margins")
    vote_windows = _parse_int_tokens(args.vote_windows, name="--vote_windows")
    vote_mins = _parse_int_tokens(args.vote_min_counts, name="--vote_min_counts")
    boosts = _parse_float_tokens(args.switch_confidence_boosts, name="--switch_confidence_boosts")

    rows: list[dict] = []
    for conf, margin, vw, vm, boost in itertools.product(confs, margins, vote_windows, vote_mins, boosts):
        params = {
            "confidence_threshold": float(conf),
            "activation_margin_threshold": float(margin),
            "vote_window": int(vw),
            "vote_min_count": int(vm),
            "switch_confidence_boost": float(boost),
        }
        metrics = _evaluate_combo(
            clips=clips,
            class_names=class_names,
            label_to_state=label_to_state,
            data_cfg=runtime_cfg.data,
            runtime_cfg=runtime_cfg,
            predict_proba=predictor.predict_proba,
            params=params,
        )
        row = dict(params)
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise RuntimeError("No threshold combinations evaluated.")

    ranked = sorted(rows, key=_rank_key, reverse=True)
    best = dict(ranked[0])
    best["selected"] = True

    output_json = (
        Path(str(args.output_json)).resolve()
        if str(args.output_json or "").strip()
        else (run_dir / "evaluation" / "runtime_threshold_tuning_summary.json")
    )
    output_csv = (
        Path(str(args.output_csv)).resolve()
        if str(args.output_csv or "").strip()
        else (run_dir / "evaluation" / "runtime_threshold_tuning_summary.csv")
    )
    output_runtime_config = (
        Path(str(args.output_runtime_config)).resolve()
        if str(args.output_runtime_config or "").strip()
        else (run_dir / "evaluation" / "runtime_event_onset_demo_latch_tuned.yaml")
    )

    _write_csv(output_csv, ranked)
    _write_runtime_config(
        source_runtime_config=Path(args.runtime_config).resolve(),
        best_row=best,
        output_path=output_runtime_config,
    )

    summary = {
        "status": "ok",
        "run_id": str(args.run_id),
        "run_dir": str(run_dir),
        "target_db5_keys": list(target_keys),
        "training_config": str(args.training_config),
        "runtime_config": str(args.runtime_config),
        "mapping": mapping_by_name,
        "rank_rule": "command_success_rate desc, false_trigger_rate asc, false_release_rate asc",
        "search_space": {
            "confidence_thresholds": confs,
            "activation_margins": margins,
            "vote_windows": vote_windows,
            "vote_min_counts": vote_mins,
            "switch_confidence_boosts": boosts,
        },
        "best": best,
        "rows": ranked,
        "output_csv": str(output_csv),
        "output_runtime_config": str(output_runtime_config),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("runtime_threshold_tuning_summary=%s", output_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
