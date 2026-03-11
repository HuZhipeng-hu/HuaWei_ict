from __future__ import annotations

import json
from pathlib import Path

from event_onset.inference import EventModelMetadata
from scripts.convert_event_onset import _build_model_metadata


def test_event_conversion_metadata_schema_roundtrip(tmp_path: Path):
    payload = _build_model_metadata(
        training_config_path="configs/training_event_onset.yaml",
        checkpoint_path="checkpoints/event.ckpt",
        model_path="models/event_onset.mindir",
        emg_shape=(1, 8, 24, 3),
        imu_shape=(1, 6, 16),
        class_names=["RELAX", "FIST", "PINCH"],
    )
    path = tmp_path / "event.model_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = EventModelMetadata.load(path)
    assert metadata is not None
    assert metadata.inputs["emg"] == (1, 8, 24, 3)
    assert metadata.inputs["imu"] == (1, 6, 16)
    assert metadata.class_names == ("RELAX", "FIST", "PINCH")
