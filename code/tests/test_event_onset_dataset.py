import csv
from pathlib import Path

import numpy as np

from event_onset.config import load_event_training_config
from event_onset.dataset import EventClipDatasetLoader
from event_onset.manifest import EVENT_MANIFEST_FIELDS, upsert_event_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_CONFIG = REPO_ROOT / "configs" / "training_event_onset.yaml"


def _write_standard_csv(path: Path, matrix: np.ndarray) -> None:
    headers = [
        "emg1",
        "emg2",
        "emg3",
        "emg4",
        "emg5",
        "emg6",
        "emg7",
        "emg8",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "angle_pitch",
        "angle_roll",
        "angle_yaw",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(matrix.tolist())


def _build_action_matrix(length: int = 600, action_start: int = 220, action_end: int = 360, amp: float = 28.0) -> np.ndarray:
    t = np.linspace(0, 6 * np.pi, length, dtype=np.float32)
    emg = np.zeros((length, 8), dtype=np.float32)
    active = slice(action_start, action_end)
    for channel in range(8):
        emg[active, channel] = amp * np.sin(t[active] + channel / 3.0)
    imu = np.zeros((length, 9), dtype=np.float32)
    imu[active, :6] = 6.0
    return np.concatenate([emg, imu], axis=1).astype(np.float32)


def _build_relax_matrix(length: int = 1000, amp: float = 0.8) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
    emg = np.stack([amp * np.sin(t + channel) for channel in range(8)], axis=1).astype(np.float32)
    imu = np.zeros((length, 9), dtype=np.float32)
    return np.concatenate([emg, imu], axis=1).astype(np.float32)


def _manifest_row(relative_path: str, *, start_state: str, target_state: str, sample_count: int) -> dict[str, object]:
    payload = {field: "" for field in EVENT_MANIFEST_FIELDS}
    payload.update(
        {
            "relative_path": relative_path,
            "gesture": target_state,
            "capture_mode": "event_onset",
            "start_state": start_state,
            "target_state": target_state,
            "user_id": "user01",
            "session_id": "s01",
            "device_id": "arm01",
            "timestamp": "20260310T170000",
            "wearing_state": "normal",
            "recording_id": Path(relative_path).stem,
            "sample_count": sample_count,
            "clip_duration_ms": 2000 if target_state == "RELAX" else 1200,
            "pre_roll_ms": 0 if target_state == "RELAX" else 400,
            "device_sampling_rate_hz": 500,
            "imu_sampling_rate_hz": 50,
            "quality_status": "pass",
            "quality_reasons": "",
            "source_origin": "test",
        }
    )
    return payload


def _build_loader(tmp_path: Path) -> tuple[EventClipDatasetLoader, Path]:
    _, data_cfg, _, _ = load_event_training_config(tmp_path / "training_event_onset.yaml")
    manifest_path = tmp_path / "recordings_manifest.csv"
    loader = EventClipDatasetLoader(tmp_path, data_cfg, recordings_manifest_path=manifest_path)
    return loader, manifest_path


def test_event_loader_selects_top_k_action_windows_and_uses_target_state(tmp_path: Path):
    config_path = tmp_path / "training_event_onset.yaml"
    config_path.write_text(BASE_TRAINING_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    loader, manifest_path = _build_loader(tmp_path)

    action_a = loader.label_spec.class_names[1]
    action_b = loader.label_spec.class_names[2]
    action_a_path = tmp_path / action_a / "clip_a.csv"
    action_b_path = tmp_path / action_b / "clip_b.csv"
    _write_standard_csv(action_a_path, _build_action_matrix())
    _write_standard_csv(action_b_path, _build_action_matrix(amp=32.0))
    upsert_event_manifest(
        manifest_path,
        _manifest_row(f"{action_a}/clip_a.csv", start_state="RELAX", target_state=action_a, sample_count=600),
    )
    upsert_event_manifest(
        manifest_path,
        _manifest_row(f"{action_b}/clip_b.csv", start_state=action_a, target_state=action_b, sample_count=600),
    )

    emg, imu, labels, source_ids, metadata = loader.load_all_with_sources(return_metadata=True)

    assert emg.shape[0] == 4
    assert imu.shape[0] == 4
    assert set(labels.tolist()) == {1, 2}
    assert sum(label == 1 for label in labels.tolist()) == 2
    assert sum(label == 2 for label in labels.tolist()) == 2
    assert set(source_ids.tolist()) == {f"{action_a}/clip_a.csv", f"{action_b}/clip_b.csv"}
    assert all(item["selection_mode"] == "onset_peak_distance_energy" for item in metadata)
    assert all("onset_idx" in item for item in metadata)


def test_event_loader_relax_clip_only_produces_idle_samples(tmp_path: Path):
    config_path = tmp_path / "training_event_onset.yaml"
    config_path.write_text(BASE_TRAINING_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    loader, manifest_path = _build_loader(tmp_path)

    relax_path = tmp_path / "RELAX" / "clip_relax.csv"
    _write_standard_csv(relax_path, _build_relax_matrix())
    upsert_event_manifest(manifest_path, _manifest_row("RELAX/clip_relax.csv", start_state="RELAX", target_state="RELAX", sample_count=1000))

    emg, imu, labels, _, metadata = loader.load_all_with_sources(return_metadata=True)

    assert emg.shape[0] == 4
    assert imu.shape[0] == emg.shape[0]
    assert set(labels.tolist()) == {0}
    assert all(item["target_state"] == "CONTINUE" for item in metadata)


def test_event_loader_filters_low_energy_action_clip(tmp_path: Path):
    config_path = tmp_path / "training_event_onset.yaml"
    config_path.write_text(BASE_TRAINING_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    loader, manifest_path = _build_loader(tmp_path)

    low_energy = _build_action_matrix(amp=0.6)
    action_name = loader.label_spec.class_names[1]
    action_path = tmp_path / action_name / "clip_low.csv"
    _write_standard_csv(action_path, low_energy)
    upsert_event_manifest(
        manifest_path,
        _manifest_row(f"{action_name}/clip_low.csv", start_state="RELAX", target_state=action_name, sample_count=600),
    )

    try:
        loader.load_all_with_sources()
    except RuntimeError as exc:
        assert "No event-onset samples" in str(exc)
    else:
        raise AssertionError("Expected low-energy event clip to be filtered out")
