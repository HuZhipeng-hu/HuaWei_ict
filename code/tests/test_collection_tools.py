import csv
import json
from pathlib import Path

import numpy as np

from scripts.collection_utils import (
    MANIFEST_FIELDS,
    QUALITY_PASS,
    QUALITY_RETAKE,
    QUALITY_WARN,
    STANDARD_CSV_HEADERS,
    evaluate_recording_quality,
    read_source_csv,
    upsert_recordings_manifest,
    validate_metadata,
)
from scripts.collect_recordings import run_collection_batch
from scripts.import_armband_app_csv import run_import_batch
from shared.config import PreprocessConfig, QualityFilterConfig
from training.data.csv_dataset import CSVDatasetLoader


def _make_emg_signal(length: int, amp: float = 25.0) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, length, dtype=np.float32)
    channels = [amp * np.sin(t + phase) for phase in np.linspace(0.0, np.pi, 8, dtype=np.float32)]
    return np.stack(channels, axis=1).astype(np.float32)



def _standard_matrix(length: int, amp: float = 25.0) -> np.ndarray:
    emg = _make_emg_signal(length, amp=amp)
    imu = np.zeros((length, len(STANDARD_CSV_HEADERS) - 8), dtype=np.float32)
    return np.concatenate([emg, imu], axis=1).astype(np.float32)



def _write_standard_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STANDARD_CSV_HEADERS)
        writer.writerows(matrix.tolist())



def _write_app_export_csv(path: Path, matrix: np.ndarray) -> None:
    headers = ["timestamp", *STANDARD_CSV_HEADERS, "battery", "emg_pack_index"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index, row in enumerate(matrix.tolist(), start=1):
            writer.writerow([index, *row, 98, (index % 10) + 1])



def test_recordings_manifest_upsert_dedupes_by_relative_path(tmp_path: Path):
    manifest_path = tmp_path / "recordings_manifest.csv"
    first = {
        "relative_path": "RELAX/a.csv",
        "gesture": "RELAX",
        "user_id": "user1",
        "session_id": "sess1",
        "device_id": "dev1",
        "timestamp": "20260308T120000",
        "wearing_state": "normal",
        "recording_id": "a",
        "sample_count": "420",
        "quality_status": "warn",
        "quality_reasons": "low_energy",
        "source_origin": "import",
    }
    second = dict(first)
    second["quality_status"] = "pass"
    second["quality_reasons"] = ""

    upsert_recordings_manifest(manifest_path, first)
    upsert_recordings_manifest(manifest_path, second)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["relative_path"] for row in rows] == ["RELAX/a.csv"]
    assert rows[0]["quality_status"] == "pass"
    assert set(rows[0].keys()) == set(MANIFEST_FIELDS)



def test_import_batch_standardizes_app_csv_and_loader_can_read(tmp_path: Path):
    source_dir = tmp_path / "source"
    data_dir = tmp_path / "dataset"
    matrix = _standard_matrix(700, amp=32.0)
    source_csv = source_dir / "export_01.csv"
    _write_app_export_csv(source_csv, matrix)

    result = run_import_batch(
        data_dir=data_dir,
        gesture="relax",
        user_id="userA",
        session_id="session01",
        device_id="arm01",
        wearing_state="normal",
        source_csvs=[source_csv],
        move=False,
    )

    imported = Path(result["records"][0]["absolute_path"])
    standardized = read_source_csv(imported)
    assert standardized.shape == matrix.shape

    loader = CSVDatasetLoader(
        data_dir,
        {"RELAX": 0},
        PreprocessConfig(),
        quality_filter=QualityFilterConfig(enabled=False),
        recordings_manifest_path=data_dir / "recordings_manifest.csv",
    )
    samples, labels, source_ids, metadata = loader.load_all_with_sources(return_metadata=True)

    assert samples.shape[0] > 0
    assert labels.tolist() == [0] * len(labels)
    assert source_ids.shape[0] == samples.shape[0]
    assert all(item["user_id"] == "userA" for item in metadata)
    assert imported.exists()
    assert source_csv.exists()



def test_quality_evaluation_distinguishes_pass_warn_and_retake():
    preprocess = PreprocessConfig()
    quality = QualityFilterConfig(enabled=True, energy_min=2.5, clip_ratio_max=0.2, static_std_max=0.5)

    passed = evaluate_recording_quality(
        _standard_matrix(700, amp=30.0),
        preprocess_config=preprocess,
        quality_filter=quality,
    )
    assert passed["quality_status"] == QUALITY_PASS

    warn_matrix = _standard_matrix(700, amp=30.0)
    warn_matrix[:, 0] = 0.0
    warned = evaluate_recording_quality(
        warn_matrix,
        preprocess_config=preprocess,
        quality_filter=quality,
    )
    assert warned["quality_status"] == QUALITY_WARN
    assert "channel_anomaly" in warned["quality_reasons"]

    retake = evaluate_recording_quality(
        _standard_matrix(200, amp=30.0),
        preprocess_config=preprocess,
        quality_filter=quality,
    )
    assert retake["quality_status"] == QUALITY_RETAKE
    assert "length_insufficient" in retake["quality_reasons"]



def test_import_requires_explicit_metadata(tmp_path: Path):
    source_csv = tmp_path / "source.csv"
    _write_app_export_csv(source_csv, _standard_matrix(500))

    try:
        run_import_batch(
            data_dir=tmp_path / "dataset",
            gesture="relax",
            user_id="",
            session_id="session01",
            device_id="arm01",
            wearing_state="normal",
            source_csvs=[source_csv],
        )
    except ValueError as exc:
        assert "Missing required metadata" in str(exc)
    else:
        raise AssertionError("Expected explicit metadata validation error")



def test_collect_batch_from_source_csv_writes_csv_manifest_and_report(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    report_dir = tmp_path / "reports"
    source_csv = tmp_path / "seed.csv"
    _write_standard_csv(source_csv, _standard_matrix(700, amp=28.0))

    result = run_collection_batch(
        data_dir=data_dir,
        gesture="fist",
        user_id="userB",
        session_id="session02",
        device_id="arm02",
        wearing_state="offset",
        count=1,
        countdown_sec=0,
        rest_seconds=0,
        source_csvs=[source_csv],
        report_dir=report_dir,
    )

    manifest_path = data_dir / "recordings_manifest.csv"
    report_path = Path(result["report_path"])
    recording_path = Path(result["records"][0]["absolute_path"])

    assert manifest_path.exists()
    assert report_path.exists()
    assert recording_path.exists()

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["mode"] == "collect"
    assert report_payload["records"][0]["quality_status"] in {QUALITY_PASS, QUALITY_WARN, QUALITY_RETAKE}
    assert len(report_payload["records"][0]["emg_channel_mean_abs"]) == 8
    assert len(report_payload["records"][0]["emg_channel_std"]) == 8

    loader = CSVDatasetLoader(
        data_dir,
        {"FIST": 0},
        PreprocessConfig(),
        quality_filter=QualityFilterConfig(enabled=False),
        recordings_manifest_path=manifest_path,
    )
    samples, labels, _, metadata = loader.load_all_with_sources(return_metadata=True)

    assert samples.shape[0] > 0
    assert labels.shape[0] == samples.shape[0]
    assert all(item["wearing_state"] == "offset" for item in metadata)



def test_validate_metadata_normalizes_gesture_name():
    metadata = validate_metadata(
        gesture="side_grip",
        user_id="user-c",
        session_id="session03",
        device_id="arm03",
        wearing_state="normal",
    )

    assert metadata.gesture == "SIDEGRIP"
