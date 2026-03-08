import csv
from pathlib import Path

import numpy as np

from shared.config import PreprocessConfig, QualityFilterConfig
from training.data.csv_dataset import CSVDatasetLoader


def _write_csv(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"])
        for row in arr:
            writer.writerow([float(x) for x in row])


def _make_signal(length: int, amp: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
    sig = np.stack([amp * np.sin(t + i) for i in range(6)], axis=1)
    return sig.astype(np.float32)


def test_quality_filter_removes_low_energy_windows(tmp_path: Path):
    gestures = {"RELAX": 0, "FIST": 1}
    cfg = PreprocessConfig()
    cfg.dual_branch.enabled = True

    weak = np.zeros((420, 6), dtype=np.float32)
    strong = _make_signal(420, 30.0)

    _write_csv(tmp_path / "RELAX" / "user1_session1_20260101.csv", weak)
    _write_csv(tmp_path / "FIST" / "user1_session1_20260101.csv", strong)

    qf = QualityFilterConfig(enabled=True, energy_min=2.0, clip_ratio_max=0.5, static_std_max=0.1)
    loader = CSVDatasetLoader(tmp_path, gestures, cfg, quality_filter=qf)
    samples, labels, source_ids, metadata = loader.load_all_with_sources(return_metadata=True)

    assert samples.ndim == 4
    assert len(labels) > 0
    assert all(m["recording_id"] for m in metadata)

    report = loader.get_quality_report()
    assert report["quality"]["filtered_total"] >= 1
    assert report["quality"]["kept_windows"] == len(labels)


def test_multi_phase_expands_samples(tmp_path: Path):
    gestures = {"RELAX": 0}
    cfg = PreprocessConfig()
    cfg.dual_branch.enabled = True
    cfg.dual_branch.multi_phase_offsets = [0.0, 0.5]
    qf = QualityFilterConfig(enabled=False)

    sig = _make_signal(700, 20.0)
    _write_csv(tmp_path / "RELAX" / "user1_session2_20260102.csv", sig)

    loader = CSVDatasetLoader(tmp_path, gestures, cfg, quality_filter=qf)
    samples, labels, source_ids = loader.load_all_with_sources()
    assert samples.shape[0] >= 2
    assert samples.shape[1:] == (12, 24, 6)
    assert labels.shape[0] == samples.shape[0]
    assert source_ids.shape[0] == samples.shape[0]


def test_recordings_manifest_overrides_filename_metadata(tmp_path: Path):
    gestures = {"RELAX": 0}
    cfg = PreprocessConfig()
    cfg.dual_branch.enabled = True
    qf = QualityFilterConfig(enabled=False)

    sig = _make_signal(700, 20.0)
    rel_path = Path("RELAX") / "1772246123.csv"
    _write_csv(tmp_path / rel_path, sig)

    manifest_path = tmp_path / "recordings_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "relative_path,gesture,user_id,session_id,device_id,timestamp,wearing_state",
                "RELAX/1772246123.csv,RELAX,user_a,sess_03,dev_x,20260308T101500,offset",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loader = CSVDatasetLoader(
        tmp_path,
        gestures,
        cfg,
        quality_filter=qf,
        recordings_manifest_path=manifest_path,
    )
    samples, labels, source_ids, metadata = loader.load_all_with_sources(return_metadata=True)

    assert samples.shape[0] > 0
    assert set(source_ids.tolist()) == {"RELAX/1772246123.csv"}
    assert all(item["user_id"] == "user_a" for item in metadata)
    assert all(item["session_id"] == "sess_03" for item in metadata)
    assert all(item["wearing_state"] == "offset" for item in metadata)
