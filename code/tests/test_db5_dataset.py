from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from ninapro_db5.config import DB5FeatureConfig, DB5PretrainConfig
from ninapro_db5.dataset import DB5PretrainDatasetLoader


def _write_db5_zip(path: Path) -> None:
    emg = np.random.randn(140, 8).astype(np.float32)
    rest = np.zeros((70, 1), dtype=np.int32)
    active = np.ones((70, 1), dtype=np.int32)
    labels = np.vstack([rest, active])
    repetitions = np.ones((140, 1), dtype=np.int32)
    mat_payload = {
        "exercise": np.asarray([[1]], dtype=np.int32),
        "restimulus": labels,
        "stimulus": labels,
        "rerepetition": repetitions,
        "emg": emg,
    }
    buffer = io.BytesIO()
    sio.savemat(buffer, mat_payload)
    buffer.seek(0)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("s1/S1_E1_A1.mat", buffer.read())


def _base_feature_config() -> DB5FeatureConfig:
    return DB5FeatureConfig(
        source_sampling_rate_hz=200,
        target_sampling_rate_hz=500,
        use_first_myo_only=True,
        first_myo_channel_count=8,
        context_window_ms=240,
        window_step_ms=80,
        max_windows_per_segment=1,
        max_rest_windows_per_segment=1,
        emg_stft_window=64,
        emg_stft_hop=24,
        emg_n_fft=96,
        emg_freq_bins=24,
    )


def test_db5_loader_includes_rest_when_enabled(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    _write_db5_zip(zip_path)
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=True,
        use_restimulus=True,
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    samples, labels, _, metadata = loader.load_all_with_sources(return_metadata=True)

    assert samples.ndim == 4
    assert samples.shape[1:] == (8, 24, 3)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert any(item["class_name"] == "REST" for item in metadata)
    assert loader.get_class_names()[0] == "REST"


def test_db5_loader_excludes_rest_when_disabled(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    _write_db5_zip(zip_path)
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=False,
        use_restimulus=True,
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    _, labels, _, _ = loader.load_all_with_sources(return_metadata=True)
    class_names = loader.get_class_names()

    assert set(np.unique(labels).tolist()) == {1}
    assert "REST" not in class_names
