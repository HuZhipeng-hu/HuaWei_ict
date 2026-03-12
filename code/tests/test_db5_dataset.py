from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from ninapro_db5.config import DB5FeatureConfig, DB5PretrainConfig
from ninapro_db5.dataset import DB5PretrainDatasetLoader


def _write_db5_zip(
    path: Path,
    *,
    exercise: int = 1,
    active_label: int = 1,
    emg: np.ndarray | None = None,
    num_channels: int = 8,
    member_name: str = "s1/S1_E1_A1.mat",
) -> None:
    if emg is None:
        rest_emg = np.zeros((70, int(num_channels)), dtype=np.float32)
        active_emg = (0.8 * np.random.randn(70, int(num_channels))).astype(np.float32)
        emg = np.vstack([rest_emg, active_emg]).astype(np.float32)
    else:
        emg = np.asarray(emg, dtype=np.float32)
    rest = np.zeros((70, 1), dtype=np.int32)
    active = np.full((70, 1), int(active_label), dtype=np.int32)
    labels = np.vstack([rest, active])
    repetitions = np.ones((emg.shape[0], 1), dtype=np.int32)
    mat_payload = {
        "exercise": np.asarray([[int(exercise)]], dtype=np.int32),
        "restimulus": labels,
        "stimulus": labels,
        "rerepetition": repetitions,
        "emg": emg,
    }
    buffer = io.BytesIO()
    sio.savemat(buffer, mat_payload)
    buffer.seek(0)
    mode = "a" if path.exists() else "w"
    with zipfile.ZipFile(path, mode=mode, compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, buffer.read())


def _base_feature_config() -> DB5FeatureConfig:
    return DB5FeatureConfig(
        source_sampling_rate_hz=200,
        target_sampling_rate_hz=500,
        use_first_myo_only=True,
        first_myo_channel_count=8,
        lowcut_hz=20.0,
        highcut_hz=180.0,
        energy_min=0.25,
        static_std_min=0.08,
        clip_ratio_max=0.08,
        saturation_abs=126.0,
        action_quantile_percent=30.0,
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

    assert set(np.unique(labels).tolist()) == {0}
    assert "REST" not in class_names


def test_db5_label_mapping_keeps_exercises_distinct(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    _write_db5_zip(
        zip_path,
        exercise=1,
        active_label=13,
        member_name="s1/S1_E1_A1.mat",
    )
    _write_db5_zip(
        zip_path,
        exercise=2,
        active_label=1,
        member_name="s1/S1_E2_A1.mat",
    )
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=True,
        use_restimulus=True,
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    _, labels, _, metadata = loader.load_all_with_sources(return_metadata=True)

    mapping = {
        (item["exercise"], item["local_label"]): item["global_label"]
        for item in metadata
        if item["local_label"] != 0
    }
    assert mapping[(1, 13)] != mapping[(2, 1)]
    assert set(item["class_name"] for item in metadata if item["local_label"] != 0) == {"E1_G13", "E2_G01"}
    assert int(labels.max()) + 1 == len(loader.get_class_names())


def test_db5_full53_without_rest_keeps_two_contiguous_classes(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    _write_db5_zip(
        zip_path,
        exercise=1,
        active_label=13,
        member_name="s1/S1_E1_A1.mat",
    )
    _write_db5_zip(
        zip_path,
        exercise=1,
        active_label=14,
        member_name="s1/S1_E1_A2.mat",
    )
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=False,
        use_restimulus=True,
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    _, labels, _, _ = loader.load_all_with_sources(return_metadata=True)

    assert set(np.unique(labels).tolist()) == {0, 1}
    assert loader.get_class_names() == ["E1_G13", "E1_G14"]


def test_db5_feature_config_exposes_quality_and_bandpass_fields():
    cfg = _base_feature_config()
    assert cfg.lowcut_hz == 20.0
    assert cfg.highcut_hz == 180.0
    assert cfg.energy_min == 0.25
    assert cfg.static_std_min == 0.08
    assert cfg.clip_ratio_max == 0.08
    assert cfg.saturation_abs == 126.0
    assert cfg.action_quantile_percent == 30.0


def test_db5_loader_quality_first_window_selection_and_diagnostics(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    emg = np.zeros((140, 16), dtype=np.float32)
    t = np.linspace(0.0, 2.0 * np.pi, 70, dtype=np.float32)
    high_activity = 0.9 * np.sin(t)
    emg[70:, :] = high_activity[:, None]
    emg[70:100, :] *= 0.05
    _write_db5_zip(
        zip_path,
        exercise=1,
        active_label=1,
        emg=emg,
        num_channels=16,
        member_name="s1/S1_E1_A1.mat",
    )
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=False,
        use_restimulus=True,
        feature=DB5FeatureConfig(
            source_sampling_rate_hz=200,
            target_sampling_rate_hz=500,
            use_first_myo_only=False,
            first_myo_channel_count=16,
            lowcut_hz=20.0,
            highcut_hz=180.0,
            energy_min=0.2,
            static_std_min=0.05,
            clip_ratio_max=0.08,
            saturation_abs=126.0,
            action_quantile_percent=30.0,
            context_window_ms=240,
            window_step_ms=80,
            max_windows_per_segment=2,
            max_rest_windows_per_segment=1,
            emg_stft_window=64,
            emg_stft_hop=24,
            emg_n_fft=96,
            emg_freq_bins=24,
        ),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    samples, labels, _, _ = loader.load_all_with_sources(return_metadata=True)
    diagnostics = loader.get_window_diagnostics()

    assert samples.shape[1:] == (16, 24, 3)
    assert set(np.unique(labels).tolist()) == {0}
    assert diagnostics["totals"]["raw_candidates"] >= diagnostics["totals"]["selected"]
    assert diagnostics["totals"]["filtered_by_quality"] > 0
    assert diagnostics["per_class"]["E1_G01"]["selected"] == 1


def test_db5_loader_metadata_contains_recording_identity_fields(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    _write_db5_zip(zip_path, exercise=1, active_label=1, member_name="s1/S1_E1_A1.mat")
    feature = _base_feature_config()
    feature.action_quantile_percent = 0.0
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=False,
        use_restimulus=True,
        feature=feature,
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    _, _, source_ids, metadata = loader.load_all_with_sources(return_metadata=True)

    assert metadata
    first = metadata[0]
    assert str(first["user_id"]).strip()
    assert str(first["session_id"]).strip()
    assert str(first["recording_id"]).strip()
    assert "user_id=" in str(source_ids[0])
    assert "recording_id=" in str(source_ids[0])


def test_db5_loader_action_threshold_uses_adaptive_quantile(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    emg = np.zeros((140, 16), dtype=np.float32)
    emg[70:, :] = (0.24 * np.random.randn(70, 16)).astype(np.float32)
    _write_db5_zip(
        zip_path,
        exercise=1,
        active_label=1,
        emg=emg,
        num_channels=16,
        member_name="s1/S1_E1_A1.mat",
    )
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=False,
        use_restimulus=True,
        feature=DB5FeatureConfig(
            source_sampling_rate_hz=200,
            target_sampling_rate_hz=500,
            use_first_myo_only=False,
            first_myo_channel_count=16,
            lowcut_hz=20.0,
            highcut_hz=180.0,
            energy_min=0.8,
            static_std_min=0.6,
            clip_ratio_max=0.12,
            saturation_abs=126.0,
            action_quantile_percent=30.0,
            context_window_ms=240,
            window_step_ms=80,
            max_windows_per_segment=10,
            max_rest_windows_per_segment=2,
            emg_stft_window=64,
            emg_stft_hop=24,
            emg_n_fft=96,
            emg_freq_bins=24,
        ),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg)
    _, labels, _, _ = loader.load_all_with_sources(return_metadata=True)
    diagnostics = loader.get_window_diagnostics()

    assert set(np.unique(labels).tolist()) == {0}
    assert diagnostics["per_class"]["E1_G01"]["selected"] > 0
