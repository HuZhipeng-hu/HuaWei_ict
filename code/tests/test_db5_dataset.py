from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from ninapro_db5.config import DB5Aligned3Config, DB5FeatureConfig, DB5MappingProfileConfig, DB5PretrainConfig
from ninapro_db5.dataset import DB5PretrainDatasetLoader


def _write_db5_zip(
    path: Path,
    *,
    exercise: int = 1,
    active_label: int = 1,
    member_name: str = "s1/S1_E1_A1.mat",
) -> None:
    emg = np.random.randn(140, 8).astype(np.float32)
    rest = np.zeros((70, 1), dtype=np.int32)
    active = np.full((70, 1), int(active_label), dtype=np.int32)
    labels = np.vstack([rest, active])
    repetitions = np.ones((140, 1), dtype=np.int32)
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


def test_db5_aligned3_profile_filters_to_rest_fist_pinch(tmp_path: Path):
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
        include_rest_class=True,
        use_restimulus=True,
        aligned3=DB5Aligned3Config(
            enabled=True,
            candidate_mapping_profiles=[
                DB5MappingProfileConfig(
                    name="p_test",
                    fist=["E1_G13"],
                    pinch=["E1_G14"],
                )
            ],
            mapping_override=None,
            min_samples_per_class=1,
        ),
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg, pretrain_mode="aligned3")
    _, labels, _, metadata = loader.load_all_with_sources(return_metadata=True)
    report = loader.get_mapping_profile_report()

    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    assert loader.get_class_names() == ["REST", "FIST", "PINCH"]
    assert report["selected_profile"] == "p_test"
    assert report["selected_counts"]["FIST"] >= 1
    assert report["selected_counts"]["PINCH"] >= 1
    assert set(item["class_name"] for item in metadata) <= {"REST", "FIST", "PINCH"}


def test_db5_aligned3_fail_fast_when_mapping_lacks_samples(tmp_path: Path):
    zip_path = tmp_path / "s1.zip"
    _write_db5_zip(
        zip_path,
        exercise=1,
        active_label=13,
        member_name="s1/S1_E1_A1.mat",
    )
    cfg = DB5PretrainConfig(
        data_dir=str(tmp_path),
        zip_glob="s*.zip",
        include_rest_class=True,
        use_restimulus=True,
        aligned3=DB5Aligned3Config(
            enabled=True,
            candidate_mapping_profiles=[
                DB5MappingProfileConfig(
                    name="bad",
                    fist=["E1_G99"],
                    pinch=["E1_G98"],
                )
            ],
            mapping_override=None,
            min_samples_per_class=1,
        ),
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg, pretrain_mode="aligned3")
    try:
        loader.load_all_with_sources()
    except RuntimeError as exc:
        assert "No aligned3 mapping profile satisfies" in str(exc)
    else:
        raise AssertionError("Expected aligned3 mapping to fail fast when class coverage is missing")


def test_db5_aligned3_without_rest_keeps_two_contiguous_classes(tmp_path: Path):
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
        aligned3=DB5Aligned3Config(
            enabled=True,
            candidate_mapping_profiles=[
                DB5MappingProfileConfig(
                    name="p_test_no_rest",
                    fist=["E1_G13"],
                    pinch=["E1_G14"],
                )
            ],
            mapping_override=None,
            min_samples_per_class=1,
        ),
        feature=_base_feature_config(),
    )
    loader = DB5PretrainDatasetLoader(tmp_path, cfg, pretrain_mode="aligned3")
    _, labels, _, _ = loader.load_all_with_sources(return_metadata=True)

    assert set(np.unique(labels).tolist()) == {0, 1}
    assert loader.get_class_names() == ["FIST", "PINCH"]
