"""
Unit tests for CSV dataset loader.
"""

import os
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.preprocessing.stft import PreprocessPipeline
from training.data.csv_dataset import CSVDatasetLoader


TEST_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_testdata"


@contextmanager
def _managed_temp_dir(prefix: str = "csv_loader_"):
    """
    Create test temp directories inside the workspace.
    This avoids system-temp permission issues seen on some Windows setups.
    """
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = TEST_TMP_ROOT / f"{prefix}{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
        if TEST_TMP_ROOT.exists() and not any(TEST_TMP_ROOT.iterdir()):
            TEST_TMP_ROOT.rmdir()


def _create_test_data(tmpdir: str, num_files: int = 3, num_rows: int = 600):
    gestures = ["Relax", "fist", "Pinch", "ok", "ye", "Sidegrip"]

    for gesture in gestures:
        gesture_dir = os.path.join(tmpdir, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        for i in range(num_files):
            csv_path = os.path.join(gesture_dir, f"test_{i}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8,"
                    "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n"
                )
                for _ in range(num_rows):
                    emg = np.random.randint(100, 160, size=8)
                    imu = np.random.randn(6) * 100
                    values = list(emg) + list(imu.astype(int))
                    f.write(",".join(str(v) for v in values) + "\n")


def test_scan_folders():
    with _managed_temp_dir() as tmpdir:
        _create_test_data(tmpdir, num_files=2)

        pipeline = PreprocessPipeline(sampling_rate=200, num_channels=6)
        loader = CSVDatasetLoader(
            data_dir=tmpdir,
            preprocess=pipeline,
            num_emg_channels=8,
        )

        stats = loader.get_stats()
        assert stats["total_files"] == 12, f"expected 12 files, got {stats}"
        assert len(loader.gesture_folders) == 6


def test_load_all():
    with _managed_temp_dir() as tmpdir:
        _create_test_data(tmpdir, num_files=2, num_rows=600)

        pipeline = PreprocessPipeline(sampling_rate=200, num_channels=6)
        loader = CSVDatasetLoader(
            data_dir=tmpdir,
            preprocess=pipeline,
            num_emg_channels=8,
            device_sampling_rate=1000,
            target_sampling_rate=200,
            segment_length=100,
            segment_stride=50,
        )

        samples, labels = loader.load_all()
        assert samples.ndim == 4
        assert samples.shape[1] == 6
        assert labels.ndim == 1
        assert len(samples) == len(labels)
        assert len(np.unique(labels)) == 6, "expected 6 classes"

        print(f"  loaded {len(samples)} samples, shape={samples.shape}")


def test_split():
    samples = np.random.randn(100, 6, 24, 13).astype(np.float32)
    labels = np.array([i % 6 for i in range(100)], dtype=np.int32)

    (train_s, train_l), (val_s, val_l) = CSVDatasetLoader.split(
        samples, labels, val_ratio=0.2
    )

    assert len(train_s) + len(val_s) == 100
    assert len(train_l) + len(val_l) == 100
    for c in range(6):
        assert np.sum(train_l == c) > 0
        assert np.sum(val_l == c) > 0


if __name__ == "__main__":
    tests = [test_scan_folders, test_load_all, test_split]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {test.__name__}: {exc}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
