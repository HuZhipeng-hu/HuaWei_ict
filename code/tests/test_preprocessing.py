import numpy as np

from runtime.control.controller import RuntimeController
from shared.config import PreprocessConfig
from shared.preprocessing import PreprocessPipeline


def test_dual_branch_output_shape():
    cfg = PreprocessConfig()
    cfg.dual_branch.enabled = True
    pipeline = PreprocessPipeline(cfg)

    raw = np.random.randn(420, 8).astype(np.float32)
    feat = pipeline.process_window(raw)
    assert feat.shape == (16, 24, 6)


def test_tta_window_slicing_respects_offsets():
    base_window = 420
    stride = 210
    offsets = [0.0, 0.33, 0.66]
    read_window = RuntimeController._calc_read_window_size(base_window, stride, offsets)
    raw = np.random.randn(read_window, 8).astype(np.float32)

    slices = RuntimeController._slice_tta_windows(raw, base_window, stride, offsets)
    assert len(slices) == 3
    assert all(seg.shape == (420, 8) for seg in slices)
