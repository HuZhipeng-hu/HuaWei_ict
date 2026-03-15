import numpy as np

from scripts.collect_event_data_continuous import (
    _evaluate_collection_gate,
    _iter_clip_rows,
    _normalize_emg_domain,
    _parse_keep_quality,
    detect_onsets,
)


def test_normalize_emg_domain_uint8_centering():
    emg = np.array([[120, 130], [127, 128]], dtype=np.float32)
    centered = _normalize_emg_domain(emg)
    assert float(centered[0, 0]) == -8.0
    assert float(centered[0, 1]) == 2.0


def test_detect_onsets_finds_two_events():
    samples = 2000
    emg = np.zeros((samples, 8), dtype=np.float32)
    emg[300:430, :] = 25.0
    emg[1100:1260, :] = 30.0
    result = detect_onsets(
        emg,
        sample_rate_hz=500,
        smooth_ms=50,
        q_low=0.1,
        q_high=0.95,
        threshold_alpha=0.3,
        min_active_sec=0.12,
        min_gap_sec=0.5,
    )
    onsets = result["onsets"]
    assert len(onsets) == 2
    assert 260 <= onsets[0] <= 340
    assert 1060 <= onsets[1] <= 1140


def test_iter_clip_rows_skips_out_of_bounds():
    matrix = np.zeros((1000, 17), dtype=np.float32)
    clips = _iter_clip_rows(
        matrix,
        onsets=[100, 400, 950],
        pre_roll_samples=200,
        clip_samples=500,
        max_clips=0,
    )
    assert clips == [(400, 200, 700)]


def test_parse_keep_quality_normalizes_tokens():
    keep = _parse_keep_quality("pass, warn")
    assert keep == {"pass", "warn"}


def test_evaluate_collection_gate_passes_when_thresholds_met():
    ok, failures = _evaluate_collection_gate(
        rows=18000,
        slice_candidate_count=8,
        accepted_clip_count=3,
        min_rows_gate=15000,
        min_candidates_gate=4,
        min_accepted_gate=2,
    )
    assert ok is True
    assert failures == []


def test_evaluate_collection_gate_fails_with_multiple_reasons():
    ok, failures = _evaluate_collection_gate(
        rows=12000,
        slice_candidate_count=2,
        accepted_clip_count=1,
        min_rows_gate=15000,
        min_candidates_gate=4,
        min_accepted_gate=2,
    )
    assert ok is False
    assert "rows<15000" in failures
    assert "slice_candidate_count<4" in failures
    assert "accepted_clip_count<2" in failures
