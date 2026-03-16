from __future__ import annotations

import numpy as np

from event_onset.algo import (
    ALGO_MODE_V1,
    ALGO_MODE_V2,
    EventAlgoModel,
    EventAlgoPredictor,
    build_event_algo_feature_vector,
    compute_rule_signal_scores,
    compute_algo_stage_metrics,
    fit_event_algo_model,
    predict_algo_proba_from_vector,
    predict_algo_proba_with_meta_from_vector,
    save_event_algo_model,
    suggest_rule_thresholds_from_features,
)


def _make_sample(offset: float, *, gyro_bias: float = 0.0, emg_boost: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    emg = np.full((8, 24, 5), offset, dtype=np.float32)
    imu = np.full((6, 16), offset * 0.3, dtype=np.float32)
    emg[0, :, :] += np.linspace(0.0, 1.0, 5, dtype=np.float32)[None, :]
    emg[1, :, 2:] += float(emg_boost)
    imu[5, :] += float(gyro_bias)
    return emg, imu


def _make_dataset(class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    labels = []
    for label, name in enumerate(class_names):
        for _ in range(6):
            if name == "RELAX":
                emg, imu = _make_sample(0.05, gyro_bias=0.0, emg_boost=0.0)
            elif name == "TENSE_OPEN":
                emg, imu = _make_sample(0.9, gyro_bias=0.0, emg_boost=0.7)
            elif name == "THUMB_UP":
                emg, imu = _make_sample(1.7, gyro_bias=0.1, emg_boost=1.2)
            elif name == "WRIST_CW":
                emg, imu = _make_sample(0.8, gyro_bias=2.2, emg_boost=0.4)
            else:
                emg, imu = _make_sample(-0.8, gyro_bias=-2.2, emg_boost=0.3)
            vectors.append(build_event_algo_feature_vector(emg, imu))
            labels.append(label)
    return np.stack(vectors, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int32)


def test_event_algo_v1_roundtrip_predict_and_compat(tmp_path):
    class_names = ["RELAX", "THUMB_UP", "WRIST_CW"]
    vectors, labels = _make_dataset(class_names)
    model = fit_event_algo_model(vectors, labels, class_names=class_names, temperature=0.12, algo_mode=ALGO_MODE_V1)
    assert model.algo_mode == ALGO_MODE_V1

    emg, imu = _make_sample(1.7, gyro_bias=0.1, emg_boost=1.2)
    vector = build_event_algo_feature_vector(emg, imu)
    probs = predict_algo_proba_from_vector(model, vector)
    assert probs.shape == (3,)
    assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-5)

    payload = model.to_json_dict()
    payload.pop("algo_mode", None)
    restored = EventAlgoModel.from_json_dict(payload)
    assert restored.algo_mode == ALGO_MODE_V1

    path = save_event_algo_model(model, tmp_path / "algo_model_v1.json")
    predictor = EventAlgoPredictor(model_path=path)
    pred_probs, meta = predictor.predict_proba_with_meta(emg, imu)
    assert pred_probs.shape == (3,)
    assert bool(meta.get("rule_hit", False)) is False


def test_event_algo_v2_gate_reject_then_accept():
    class_names = ["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW"]
    vectors, labels = _make_dataset(class_names)

    blocked = fit_event_algo_model(
        vectors,
        labels,
        class_names=class_names,
        temperature=0.12,
        algo_mode=ALGO_MODE_V2,
        gate_action_threshold=1.1,
        gate_margin_threshold=0.0,
    )
    emg, imu = _make_sample(1.7, gyro_bias=0.1, emg_boost=1.2)
    vector = build_event_algo_feature_vector(emg, imu)
    probs_blocked, meta_blocked = predict_algo_proba_with_meta_from_vector(blocked, vector)
    assert bool(meta_blocked.get("gate_accepted", True)) is False
    assert bool(meta_blocked.get("stage2_used", True)) is False
    assert int(np.argmax(probs_blocked)) == 0

    open_gate = fit_event_algo_model(
        vectors,
        labels,
        class_names=class_names,
        temperature=0.12,
        algo_mode=ALGO_MODE_V2,
        gate_action_threshold=0.0,
        gate_margin_threshold=-1.0,
    )
    probs_open, meta_open = predict_algo_proba_with_meta_from_vector(open_gate, vector)
    assert bool(meta_open.get("gate_accepted", False)) is True
    assert bool(meta_open.get("stage2_used", False)) is True
    assert int(np.argmax(probs_open)) in {1, 2, 3}


def test_event_algo_rule_priority_over_learning(tmp_path):
    class_names = ["RELAX", "THUMB_UP", "WRIST_CW"]
    vectors, labels = _make_dataset(class_names)
    fitted = fit_event_algo_model(
        vectors,
        labels,
        class_names=class_names,
        temperature=0.12,
        algo_mode=ALGO_MODE_V2,
        gate_action_threshold=0.0,
        gate_margin_threshold=-1.0,
    )
    model = EventAlgoModel(
        class_names=fitted.class_names,
        feature_mean=fitted.feature_mean,
        feature_std=fitted.feature_std,
        centroids=fitted.centroids,
        temperature=fitted.temperature,
        rule_config={
            "enabled": True,
            "wrist_rule_min": 0.2,
            "wrist_rule_margin": 0.01,
            "release_emg_min": 10.0,
            "release_imu_max": 0.1,
            "rule_confidence": 0.96,
        },
        algo_mode=fitted.algo_mode,
        gate_feature_mean=fitted.gate_feature_mean,
        gate_feature_std=fitted.gate_feature_std,
        gate_centroids=fitted.gate_centroids,
        gate_action_threshold=fitted.gate_action_threshold,
        gate_margin_threshold=fitted.gate_margin_threshold,
        action_class_names=fitted.action_class_names,
        action_feature_mean=fitted.action_feature_mean,
        action_feature_std=fitted.action_feature_std,
        action_centroids=fitted.action_centroids,
    )

    path = save_event_algo_model(model, tmp_path / "algo_model_v2_rule.json")
    predictor = EventAlgoPredictor(model_path=path)

    emg, imu = _make_sample(1.5, gyro_bias=3.0, emg_boost=0.6)
    probs, meta = predictor.predict_proba_with_meta(emg, imu)
    assert bool(meta.get("rule_hit", False)) is True
    assert bool(meta.get("stage2_used", True)) is False
    assert int(np.argmax(probs)) == class_names.index("WRIST_CW")


def test_event_algo_stage_metrics_keys():
    class_names = ["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW"]
    vectors, labels = _make_dataset(class_names)
    model = fit_event_algo_model(
        vectors,
        labels,
        class_names=class_names,
        temperature=0.12,
        algo_mode=ALGO_MODE_V2,
        gate_action_threshold=0.55,
        gate_margin_threshold=0.05,
    )
    metrics = compute_algo_stage_metrics(model, vectors, labels, class_names=class_names)
    for key in ["gate_accept_rate", "gate_action_recall", "stage2_action_acc", "rule_hit_rate"]:
        assert key in metrics
        assert 0.0 <= float(metrics[key]) <= 1.0


def test_rule_signal_scores_and_calibration_from_features():
    class_names = ["RELAX", "TENSE_OPEN", "THUMB_UP", "WRIST_CW", "WRIST_CCW"]
    vectors, labels = _make_dataset(class_names)
    # Rebuild synthetic features to compute rule statistics.
    emg_list = []
    imu_list = []
    for label in labels.tolist():
        name = class_names[int(label)]
        if name == "RELAX":
            emg, imu = _make_sample(0.05, gyro_bias=0.0, emg_boost=0.0)
        elif name == "TENSE_OPEN":
            emg, imu = _make_sample(0.9, gyro_bias=0.0, emg_boost=0.7)
        elif name == "THUMB_UP":
            emg, imu = _make_sample(1.7, gyro_bias=0.1, emg_boost=1.2)
        elif name == "WRIST_CW":
            emg, imu = _make_sample(0.8, gyro_bias=2.2, emg_boost=0.4)
        else:
            emg, imu = _make_sample(-0.8, gyro_bias=-2.2, emg_boost=0.3)
        emg_list.append(emg)
        imu_list.append(imu)
    emg_samples = np.asarray(emg_list, dtype=np.float32)
    imu_samples = np.asarray(imu_list, dtype=np.float32)

    scores = compute_rule_signal_scores(emg_samples[0], imu_samples[0])
    for key in ["emg_energy", "imu_motion", "cw_score", "ccw_score", "wrist_peak", "wrist_margin"]:
        assert key in scores
        assert np.isfinite(float(scores[key]))

    report = suggest_rule_thresholds_from_features(
        emg_samples,
        imu_samples,
        labels,
        class_names=class_names,
        fallback={
            "wrist_rule_min": 0.55,
            "wrist_rule_margin": 0.10,
            "release_emg_min": 0.45,
            "release_imu_max": 1.50,
        },
    )
    assert report["status"] in {"ok", "fallback_no_samples"}
    thresholds = report["thresholds"]
    for key in ["wrist_rule_min", "wrist_rule_margin", "release_emg_min", "release_imu_max"]:
        assert key in thresholds
        assert np.isfinite(float(thresholds[key]))
