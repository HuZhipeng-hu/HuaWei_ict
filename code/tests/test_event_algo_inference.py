from __future__ import annotations

import numpy as np

from event_onset.algo import (
    EventAlgoPredictor,
    build_event_algo_feature_vector,
    fit_event_algo_model,
    predict_algo_proba_from_vector,
    save_event_algo_model,
)


def _make_sample(offset: float) -> tuple[np.ndarray, np.ndarray]:
    emg = np.full((8, 24, 5), offset, dtype=np.float32)
    imu = np.full((6, 16), offset * 0.5, dtype=np.float32)
    emg[0, :, :] += np.linspace(0.0, 1.0, 5, dtype=np.float32)[None, :]
    imu[1, :] += np.linspace(0.0, 0.4, 16, dtype=np.float32)
    return emg, imu


def test_event_algo_model_roundtrip_and_predict(tmp_path):
    class_names = ["RELAX", "THUMB_UP", "WRIST_CW"]
    vectors = []
    labels = []
    for label, offset in enumerate([0.2, 2.0, -1.2]):
        for _ in range(4):
            emg, imu = _make_sample(offset)
            vectors.append(build_event_algo_feature_vector(emg, imu))
            labels.append(label)

    vectors_np = np.stack(vectors, axis=0).astype(np.float32)
    labels_np = np.asarray(labels, dtype=np.int32)
    model = fit_event_algo_model(vectors_np, labels_np, class_names=class_names, temperature=0.12)

    for label, offset in enumerate([0.2, 2.0, -1.2]):
        emg, imu = _make_sample(offset)
        vector = build_event_algo_feature_vector(emg, imu)
        probs = predict_algo_proba_from_vector(model, vector)
        assert probs.shape == (3,)
        assert int(np.argmax(probs)) == label

    path = save_event_algo_model(model, tmp_path / "algo_model.json")
    predictor = EventAlgoPredictor(model_path=path)
    assert predictor.class_names == tuple(class_names)
    emg, imu = _make_sample(2.0)
    probs = predictor.predict_proba(emg, imu)
    assert probs.shape == (3,)
    assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-5)
    assert int(np.argmax(probs)) == 1

