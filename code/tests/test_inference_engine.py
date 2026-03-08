import numpy as np
import pytest

from runtime.inference.engine import InferenceEngine


def test_engine_prefers_loaded_model_shape_when_available():
    engine = InferenceEngine("missing_model.mindir", expected_input_shape=(1, 12, 24, 6))
    engine._input_shape = (1, 6, 24, 6)

    assert engine.get_input_shape() == (1, 6, 24, 6)


def test_engine_validation_uses_loaded_shape_not_expected_fallback():
    engine = InferenceEngine("missing_model.mindir", expected_input_shape=(1, 12, 24, 6))
    engine._input_shape = (1, 6, 24, 6)

    with pytest.raises(ValueError, match=r"expected \(1, 6, 24, 6\)"):
        engine._validate_input(np.zeros((12, 24, 6), dtype=np.float32))
