"""Inference backends for event-onset runtime (MindIR Lite + CKPT)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from event_onset.config import EventModelConfig
from event_onset.evaluate import load_event_model_from_checkpoint

try:
    import mindspore as ms
    from mindspore import Tensor, context
except Exception:
    ms = None  # type: ignore
    Tensor = None  # type: ignore
    context = None  # type: ignore

try:
    from mindspore_lite import Context as LiteContext
    from mindspore_lite import Model as LiteModel
except Exception:
    LiteContext = None  # type: ignore
    LiteModel = None  # type: ignore


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return (exp_logits / np.sum(exp_logits)).astype(np.float32)


def _shape_matches(actual: Sequence[int], expected: Sequence[int]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if int(e) < 0:
            continue
        if int(a) != int(e):
            return False
    return True


@dataclass(frozen=True)
class EventModelMetadata:
    inputs: dict[str, tuple[int, ...]]
    output_dtype: str | None = None
    class_names: tuple[str, ...] = ()

    @staticmethod
    def load(path: str | Path | None) -> "EventModelMetadata | None":
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Event model metadata not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        inputs: dict[str, tuple[int, ...]] = {}
        for item in payload.get("inputs", []):
            name = str(item.get("name") or "").strip().lower()
            if not name:
                continue
            shape = tuple(int(x) for x in item.get("shape", []))
            inputs[name] = shape
        class_names = tuple(str(x) for x in payload.get("class_names", []))
        output_dtype = payload.get("output", {}).get("dtype")
        return EventModelMetadata(inputs=inputs, output_dtype=output_dtype, class_names=class_names)


class EventPredictor:
    """Backend-agnostic predictor for event-onset runtime."""

    def __init__(
        self,
        *,
        backend: str,
        model_config: EventModelConfig,
        device_target: str = "CPU",
        checkpoint_path: str | Path | None = None,
        model_path: str | Path | None = None,
        model_metadata_path: str | Path | None = None,
    ):
        normalized = str(backend).strip().lower()
        if normalized not in {"ckpt", "lite"}:
            raise ValueError(f"Unsupported backend: {backend}. Expected 'ckpt' or 'lite'.")
        self.backend = normalized
        self.model_config = model_config
        self.device_target = device_target
        # CKPT backend does not need model metadata; keep it optional to avoid false file-not-found failures.
        self.metadata = EventModelMetadata.load(model_metadata_path) if self.backend == "lite" else None
        self._ckpt_model = None
        self._lite_model = None
        self._emg_input_index = 0
        self._imu_input_index = 1
        self._expected_emg_shape: tuple[int, ...] | None = None
        self._expected_imu_shape: tuple[int, ...] | None = None

        if self.backend == "ckpt":
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when backend='ckpt'.")
            self._load_ckpt(checkpoint_path)
        else:
            if model_path is None:
                raise ValueError("model_path is required when backend='lite'.")
            self._load_lite(model_path)

    def _load_ckpt(self, checkpoint_path: str | Path) -> None:
        if ms is None or Tensor is None or context is None:
            raise RuntimeError("MindSpore is not available for ckpt runtime.")
        context.set_context(mode=context.GRAPH_MODE, device_target=self.device_target)
        self._ckpt_model = load_event_model_from_checkpoint(checkpoint_path, self.model_config)
        self._expected_emg_shape = (
            1,
            int(self.model_config.emg_in_channels),
            int(self.model_config.emg_freq_bins),
            int(self.model_config.emg_time_frames),
        )
        self._expected_imu_shape = (
            1,
            int(self.model_config.imu_input_dim),
            int(self.model_config.imu_num_steps),
        )

    def _load_lite(self, model_path: str | Path) -> None:
        if LiteModel is None or LiteContext is None:
            raise RuntimeError("mindspore_lite is not available for lite runtime.")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"MindIR model not found: {path}")

        lite_context = LiteContext()
        target = str(self.device_target).strip().lower()
        if hasattr(lite_context, "target") and target:
            try:
                lite_context.target = [target]
            except Exception:
                pass
        self._lite_model = LiteModel()
        self._lite_model.build_from_file(str(path), model_type=0, context=lite_context)

        raw_inputs = self._lite_model.get_inputs()
        if len(raw_inputs) != 2:
            raise ValueError(f"Expected 2 inputs for event MindIR, got {len(raw_inputs)}.")

        shapes = [tuple(int(x) for x in item.shape) for item in raw_inputs]
        rank4_indices = [idx for idx, shape in enumerate(shapes) if len(shape) == 4]
        rank3_indices = [idx for idx, shape in enumerate(shapes) if len(shape) == 3]
        if len(rank4_indices) != 1 or len(rank3_indices) != 1:
            raise ValueError(f"Cannot determine EMG/IMU input order from shapes: {shapes}")

        self._emg_input_index = rank4_indices[0]
        self._imu_input_index = rank3_indices[0]
        self._expected_emg_shape = shapes[self._emg_input_index]
        self._expected_imu_shape = shapes[self._imu_input_index]

        if self.metadata is not None:
            emg_shape = self.metadata.inputs.get("emg")
            imu_shape = self.metadata.inputs.get("imu")
            if emg_shape is not None and not _shape_matches(self._expected_emg_shape, emg_shape):
                raise ValueError(
                    f"MindIR EMG input shape {self._expected_emg_shape} mismatches metadata {emg_shape}."
                )
            if imu_shape is not None and not _shape_matches(self._expected_imu_shape, imu_shape):
                raise ValueError(
                    f"MindIR IMU input shape {self._expected_imu_shape} mismatches metadata {imu_shape}."
                )

    def _validate_inputs(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emg = np.asarray(emg_feature, dtype=np.float32)
        imu = np.asarray(imu_feature, dtype=np.float32)
        if emg.ndim == 3:
            emg = emg[np.newaxis, ...]
        if imu.ndim == 2:
            imu = imu[np.newaxis, ...]
        if emg.ndim != 4:
            raise ValueError(f"Invalid EMG input ndim={emg.ndim}, expected 3 or 4.")
        if imu.ndim != 3:
            raise ValueError(f"Invalid IMU input ndim={imu.ndim}, expected 2 or 3.")
        if self._expected_emg_shape is not None and not _shape_matches(emg.shape, self._expected_emg_shape):
            raise ValueError(f"EMG shape mismatch: got {tuple(emg.shape)}, expected {self._expected_emg_shape}.")
        if self._expected_imu_shape is not None and not _shape_matches(imu.shape, self._expected_imu_shape):
            raise ValueError(f"IMU shape mismatch: got {tuple(imu.shape)}, expected {self._expected_imu_shape}.")
        return emg, imu

    def predict_proba(self, emg_feature: np.ndarray, imu_feature: np.ndarray) -> np.ndarray:
        emg, imu = self._validate_inputs(emg_feature, imu_feature)

        if self.backend == "ckpt":
            assert self._ckpt_model is not None
            logits = self._ckpt_model(Tensor(emg, ms.float32), Tensor(imu, ms.float32)).asnumpy()[0]
            return _softmax(logits)

        assert self._lite_model is not None
        inputs = self._lite_model.get_inputs()
        inputs[self._emg_input_index].set_data_from_numpy(emg)
        inputs[self._imu_input_index].set_data_from_numpy(imu)
        outputs = self._lite_model.predict(inputs)
        logits = outputs[0].get_data_to_numpy()
        if logits.ndim == 2:
            logits = logits[0]
        return _softmax(logits)
