"""
MindSpore Lite 推理引擎

加载 .mindir 模型文件，执行实时推理。
支持性能统计（延迟 P50/P95/P99）。

使用方式:
    engine = InferenceEngine(model_path="models/neurogrip.mindir")
    gesture_id, confidence = engine.predict(spectrogram)
"""

import time
import logging
from pathlib import Path
from collections import deque
from typing import Tuple, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mindspore_lite as mslite
    MSLITE_AVAILABLE = True
except ImportError:
    MSLITE_AVAILABLE = False


class InferenceEngine:
    """
    MindSpore Lite 推理引擎

    封装 MindSpore Lite 的模型加载和推理，提供简洁的
    predict(spectrogram) → (gesture_id, confidence) 接口。

    Args:
        model_path: .mindir 模型文件路径
        device: 推理设备 ("CPU" / "GPU" / "NPU")
        num_threads: CPU 推理线程数
        latency_window: 延迟统计的滑动窗口大小
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        num_threads: int = 4,
        latency_window: int = 100,
    ):
        self.model_path = model_path
        self._latencies = deque(maxlen=latency_window)
        self._inference_count = 0

        if not MSLITE_AVAILABLE:
            logger.warning(
                "MindSpore Lite 未安装，推理引擎将使用模拟模式。"
                "实际部署请安装 mindspore_lite。"
            )
            self._model = None
            self._mock_mode = True
            return

        self._mock_mode = False

        # 加载模型
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 构建推理上下文
        cpu_context = mslite.Context()
        cpu_context.target = [device.lower()]
        cpu_context.cpu.thread_num = num_threads

        # 加载模型
        self._model = mslite.Model()
        self._model.build_from_file(str(model_file), mslite.ModelType.MINDIR, cpu_context)

        logger.info(f"推理引擎已就绪: {model_path}")
        logger.info(f"  设备: {device}, 线程数: {num_threads}")

    def predict(
        self,
        spectrogram: np.ndarray,
    ) -> Tuple[int, float]:
        """
        执行一次推理

        Args:
            spectrogram: (C, F, T) 或 (1, C, F, T) 时频谱图，float32

        Returns:
            (gesture_id, confidence):
                gesture_id: 预测的手势类别索引
                confidence: 预测置信度 (0~1)
        """
        # 确保 4D: (1, C, F, T)
        if spectrogram.ndim == 3:
            spectrogram = spectrogram[np.newaxis, ...]
        spectrogram = spectrogram.astype(np.float32)

        start_time = time.perf_counter()

        if self._mock_mode:
            # 模拟模式: 返回随机预测
            probabilities = self._mock_predict(spectrogram)
        else:
            # 真实推理
            probabilities = self._real_predict(spectrogram)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._latencies.append(elapsed_ms)
        self._inference_count += 1

        # 取最大概率
        gesture_id = int(np.argmax(probabilities))
        confidence = float(probabilities[gesture_id])

        return gesture_id, confidence

    def _real_predict(self, input_data: np.ndarray) -> np.ndarray:
        """使用 MindSpore Lite 执行真实推理"""
        inputs = self._model.get_inputs()
        inputs[0].set_data_from_numpy(input_data)

        outputs = self._model.predict(inputs)
        logits = outputs[0].get_data_to_numpy()

        # Softmax
        probabilities = self._softmax(logits[0])
        return probabilities

    def _mock_predict(self, input_data: np.ndarray) -> np.ndarray:
        """模拟推理（无 MindSpore Lite 时使用）"""
        # 模拟 ~15ms 推理延迟
        time.sleep(0.015)
        # 返回均匀分布 + 噪声
        from shared.gestures import NUM_CLASSES
        logits = np.random.randn(NUM_CLASSES).astype(np.float32)
        return self._softmax(logits)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的 Softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_latency_stats(self) -> Dict[str, float]:
        """
        获取推理延迟统计

        Returns:
            {
                "count": 推理次数,
                "mean_ms": 平均延迟,
                "p50_ms": 中位数延迟,
                "p95_ms": P95 延迟,
                "p99_ms": P99 延迟,
            }
        """
        if not self._latencies:
            return {"count": 0}

        latencies = np.array(self._latencies)
        return {
            "count": self._inference_count,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
