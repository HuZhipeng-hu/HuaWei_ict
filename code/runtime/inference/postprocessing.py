"""
滑动窗口多数投票后处理

将连续的单次推理结果通过多数投票平滑，过滤偶发误判。
配合置信度阈值，实现"宁可不动也不做错"的稳健控制。

原理:
    最近 N 次推理结果: [FIST, FIST, OK, FIST, FIST]
    多数投票结果:       FIST (4/5 票)
    如果最高票数 < min_count: 输出 None（不确定，保持上一状态）
"""

import logging
from collections import deque, Counter
from typing import Optional, Tuple

from shared.gestures import GestureType, NUM_CLASSES

logger = logging.getLogger(__name__)


class SlidingWindowVoter:
    """
    滑动窗口多数投票器

    对连续推理结果进行平滑过滤，输出稳定的手势预测。

    Args:
        window_size: 投票窗口大小（保留最近 N 次推理结果）
        min_count: 最小投票数（最高票数 >= min_count 才输出结果）
        confidence_threshold: 置信度阈值（单次推理置信度 < 阈值则丢弃）
    """

    def __init__(
        self,
        window_size: int = 5,
        min_count: int = 3,
        confidence_threshold: float = 0.5,
    ):
        if min_count > window_size:
            raise ValueError(
                f"min_count ({min_count}) 不能大于 "
                f"window_size ({window_size})"
            )

        self.window_size = window_size
        self.min_count = min_count
        self.confidence_threshold = confidence_threshold

        self._window: deque = deque(maxlen=window_size)
        self._last_output: Optional[GestureType] = None

    def update(
        self,
        gesture_id: int,
        confidence: float,
    ) -> Optional[GestureType]:
        """
        输入一次推理结果，输出投票后的稳定结果

        Args:
            gesture_id: 单次推理的手势ID
            confidence: 对应的置信度

        Returns:
            投票通过的手势类型，或 None（不确定时保持上一状态）
        """
        # 置信度过低则不纳入投票
        if confidence < self.confidence_threshold:
            logger.debug(
                f"置信度过低 ({confidence:.3f} < {self.confidence_threshold})，"
                f"不纳入投票"
            )
            # 仍然返回上一结果以保持稳定
            return self._last_output

        # 加入投票窗口
        self._window.append(gesture_id)

        # 窗口未满时，仅当所有结果一致才输出
        if len(self._window) < self.window_size:
            if len(set(self._window)) == 1:
                self._last_output = GestureType(gesture_id)
                return self._last_output
            return self._last_output

        # 多数投票
        counter = Counter(self._window)
        most_common_id, most_common_count = counter.most_common(1)[0]

        if most_common_count >= self.min_count:
            result = GestureType(most_common_id)
            if result != self._last_output:
                logger.info(
                    f"手势切换: {self._last_output} → {result} "
                    f"(票数: {most_common_count}/{self.window_size})"
                )
            self._last_output = result
            return result
        else:
            # 票数不足，保持上一状态
            logger.debug(
                f"票数不足: {most_common_id}={most_common_count} "
                f"< {self.min_count}，保持 {self._last_output}"
            )
            return self._last_output

    def reset(self) -> None:
        """重置投票窗口和状态"""
        self._window.clear()
        self._last_output = None

    @property
    def current_gesture(self) -> Optional[GestureType]:
        """当前输出的手势"""
        return self._last_output

    @property
    def window_state(self) -> list:
        """当前窗口内容"""
        return list(self._window)
