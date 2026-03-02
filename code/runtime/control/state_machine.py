"""
系统状态机

定义义肢控制系统的运行状态及合法状态转换。
用于管理系统的生命周期，确保状态转换的安全性。

状态流转:
    IDLE → CALIBRATING → RUNNING → IDLE
                ↕              ↕
              ERROR ←─────── ERROR
"""

import logging
from enum import Enum, auto
from typing import Optional, Set, Dict

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """系统运行状态"""
    IDLE        = auto()  # 空闲 / 未启动
    CALIBRATING = auto()  # 校准中（传感器初始化、基线采集）
    RUNNING     = auto()  # 正常运行（控制循环活跃）
    ERROR       = auto()  # 错误状态（需要恢复或重启）
    STOPPING    = auto()  # 正在停止（优雅关闭中）


# 合法的状态转换表
_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
    SystemState.IDLE:        {SystemState.CALIBRATING, SystemState.RUNNING, SystemState.ERROR},
    SystemState.CALIBRATING: {SystemState.RUNNING, SystemState.ERROR, SystemState.IDLE},
    SystemState.RUNNING:     {SystemState.STOPPING, SystemState.ERROR},
    SystemState.ERROR:       {SystemState.IDLE},
    SystemState.STOPPING:    {SystemState.IDLE, SystemState.ERROR},
}


class SystemStateMachine:
    """
    系统状态机

    管理义肢控制系统的生命周期状态，确保只有合法的状态转换能够发生。
    非法转换会记录警告但不抛异常（保障系统鲁棒性）。

    Usage:
        sm = SystemStateMachine()
        sm.transition_to(SystemState.CALIBRATING)
        sm.transition_to(SystemState.RUNNING)
        sm.transition_to(SystemState.STOPPING)
        sm.transition_to(SystemState.IDLE)
    """

    def __init__(self):
        self._state = SystemState.IDLE
        self._error_message: Optional[str] = None

    @property
    def state(self) -> SystemState:
        """当前状态"""
        return self._state

    @property
    def is_running(self) -> bool:
        """是否在运行状态"""
        return self._state == SystemState.RUNNING

    @property
    def is_error(self) -> bool:
        """是否在错误状态"""
        return self._state == SystemState.ERROR

    @property
    def error_message(self) -> Optional[str]:
        """错误信息（仅在 ERROR 状态有效）"""
        return self._error_message

    def transition_to(self, new_state: SystemState) -> bool:
        """
        尝试转换到新状态

        Args:
            new_state: 目标状态

        Returns:
            True 如果转换成功，False 如果转换非法
        """
        allowed = _TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            logger.warning(
                f"非法状态转换: {self._state.name} → {new_state.name}，"
                f"当前状态允许转换到: {[s.name for s in allowed]}"
            )
            return False

        old_state = self._state
        self._state = new_state

        # 离开 ERROR 状态时清除错误信息
        if old_state == SystemState.ERROR:
            self._error_message = None

        logger.info(f"状态转换: {old_state.name} → {new_state.name}")
        return True

    def set_error(self, message: str) -> bool:
        """
        进入错误状态

        Args:
            message: 错误描述

        Returns:
            True 如果成功进入错误状态
        """
        success = self.transition_to(SystemState.ERROR)
        if success:
            self._error_message = message
            logger.error(f"系统错误: {message}")
        return success

    def reset(self) -> bool:
        """
        从错误状态重置回 IDLE

        Returns:
            True 如果重置成功
        """
        if self._state != SystemState.ERROR:
            logger.warning(f"只能从 ERROR 状态重置，当前状态: {self._state.name}")
            return False
        return self.transition_to(SystemState.IDLE)
