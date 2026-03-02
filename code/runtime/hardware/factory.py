"""
硬件工厂 — 根据配置创建传感器和执行器实例

支持两种模式:
- 真实硬件模式: 连接物理设备
- standalone 模式: 模拟硬件，用于开发调试
"""

import logging
from typing import Optional

import numpy as np

from .base import SensorBase, ActuatorBase
from .armband_sensor import ArmbandSensor
from .pca9685_actuator import PCA9685Actuator
from shared.config import HardwareConfig
from shared.gestures import GestureType, NUM_FINGERS, NUM_EMG_CHANNELS

logger = logging.getLogger(__name__)


# =============================================================================
# 模拟传感器（standalone 模式）
# =============================================================================

class StandaloneSensor(SensorBase):
    """
    模拟传感器

    生成随机 EMG 数据，用于在没有物理臂环时测试系统。
    """

    def __init__(self, num_channels: int = NUM_EMG_CHANNELS):
        self._num_channels = num_channels
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("模拟传感器已连接")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("模拟传感器已断开")

    def read(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        return np.random.randn(self._num_channels).astype(np.float32) * 10

    def read_window(self, window_size: int) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        return np.random.randn(
            window_size, self._num_channels,
        ).astype(np.float32) * 10

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self):
        return {"type": "StandaloneSensor", "channels": self._num_channels}


# =============================================================================
# 模拟执行器（standalone 模式）
# =============================================================================

class StandaloneActuator(ActuatorBase):
    """
    模拟执行器

    仅打印日志，不驱动真实舵机。
    """

    def __init__(self):
        self._connected = False
        self._current_gesture: Optional[GestureType] = None

    def connect(self) -> bool:
        self._connected = True
        logger.info("模拟执行器已连接")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("模拟执行器已断开")

    def execute_gesture(self, gesture: GestureType) -> None:
        if gesture != self._current_gesture:
            logger.info(f"[模拟] 执行手势: {gesture.name}")
            self._current_gesture = gesture

    def set_finger_angles(self, angles: list) -> None:
        logger.debug(f"[模拟] 设置角度: {angles}")

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self):
        return {
            "type": "StandaloneActuator",
            "current_gesture": (
                self._current_gesture.name
                if self._current_gesture
                else None
            ),
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_sensor(config: HardwareConfig) -> SensorBase:
    """
    根据配置创建传感器实例

    Args:
        config: 硬件配置

    Returns:
        传感器实例（真实或模拟）
    """
    if config.sensor_mode == "standalone":
        return StandaloneSensor()

    return ArmbandSensor(
        port=config.sensor_port,
        baudrate=config.sensor_baudrate,
        device_sampling_rate=config.armband_sampling_rate,
        target_sampling_rate=200,  # 目标采样率
        buffer_size=config.sensor_buffer_size,
    )


def create_actuator(config: HardwareConfig) -> ActuatorBase:
    """
    根据配置创建执行器实例

    Args:
        config: 硬件配置

    Returns:
        执行器实例（真实或模拟）
    """
    if config.actuator_mode == "standalone":
        return StandaloneActuator()

    return PCA9685Actuator(
        i2c_bus=config.actuator_i2c_bus,
        i2c_address=config.actuator_i2c_address,
        frequency=config.actuator_frequency,
        angle_open=config.servo_angle_open,
        angle_half=config.servo_angle_half,
        angle_closed=config.servo_angle_closed,
    )
