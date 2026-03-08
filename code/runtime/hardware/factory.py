"""Hardware factory helpers."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from shared.config import HardwareConfig
from shared.gestures import NUM_EMG_CHANNELS, GestureType

from .armband_sensor import ArmbandSensor
from .base import ActuatorBase, SensorBase
from .pca9685_actuator import PCA9685Actuator

logger = logging.getLogger(__name__)


class StandaloneSensor(SensorBase):
    def __init__(self, num_channels: int = NUM_EMG_CHANNELS):
        self._num_channels = num_channels
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("Standalone sensor connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Standalone sensor disconnected")

    def read(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        return np.random.randn(self._num_channels).astype(np.float32) * 10

    def read_window(self, window_size: int) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        return np.random.randn(window_size, self._num_channels).astype(np.float32) * 10

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self):
        return {"type": "StandaloneSensor", "channels": self._num_channels}


class StandaloneActuator(ActuatorBase):
    def __init__(self):
        self._connected = False
        self._current_gesture: Optional[GestureType] = None

    def connect(self) -> bool:
        self._connected = True
        logger.info("Standalone actuator connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Standalone actuator disconnected")

    def execute_gesture(self, gesture: GestureType) -> None:
        if gesture != self._current_gesture:
            logger.info("[Standalone] execute gesture: %s", gesture.name)
            self._current_gesture = gesture

    def set_finger_angles(self, angles: list) -> None:
        logger.debug("[Standalone] set angles: %s", angles)

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self):
        return {
            "type": "StandaloneActuator",
            "current_gesture": self._current_gesture.name if self._current_gesture else None,
        }


def create_sensor(config: HardwareConfig) -> SensorBase:
    if config.sensor_mode == "standalone":
        return StandaloneSensor(num_channels=6)

    return ArmbandSensor(
        port=config.sensor_port,
        baudrate=config.sensor_baudrate,
        device_sampling_rate=config.armband_sampling_rate,
        target_sampling_rate=config.target_sampling_rate,
        buffer_size=config.sensor_buffer_size,
    )


def create_actuator(config: HardwareConfig) -> ActuatorBase:
    if config.actuator_mode == "standalone":
        return StandaloneActuator()

    return PCA9685Actuator(
        i2c_bus=config.actuator_i2c_bus,
        i2c_address=config.actuator_i2c_address,
        frequency=config.actuator_frequency,
        angle_open=config.servo_angle_open,
        angle_half=config.servo_angle_half,
        angle_closed=config.servo_angle_closed,
        channels=config.actuator_channels,
    )
