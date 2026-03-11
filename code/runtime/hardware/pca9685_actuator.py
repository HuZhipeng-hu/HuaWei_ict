"""
PCA9685 PWM 舵机执行器

通过 I2C 总线控制 PCA9685 PWM 驱动板，驱动 5 个手指舵机。
每个舵机通过 PWM 脉宽控制角度 (0°~180°)。

接线:
    PCA9685 Channel 0 → 拇指舵机
    PCA9685 Channel 1 → 食指舵机
    PCA9685 Channel 2 → 中指舵机
    PCA9685 Channel 3 → 无名指舵机
    PCA9685 Channel 4 → 小指舵机
"""

import time
import logging
from typing import List, Dict, Any, Optional

from .base import ActuatorBase
from shared.gestures import (
    GestureType, NUM_FINGERS,
    get_finger_angles,
)

logger = logging.getLogger(__name__)

try:
    from smbus2 import SMBus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False


# PCA9685 寄存器地址
PCA9685_MODE1 = 0x00
PCA9685_PRESCALE = 0xFE
PCA9685_LED0_ON_L = 0x06


class PCA9685Actuator(ActuatorBase):
    """
    PCA9685 PWM 舵机执行器

    继承 ActuatorBase，通过 I2C → PCA9685 → PWM → 舵机
    控制义肢的 5 个手指。

    Args:
        i2c_bus: I2C 总线编号
        i2c_address: PCA9685 I2C 地址
        frequency: PWM 频率 (Hz)
        angle_open: 张开角度 (度)
        angle_half: 半弯角度 (度)
        angle_closed: 闭合角度 (度)
        min_pulse_ms: 最小脉宽 (ms，对应 0°)
        max_pulse_ms: 最大脉宽 (ms，对应 180°)
        channels: 5 个手指对应的 PCA9685 通道编号
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = 0x40,
        frequency: int = 50,
        angle_open: float = 0.0,
        angle_half: float = 90.0,
        angle_closed: float = 180.0,
        min_pulse_ms: float = 0.5,
        max_pulse_ms: float = 2.5,
        channels: Optional[List[int]] = None,
    ):
        self._bus_num = i2c_bus
        self._address = i2c_address
        self._frequency = frequency
        self._angle_open = angle_open
        self._angle_half = angle_half
        self._angle_closed = angle_closed
        self._min_pulse = min_pulse_ms
        self._max_pulse = max_pulse_ms
        self._channels = channels or [0, 1, 2, 3, 4]  # 默认通道映射

        self._bus: Optional[SMBus] = None
        self._connected = False
        self._current_angles = [angle_open] * NUM_FINGERS
        self._current_gesture: Optional[GestureType] = None

    def connect(self) -> bool:
        """初始化 PCA9685 并将所有手指归零"""
        if not SMBUS_AVAILABLE:
            logger.warning(
                "smbus2 未安装，执行器使用模拟模式。"
                "在 Orange Pi 上请安装: pip install smbus2"
            )
            self._connected = True
            return True

        try:
            self._bus = SMBus(self._bus_num)

            # 复位 PCA9685
            self._bus.write_byte_data(self._address, PCA9685_MODE1, 0x00)
            time.sleep(0.01)

            # 设置 PWM 频率
            self._set_frequency(self._frequency)

            self._connected = True

            # 归零: 所有手指张开
            self.execute_gesture(GestureType.RELAX)

            logger.info(
                f"执行器已连接: I2C bus={self._bus_num} "
                f"addr=0x{self._address:02X} freq={self._frequency}Hz"
            )
            return True

        except Exception as e:
            logger.error(f"执行器连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """回到安全位置并断开"""
        if self._connected:
            # 安全姿态: 张开所有手指
            try:
                self.execute_gesture(GestureType.RELAX)
                time.sleep(0.5)
            except Exception:
                pass

        if self._bus:
            self._bus.close()

        self._connected = False
        logger.info("执行器已断开")

    def execute_gesture(self, gesture: GestureType) -> None:
        """
        执行指定手势

        自动将手势转换为 5 个手指角度并驱动舵机。
        """
        if not self._connected:
            logger.warning("执行器未连接，忽略手势指令")
            return

        angles = get_finger_angles(
            gesture,
            angle_open=self._angle_open,
            angle_half=self._angle_half,
            angle_closed=self._angle_closed,
        )

        self.set_finger_angles(angles)
        self._current_gesture = gesture

    def set_finger_angles(self, angles: List[float]) -> None:
        """直接设置各手指角度"""
        if len(angles) != NUM_FINGERS:
            raise ValueError(f"需要 {NUM_FINGERS} 个角度，实际 {len(angles)} 个")

        for i, angle in enumerate(angles):
            channel = self._channels[i]
            self._set_servo_angle(channel, angle)
            self._current_angles[i] = angle

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "PCA9685Actuator",
            "i2c_bus": self._bus_num,
            "i2c_address": f"0x{self._address:02X}",
            "frequency": self._frequency,
            "connected": self._connected,
            "current_gesture": (
                self._current_gesture.name
                if self._current_gesture
                else None
            ),
            "current_angles": self._current_angles.copy(),
        }

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _set_frequency(self, freq: int) -> None:
        """设置 PCA9685 PWM 频率"""
        if not self._bus:
            return

        prescale = int(25000000.0 / (4096.0 * freq) - 1)
        old_mode = self._bus.read_byte_data(self._address, PCA9685_MODE1)

        # 进入睡眠模式设置预分频
        self._bus.write_byte_data(
            self._address, PCA9685_MODE1, (old_mode & 0x7F) | 0x10,
        )
        self._bus.write_byte_data(self._address, PCA9685_PRESCALE, prescale)
        self._bus.write_byte_data(self._address, PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self._bus.write_byte_data(
            self._address, PCA9685_MODE1, old_mode | 0x80,
        )

    def _set_servo_angle(self, channel: int, angle: float) -> None:
        """
        设置单个舵机角度

        将角度映射为 PWM 脉宽，再转换为 PCA9685 的 12-bit 计数值。
        """
        # 限幅
        angle = max(0.0, min(180.0, angle))

        # 角度 → 脉宽 (ms)
        pulse_ms = self._min_pulse + (
            (self._max_pulse - self._min_pulse) * angle / 180.0
        )

        # 脉宽 → PCA9685 计数值 (12-bit, 即 0~4095)
        period_ms = 1000.0 / self._frequency
        pulse_count = int(pulse_ms / period_ms * 4096)

        if self._bus:
            # 写入 PCA9685 寄存器
            reg = PCA9685_LED0_ON_L + 4 * channel
            self._bus.write_byte_data(self._address, reg, 0)       # ON_L
            self._bus.write_byte_data(self._address, reg + 1, 0)   # ON_H
            self._bus.write_byte_data(
                self._address, reg + 2, pulse_count & 0xFF,
            )  # OFF_L
            self._bus.write_byte_data(
                self._address, reg + 3, pulse_count >> 8,
            )  # OFF_H
