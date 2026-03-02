"""
传感器和执行器抽象基类

定义所有硬件驱动必须实现的接口，确保上层控制器
可以无差别地使用不同的传感器/执行器实现。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import numpy as np

from shared.gestures import GestureType


class SensorBase(ABC):
    """
    传感器抽象基类

    所有 EMG 传感器驱动必须实现此接口。
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        连接传感器

        Returns:
            连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开传感器连接"""
        pass

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        读取一帧传感器数据

        Returns:
            (num_channels,) EMG 数据，失败返回 None
        """
        pass

    @abstractmethod
    def read_window(self, window_size: int) -> Optional[np.ndarray]:
        """
        读取一个窗口的传感器数据

        Args:
            window_size: 窗口大小（采样点数）

        Returns:
            (window_size, num_channels) EMG 数据，数据不足返回 None
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """传感器是否已连接"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取传感器信息"""
        pass


class ActuatorBase(ABC):
    """
    执行器抽象基类

    所有义肢执行器驱动必须实现此接口。
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        连接执行器

        Returns:
            连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开执行器连接并回到安全位置"""
        pass

    @abstractmethod
    def execute_gesture(self, gesture: GestureType) -> None:
        """
        执行指定手势

        Args:
            gesture: 目标手势类型
        """
        pass

    @abstractmethod
    def set_finger_angles(self, angles: List[float]) -> None:
        """
        直接设置各手指角度

        Args:
            angles: 5 个手指的目标角度（度）
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """执行器是否已连接"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取执行器信息"""
        pass
