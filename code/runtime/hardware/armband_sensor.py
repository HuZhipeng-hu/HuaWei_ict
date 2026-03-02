"""
思知瑞 EMG 臂环传感器驱动

通过串口（USB-Serial）与思知瑞 sEMG 臂环通信，
解析数据帧，提取 8 通道 EMG 数据，并执行降采样。

帧结构（思知瑞协议）:
    帧头: 0xAA 0x55
    数据: 8ch EMG (uint8) + 3ch ACC + 3ch GYRO + 3ch ANGLE + 电量
    帧尾: 0x55 0xAA

使用方式:
    sensor = ArmbandSensor(port="COM3")
    sensor.connect()
    window = sensor.read_window(168)
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, Dict, Any

import numpy as np

from .base import SensorBase

logger = logging.getLogger(__name__)

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# 帧协议常量
FRAME_HEADER = bytes([0xAA, 0x55])
FRAME_TAIL = bytes([0x55, 0xAA])
FRAME_LENGTH = 25  # 总帧长（含头尾）
NUM_EMG_CHANNELS = 8


class ArmbandSensor(SensorBase):
    """
    思知瑞 EMG 臂环传感器

    继承 SensorBase，提供基于串口的数据采集。
    内部使用后台线程持续读取数据并降采样后存入环形缓冲区。

    Args:
        port: 串口端口 (e.g. "COM3", "/dev/ttyUSB0")
              None = 自动检测
        baudrate: 波特率
        device_sampling_rate: 臂环原始采样率 (Hz)
        target_sampling_rate: 目标采样率 (Hz)
        buffer_size: 环形缓冲区容量（目标采样率的采样点数）
        center_value: EMG 归零中心值
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        device_sampling_rate: int = 1000,
        target_sampling_rate: int = 200,
        buffer_size: int = 2000,
        center_value: float = 128.0,
    ):
        self._port = port
        self._baudrate = baudrate
        self._device_rate = device_sampling_rate
        self._target_rate = target_sampling_rate
        self._decimate_ratio = device_sampling_rate // target_sampling_rate
        self._center_value = center_value

        # 串口和线程
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 数据缓冲 (线程安全)
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # 降采样计数器
        self._sample_counter = 0

        # 统计
        self._total_frames = 0
        self._error_frames = 0

    def connect(self) -> bool:
        """连接臂环，启动数据采集线程"""
        if not SERIAL_AVAILABLE:
            logger.error("pyserial 未安装，请运行: pip install pyserial")
            return False

        # 自动检测端口
        port = self._port or self._auto_detect_port()
        if port is None:
            logger.error("未找到可用的串口设备")
            return False

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self._baudrate,
                timeout=1,
            )
            time.sleep(0.5)  # 等待串口稳定

            # 启动采集线程
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop,
                daemon=True,
                name="ArmbandReader",
            )
            self._thread.start()

            logger.info(f"臂环已连接: {port} @ {self._baudrate}bps")
            return True

        except serial.SerialException as e:
            logger.error(f"串口连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """停止采集并断开连接"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._serial and self._serial.is_open:
            self._serial.close()

        logger.info(
            f"臂环已断开 (采集 {self._total_frames} 帧, "
            f"错误 {self._error_frames} 帧)"
        )

    def read(self) -> Optional[np.ndarray]:
        """
        读取缓冲区中最新一帧 EMG 数据

        Returns:
            (NUM_EMG_CHANNELS,) float32，缓冲区空时返回 None
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def read_window(self, window_size: int) -> Optional[np.ndarray]:
        """
        读取缓冲区中最近 window_size 个采样点

        Args:
            window_size: 需要的采样点数

        Returns:
            (window_size, NUM_EMG_CHANNELS) float32
            数据不足时返回 None
        """
        with self._lock:
            if len(self._buffer) < window_size:
                return None
            # 取最近的 window_size 个点
            data = list(self._buffer)[-window_size:]
            return np.array(data, dtype=np.float32)

    def is_connected(self) -> bool:
        return (
            self._serial is not None
            and self._serial.is_open
            and self._running
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "ArmbandSensor",
            "port": self._port,
            "device_rate_hz": self._device_rate,
            "target_rate_hz": self._target_rate,
            "buffer_size": len(self._buffer),
            "total_frames": self._total_frames,
            "error_frames": self._error_frames,
        }

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _read_loop(self) -> None:
        """后台采集线程: 持续读取并解析帧"""
        raw_buffer = bytearray()

        while self._running:
            try:
                if not self._serial or not self._serial.is_open:
                    break

                # 读取可用数据
                available = self._serial.in_waiting
                if available > 0:
                    raw_buffer.extend(self._serial.read(available))
                else:
                    time.sleep(0.001)  # 1ms 轮询间隔
                    continue

                # 从缓冲区中提取帧
                while len(raw_buffer) >= FRAME_LENGTH:
                    # 查找帧头
                    header_pos = raw_buffer.find(FRAME_HEADER)
                    if header_pos < 0:
                        raw_buffer.clear()
                        break

                    # 跳过帧头前的垃圾数据
                    if header_pos > 0:
                        raw_buffer = raw_buffer[header_pos:]

                    # 检查是否有完整帧
                    if len(raw_buffer) < FRAME_LENGTH:
                        break

                    # 校验帧尾
                    frame = raw_buffer[:FRAME_LENGTH]
                    if frame[-2:] != FRAME_TAIL:
                        self._error_frames += 1
                        raw_buffer = raw_buffer[1:]  # 跳过1字节继续搜索
                        continue

                    # 解析帧
                    self._parse_frame(frame)
                    raw_buffer = raw_buffer[FRAME_LENGTH:]

            except Exception as e:
                logger.error(f"采集线程异常: {e}")
                time.sleep(0.1)

    def _parse_frame(self, frame: bytes) -> None:
        """
        解析一帧数据

        帧结构: [AA 55] [8ch EMG] [3ch ACC] [3ch GYRO] [3ch ANGLE] [Battery] [55 AA]
        """
        self._total_frames += 1

        # 降采样: 只保留每 decimate_ratio 帧中的一帧
        self._sample_counter += 1
        if self._sample_counter % self._decimate_ratio != 0:
            return

        # 提取 8 通道 EMG (uint8, 偏移2开始)
        emg_raw = np.array(
            [frame[2 + i] for i in range(NUM_EMG_CHANNELS)],
            dtype=np.float32,
        )

        # 归零 (uint8 中心值为 128)
        emg_centered = emg_raw - self._center_value

        # 存入缓冲区
        with self._lock:
            self._buffer.append(emg_centered)

    @staticmethod
    def _auto_detect_port() -> Optional[str]:
        """自动检测可能的臂环串口"""
        if not SERIAL_AVAILABLE:
            return None

        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc = port.description.lower()
            # 常见的 USB-Serial 芯片
            if any(kw in desc for kw in ["ch340", "cp210", "ftdi", "usb"]):
                logger.info(f"自动检测到串口: {port.device} ({port.description})")
                return port.device

        return None
