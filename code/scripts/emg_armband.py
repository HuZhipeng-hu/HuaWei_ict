# -*- coding: utf-8 -*-
"""
emg_armband.py - EMG 臂带 Python 类库
======================================
参考 myo-python 风格设计的 EMG 臂带接口库。

功能：
  - Hub / Device / DeviceListener 事件回调模式
  - EMG 原始数据（10包 × 8通道）
  - IMU 数据（加速度 / 陀螺仪 / 欧拉角）
  - 电池电量
  - CSV 数据记录器
  - 手势回调扩展接口

用法：
  import emg_armband as emg

  class MyListener(emg.DeviceListener):
      def on_emg_data(self, event):
          print(event.emg)        # 10×8 EMG 数据
      def on_imu_data(self, event):
          print(event.acceleration, event.gyroscope, event.orientation)

  hub = emg.Hub(port='COM4')
  hub.run(MyListener())

依赖: pip install pyserial numpy
"""

__version__ = '1.0.0'
__author__ = 'EMG Armband SDK'

import serial
import struct
import time
import threading
import csv
import os
import sys
import numpy as np
from collections import deque
from enum import Enum, auto
from typing import List, Optional, Callable, Dict, Any


# ==================== 常量 ====================

class EmgMode(Enum):
    """EMG 数据模式"""
    RAW = auto()          # 原始 uint8 数据
    FILTERED = auto()     # 滤波后数据（预留）
    PREPROCESSED = auto() # 预处理数据（预留）


class StreamType(Enum):
    """数据流类型"""
    EMG = auto()
    IMU = auto()
    BATTERY = auto()
    ALL = auto()


# 帧协议常量
FRAME_HEADER = b'\xAA\xAA'
FRAME_TAIL = 0x55
MIN_FRAME_LEN = 6

# 默认串口配置
DEFAULT_PORT = 'COM4'
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 0.5

# 数据维度
NUM_EMG_PACKS = 10       # EMG 数据包数量
NUM_EMG_CHANNELS = 8     # 每个包的通道数
TOTAL_EMG_CHANNELS = NUM_EMG_PACKS * NUM_EMG_CHANNELS  # 80

# 载荷结构（字节偏移）
PAYLOAD_TIMESTAMP_BYTES = 4
PAYLOAD_ACC_BYTES = 3
PAYLOAD_GYRO_BYTES = 3
PAYLOAD_ANGLE_BYTES = 3
PAYLOAD_EMG_BYTES = NUM_EMG_PACKS * NUM_EMG_CHANNELS  # 80
PAYLOAD_BATTERY_BYTES = 1
EXPECTED_PAYLOAD = (PAYLOAD_TIMESTAMP_BYTES + PAYLOAD_ACC_BYTES +
                    PAYLOAD_GYRO_BYTES + PAYLOAD_ANGLE_BYTES +
                    PAYLOAD_EMG_BYTES + PAYLOAD_BATTERY_BYTES)  # 94


# ==================== 数据事件类 ====================

class Vector3:
    """三维向量"""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def to_list(self):
        return [self.x, self.y, self.z]

    def to_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)


class Orientation:
    """欧拉角姿态（Pitch, Roll, Yaw）"""
    __slots__ = ('pitch', 'roll', 'yaw')

    def __init__(self, pitch=0, roll=0, yaw=0):
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

    def __repr__(self):
        return f"Orientation(pitch={self.pitch}, roll={self.roll}, yaw={self.yaw})"

    def __iter__(self):
        yield self.pitch
        yield self.roll
        yield self.yaw

    def to_degrees(self, scale=180.0 / 127.0):
        """将原始值转换为角度"""
        return Orientation(
            self.pitch * scale,
            self.roll * scale,
            self.yaw * scale
        )

    def to_list(self):
        return [self.pitch, self.roll, self.yaw]


class EmgEvent:
    """EMG 数据事件"""
    __slots__ = ('timestamp', 'emg', 'emg_flat', '_device')

    def __init__(self, timestamp, emg_data, device=None):
        self.timestamp = timestamp
        self.emg = emg_data           # List[List[int]]: 10×8
        self.emg_flat = None          # 懒加载的扁平化数据
        self._device = device

    @property
    def emg_pack(self):
        """获取 10 个 EMG 包，每包 8 通道"""
        return self.emg

    def get_pack(self, index):
        """获取第 index 个 EMG 包（0-9），返回 8 个通道值"""
        if 0 <= index < NUM_EMG_PACKS:
            return self.emg[index]
        raise IndexError(f"EMG包索引 {index} 超出范围 [0, {NUM_EMG_PACKS-1}]")

    def get_channel(self, pack_index, channel_index):
        """获取指定包的指定通道值"""
        return self.emg[pack_index][channel_index]

    def flatten(self):
        """将 10×8 数据展平为 80 个值"""
        if self.emg_flat is None:
            self.emg_flat = [ch for pack in self.emg for ch in pack]
        return self.emg_flat

    def to_numpy(self):
        """转换为 numpy 数组 (10, 8)"""
        return np.array(self.emg, dtype=np.uint8)

    def __repr__(self):
        return f"EmgEvent(ts={self.timestamp}, packs={NUM_EMG_PACKS}×{NUM_EMG_CHANNELS})"


class ImuEvent:
    """IMU 数据事件（加速度 + 陀螺仪 + 姿态角）"""
    __slots__ = ('timestamp', 'acceleration', 'gyroscope', 'orientation', '_device')

    def __init__(self, timestamp, acc, gyro, orientation, device=None):
        self.timestamp = timestamp
        self.acceleration = acc         # Vector3
        self.gyroscope = gyro           # Vector3
        self.orientation = orientation  # Orientation
        self._device = device

    def __repr__(self):
        return (f"ImuEvent(ts={self.timestamp}, "
                f"acc={self.acceleration}, gyro={self.gyroscope}, "
                f"orient={self.orientation})")


class BatteryEvent:
    """电池事件"""
    __slots__ = ('timestamp', 'level', '_device')

    def __init__(self, timestamp, level, device=None):
        self.timestamp = timestamp
        self.level = level  # 0-100
        self._device = device

    def __repr__(self):
        return f"BatteryEvent(ts={self.timestamp}, level={self.level}%)"


class FrameEvent:
    """完整帧事件（包含所有数据）"""
    __slots__ = ('timestamp', 'emg_event', 'imu_event', 'battery_event')

    def __init__(self, timestamp, emg_event, imu_event, battery_event):
        self.timestamp = timestamp
        self.emg_event = emg_event
        self.imu_event = imu_event
        self.battery_event = battery_event


# ==================== 设备监听器（回调接口）====================

class DeviceListener:
    """
    设备事件监听器基类（类似 myo.DeviceListener）
    用户继承此类并覆写需要的回调方法
    """

    def on_connected(self, device):
        """设备连接成功"""
        pass

    def on_disconnected(self, device):
        """设备断开连接"""
        pass

    def on_emg_data(self, event: EmgEvent):
        """收到 EMG 数据"""
        pass

    def on_imu_data(self, event: ImuEvent):
        """收到 IMU 数据（加速度 + 陀螺仪 + 姿态角）"""
        pass

    def on_battery(self, event: BatteryEvent):
        """收到电池电量更新"""
        pass

    def on_frame(self, event: FrameEvent):
        """收到完整帧数据（EMG + IMU + 电池）"""
        pass

    def on_gesture(self, gesture_name: str, confidence: float):
        """手势识别回调（预留扩展）"""
        pass


# ==================== 帧解析器 ====================

class FrameParser:
    """二进制帧解析器"""

    @staticmethod
    def parse(frame_bytes: bytes) -> Optional[Dict]:
        """
        解析完整帧数据
        帧格式: [AA AA][LEN][Payload][55]
        LEN 包含帧尾, 总长 = 2 + 1 + LEN
        """
        if len(frame_bytes) < MIN_FRAME_LEN:
            return None

        if frame_bytes[:2] != FRAME_HEADER or frame_bytes[-1] != FRAME_TAIL:
            return None

        length_byte = frame_bytes[2]
        expected_total = 2 + 1 + length_byte

        if len(frame_bytes) != expected_total:
            return None

        try:
            payload = frame_bytes[3:3 + (length_byte - 1)]
            offset = 0

            # 时间戳 4B (big-endian unsigned)
            timestamp, = struct.unpack_from('>I', payload, offset)
            offset += 4

            # 加速度 3B (signed bytes)
            acc_x, acc_y, acc_z = struct.unpack_from('>3b', payload, offset)
            offset += 3

            # 陀螺仪 3B (signed bytes)
            gyro_x, gyro_y, gyro_z = struct.unpack_from('>3b', payload, offset)
            offset += 3

            # 角度 3B (signed bytes)
            pitch, roll, yaw = struct.unpack_from('>3b', payload, offset)
            offset += 3

            # EMG 10 × 8B (unsigned bytes)
            emg_data = []
            for _ in range(NUM_EMG_PACKS):
                channels = list(struct.unpack_from('>8B', payload, offset))
                emg_data.append(channels)
                offset += NUM_EMG_CHANNELS

            # 电池 1B
            battery = payload[offset] if offset < len(payload) else 0

            return {
                'timestamp': timestamp,
                'acc': (acc_x, acc_y, acc_z),
                'gyro': (gyro_x, gyro_y, gyro_z),
                'angle': (pitch, roll, yaw),
                'emg': emg_data,
                'battery': battery,
            }
        except Exception:
            return None

    @staticmethod
    def find_frame(buffer: bytearray):
        """
        在缓冲区中查找有效帧
        返回 (起始索引, 帧长度) 或 (None, None)
        """
        if len(buffer) < MIN_FRAME_LEN:
            return None, None

        idx = 0
        while idx < len(buffer) - 2:
            if buffer[idx:idx + 2] == FRAME_HEADER:
                if idx + 2 < len(buffer):
                    length_byte = buffer[idx + 2]
                    frame_len = 2 + 1 + length_byte

                    if idx + frame_len <= len(buffer):
                        if buffer[idx + frame_len - 1] == FRAME_TAIL:
                            return idx, frame_len
                        else:
                            idx += 1
                    else:
                        return None, None
            else:
                idx += 1

        return None, None


# ==================== 设备类 ====================

class Device:
    """
    EMG 臂带设备（类似 myo.Device）
    封装串口通信和帧解析
    """

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE,
                 timeout=DEFAULT_TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        self.firmware_version = "1.0"  # 预留

        # 内部状态
        self._buffer = bytearray()
        self._max_buffer = 10000

        # 统计
        self.stats = {
            'frames_received': 0,
            'frames_parsed': 0,
            'frames_failed': 0,
            'sync_errors': 0,
            'bytes_received': 0,
            'connect_time': None,
        }

        # 最新数据缓存
        self._latest_emg = None
        self._latest_imu = None
        self._latest_battery = 0

        # 数据历史（环形缓冲区）
        self._emg_history = deque(maxlen=1000)
        self._imu_history = deque(maxlen=1000)

    def connect(self) -> bool:
        """连接设备"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate,
                                        timeout=self.timeout)
            self.connected = True
            self.stats['connect_time'] = time.time()
            self._buffer.clear()
            return True
        except serial.SerialException as e:
            self.connected = False
            raise ConnectionError(f"无法连接 {self.port}: {e}")

    def disconnect(self):
        """断开连接"""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected and self.serial and self.serial.is_open

    def read_frames(self) -> List[Dict]:
        """
        读取并解析所有可用帧
        返回解析后的帧数据列表
        """
        if not self.is_connected():
            return []

        try:
            data = self.serial.read(256)
        except serial.SerialException:
            self.connected = False
            return []

        if not data:
            return []

        self._buffer.extend(data)
        self.stats['bytes_received'] += len(data)

        # 防止缓冲区溢出
        if len(self._buffer) > self._max_buffer:
            self._buffer.clear()
            return []

        frames = []
        while True:
            start, length = FrameParser.find_frame(self._buffer)

            if start is None:
                if len(self._buffer) > 500:
                    self._buffer = self._buffer[100:]
                    self.stats['sync_errors'] += 100
                break

            if start > 0:
                self.stats['sync_errors'] += start
                self._buffer = self._buffer[start:]

            frame_bytes = bytes(self._buffer[:length])
            parsed = FrameParser.parse(frame_bytes)

            if parsed:
                self.stats['frames_parsed'] += 1
                frames.append(parsed)
            else:
                self.stats['frames_failed'] += 1

            self._buffer = self._buffer[length:]

        self.stats['frames_received'] += len(frames)
        return frames

    @property
    def latest_emg(self) -> Optional[EmgEvent]:
        return self._latest_emg

    @property
    def latest_imu(self) -> Optional[ImuEvent]:
        return self._latest_imu

    @property
    def battery_level(self) -> int:
        return self._latest_battery

    @property
    def emg_history(self) -> deque:
        return self._emg_history

    @property
    def imu_history(self) -> deque:
        return self._imu_history

    def get_fps(self) -> float:
        """获取当前帧率"""
        if self.stats['connect_time'] is None:
            return 0.0
        elapsed = time.time() - self.stats['connect_time']
        if elapsed <= 0:
            return 0.0
        return self.stats['frames_parsed'] / elapsed

    def get_stats_str(self) -> str:
        """获取统计信息字符串"""
        elapsed = 0
        if self.stats['connect_time']:
            elapsed = time.time() - self.stats['connect_time']
        fps = self.stats['frames_parsed'] / elapsed if elapsed > 0 else 0
        return (f"运行: {elapsed:.1f}s | 帧: {self.stats['frames_parsed']} | "
                f"失败: {self.stats['frames_failed']} | "
                f"同步错误: {self.stats['sync_errors']} | {fps:.1f} fps")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self):
        status = "已连接" if self.is_connected() else "未连接"
        return f"Device(port={self.port}, status={status})"


# ==================== Hub 管理器 ====================

class Hub:
    """
    设备管理中心（类似 myo.Hub）
    管理设备连接、数据轮询、事件分发
    """

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE,
                 timeout=DEFAULT_TIMEOUT, reconnect_delay=2):
        self.device = Device(port, baudrate, timeout)
        self.reconnect_delay = reconnect_delay
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._listeners: List[DeviceListener] = []
        self._lock = threading.Lock()

        # 回调函数（函数式接口，作为 DeviceListener 的替代）
        self._on_emg: Optional[Callable] = None
        self._on_imu: Optional[Callable] = None
        self._on_battery: Optional[Callable] = None
        self._on_frame: Optional[Callable] = None

    def add_listener(self, listener: DeviceListener):
        """添加事件监听器"""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: DeviceListener):
        """移除事件监听器"""
        with self._lock:
            self._listeners.remove(listener)

    # --- 函数式回调注册（装饰器风格）---

    def on_emg(self, func):
        """注册 EMG 回调（装饰器）"""
        self._on_emg = func
        return func

    def on_imu(self, func):
        """注册 IMU 回调（装饰器）"""
        self._on_imu = func
        return func

    def on_battery(self, func):
        """注册电池回调（装饰器）"""
        self._on_battery = func
        return func

    def on_frame(self, func):
        """注册完整帧回调（装饰器）"""
        self._on_frame = func
        return func

    def _dispatch_events(self, parsed: Dict):
        """将解析数据分发为事件"""
        ts = parsed['timestamp']
        acc_t = parsed['acc']
        gyro_t = parsed['gyro']
        angle_t = parsed['angle']

        # 构建事件对象
        emg_event = EmgEvent(ts, parsed['emg'], self.device)
        imu_event = ImuEvent(
            ts,
            Vector3(*acc_t),
            Vector3(*gyro_t),
            Orientation(*angle_t),
            self.device
        )
        battery_event = BatteryEvent(ts, parsed['battery'], self.device)
        frame_event = FrameEvent(ts, emg_event, imu_event, battery_event)

        # 更新设备缓存
        self.device._latest_emg = emg_event
        self.device._latest_imu = imu_event
        self.device._latest_battery = parsed['battery']
        self.device._emg_history.append(emg_event)
        self.device._imu_history.append(imu_event)

        # 分发给所有监听器
        with self._lock:
            for listener in self._listeners:
                try:
                    listener.on_emg_data(emg_event)
                    listener.on_imu_data(imu_event)
                    listener.on_battery(battery_event)
                    listener.on_frame(frame_event)
                except Exception as e:
                    print(f"[WARNING] 监听器回调异常: {e}")

        # 分发给函数式回调
        if self._on_emg:
            self._on_emg(emg_event)
        if self._on_imu:
            self._on_imu(imu_event)
        if self._on_battery:
            self._on_battery(battery_event)
        if self._on_frame:
            self._on_frame(frame_event)

    def _run_loop(self):
        """内部运行循环"""
        while self._running:
            if not self.device.is_connected():
                try:
                    self.device.connect()
                    print(f"✓ 已连接: {self.device.port}")
                    with self._lock:
                        for listener in self._listeners:
                            listener.on_connected(self.device)
                except ConnectionError as e:
                    print(f"✗ {e}，{self.reconnect_delay}s后重试...")
                    time.sleep(self.reconnect_delay)
                    continue

            frames = self.device.read_frames()
            for parsed in frames:
                self._dispatch_events(parsed)

    def run(self, listener: Optional[DeviceListener] = None, duration=0):
        """
        阻塞运行（主线程）
        
        Args:
            listener: 事件监听器（可选，也可通过 add_listener 预先添加）
            duration: 运行时长（秒），0 = 无限运行直到 Ctrl+C
        """
        if listener:
            self.add_listener(listener)

        self._running = True
        start = time.time()

        print(f"EMG Armband Hub 启动 | {self.device.port} @ {self.device.baudrate}")
        print("按 Ctrl+C 停止\n")

        try:
            while self._running:
                if duration > 0 and (time.time() - start) >= duration:
                    break
                self._run_loop_once()
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.stop()

    def run_forever(self, listener: Optional[DeviceListener] = None):
        """无限运行"""
        self.run(listener, duration=0)

    def run_background(self, listener: Optional[DeviceListener] = None):
        """
        后台线程运行（非阻塞）
        返回线程对象
        """
        if listener:
            self.add_listener(listener)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self._thread

    def _run_loop_once(self):
        """单次循环"""
        if not self.device.is_connected():
            try:
                self.device.connect()
                print(f"✓ 已连接: {self.device.port}")
                with self._lock:
                    for listener in self._listeners:
                        listener.on_connected(self.device)
            except ConnectionError:
                time.sleep(self.reconnect_delay)
                return

        frames = self.device.read_frames()
        for parsed in frames:
            self._dispatch_events(parsed)

    def stop(self):
        """停止运行"""
        self._running = False
        if self.device.is_connected():
            with self._lock:
                for listener in self._listeners:
                    listener.on_disconnected(self.device)
            self.device.disconnect()
        print(f"Hub 已停止 | {self.device.get_stats_str()}")

    @property
    def running(self):
        return self._running


# ==================== 内置监听器 ====================

class PrintListener(DeviceListener):
    """打印监听器 - 将数据输出到控制台（调试用）"""

    def __init__(self, print_emg=True, print_imu=True, print_interval=0.5):
        self._print_emg = print_emg
        self._print_imu = print_imu
        self._interval = print_interval
        self._last_print = 0

    def on_frame(self, event: FrameEvent):
        now = time.time()
        if now - self._last_print < self._interval:
            return
        self._last_print = now

        imu = event.imu_event
        bat = event.battery_event
        ts = event.timestamp
        time_str = time.strftime('%H:%M:%S')

        print(f"[{time_str}] TS:{ts:10d} | "
              f"Acc({imu.acceleration.x:4d},{imu.acceleration.y:4d},{imu.acceleration.z:4d}) | "
              f"Gyro({imu.gyroscope.x:4d},{imu.gyroscope.y:4d},{imu.gyroscope.z:4d}) | "
              f"Angle(P:{imu.orientation.pitch:4d},R:{imu.orientation.roll:4d},"
              f"Y:{imu.orientation.yaw:4d}) | Bat:{bat.level:3d}%")

        if self._print_emg:
            for i, pack in enumerate(event.emg_event.emg, 1):
                hex_str = ' '.join(f'{ch:02X}' for ch in pack)
                print(f"  EMG{i:2d}: {hex_str}  ({pack})")
            print()


class CsvRecorder(DeviceListener):
    """
    CSV 记录器 - 将数据保存到 CSV 文件
    每帧展开为 10 行（每个 EMG 包一行）
    """

    CSV_HEADERS = [
        'timestamp',
        'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8',
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'angle_pitch', 'angle_roll', 'angle_yaw',
        'battery', 'emg_pack_index',
    ]

    def __init__(self, filename='emg_data.csv', include_timestamp=True):
        self.filename = filename
        self.include_timestamp = include_timestamp
        self._file = None
        self._writer = None
        self._rows_written = 0

    def on_connected(self, device):
        self._file = open(self.filename, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.CSV_HEADERS)
        self._rows_written = 0
        print(f"✓ CSV 记录器启动: {self.filename}")

    def on_frame(self, event: FrameEvent):
        if self._writer is None:
            return

        ts = event.timestamp
        imu = event.imu_event
        bat = event.battery_event.level
        acc = imu.acceleration
        gyro = imu.gyroscope
        angle = imu.orientation

        for pack_idx, emg_pack in enumerate(event.emg_event.emg):
            row = [
                ts,
                *emg_pack,
                acc.x, acc.y, acc.z,
                gyro.x, gyro.y, gyro.z,
                angle.pitch, angle.roll, angle.yaw,
                bat, pack_idx + 1,
            ]
            self._writer.writerow(row)
            self._rows_written += 1

        # 定期刷新
        if self._rows_written % 500 == 0:
            self._file.flush()

    def on_disconnected(self, device):
        self.close()

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
            size = os.path.getsize(self.filename) if os.path.exists(self.filename) else 0
            print(f"✓ CSV 已保存: {self.filename} ({self._rows_written} 行, {size/1024:.1f}KB)")

    @property
    def rows_written(self):
        return self._rows_written


class DataBuffer(DeviceListener):
    """
    数据缓冲器 - 在内存中缓存最近 N 帧数据
    适合实时分析和可视化
    """

    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.emg_buffer = deque(maxlen=maxlen)       # EmgEvent
        self.imu_buffer = deque(maxlen=maxlen)        # ImuEvent
        self.battery_buffer = deque(maxlen=maxlen)    # int
        self._frame_count = 0

    def on_emg_data(self, event: EmgEvent):
        self.emg_buffer.append(event)

    def on_imu_data(self, event: ImuEvent):
        self.imu_buffer.append(event)

    def on_battery(self, event: BatteryEvent):
        self.battery_buffer.append(event.level)

    def on_frame(self, event: FrameEvent):
        self._frame_count += 1

    def get_emg_array(self, last_n=None) -> np.ndarray:
        """获取最近 N 帧 EMG 数据，shape=(N, 10, 8)"""
        data = list(self.emg_buffer)
        if last_n:
            data = data[-last_n:]
        if not data:
            return np.empty((0, NUM_EMG_PACKS, NUM_EMG_CHANNELS), dtype=np.uint8)
        return np.array([e.emg for e in data], dtype=np.uint8)

    def get_imu_arrays(self, last_n=None):
        """获取最近 N 帧 IMU 数据，返回 (acc, gyro, angle) 各 shape=(N, 3)"""
        data = list(self.imu_buffer)
        if last_n:
            data = data[-last_n:]
        if not data:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty, empty

        acc = np.array([[e.acceleration.x, e.acceleration.y, e.acceleration.z]
                        for e in data], dtype=np.float32)
        gyro = np.array([[e.gyroscope.x, e.gyroscope.y, e.gyroscope.z]
                         for e in data], dtype=np.float32)
        angle = np.array([[e.orientation.pitch, e.orientation.roll, e.orientation.yaw]
                          for e in data], dtype=np.float32)
        return acc, gyro, angle

    @property
    def frame_count(self):
        return self._frame_count

    def clear(self):
        self.emg_buffer.clear()
        self.imu_buffer.clear()
        self.battery_buffer.clear()
        self._frame_count = 0


# ==================== 便捷函数 ====================

def init(port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE):
    """初始化并返回 Hub（类似 myo.init()）"""
    return Hub(port=port, baudrate=baudrate)


def list_ports():
    """列出可用串口"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    result = []
    for p in ports:
        result.append({
            'device': p.device,
            'description': p.description,
            'hwid': p.hwid,
        })
        print(f"  {p.device}: {p.description} [{p.hwid}]")
    return result


# ==================== 模块入口（测试）====================

if __name__ == '__main__':
    print("=" * 60)
    print(f"EMG Armband SDK v{__version__}")
    print("=" * 60)
    print("\n可用串口:")
    list_ports()

    print("\n使用示例:")
    print("  import emg_armband as emg")
    print("  hub = emg.Hub(port='COM4')")
    print("  hub.run(emg.PrintListener())")
