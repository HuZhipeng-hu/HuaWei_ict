# 文件名: emg_armband_serial.py
# 依赖: pip install pyserial numpy
# 用法: python emg_armband_serial.py

import serial
import time
import struct
import numpy as np
from collections import deque
import sys

# ==================== 配置 ====================
PORT = 'COM4'              # USB-SERIAL CH340 (COM3)
BAUDRATE = 115200          
TIMEOUT = 0.5              # 增加超时避免忙轮询
RECONNECT_DELAY = 2        # 重连延迟（秒）
MAX_BUFFER_SIZE = 10000    # 最大缓冲大小，防止内存溢出

# 帧格式常数
FRAME_HEADER = b'\xAA\xAA'
FRAME_TAIL = 0x55
MIN_FRAME_LEN = 6          # 最小帧长：AA AA + len + tail + 至少1字节payload

# 统计信息
class Stats:
    def __init__(self):
        self.frames_received = 0
        self.frames_parsed = 0
        self.frames_failed = 0
        self.sync_errors = 0
        self.start_time = time.time()
    
    def report(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rate = self.frames_parsed / elapsed
            print(f"\n[统计] 运行时间: {elapsed:.1f}s | "
                  f"接收: {self.frames_received} | 成功: {self.frames_parsed} | "
                  f"失败: {self.frames_failed} | 同步错误: {self.sync_errors} | "
                  f"率: {rate:.1f} fps")

stats = Stats()


def parse_frame(frame_bytes):
    """
    解析完整的帧数据（根据用户提供的格式）
    帧格式：
    [帧头2B: AA AA][长度1B][时间戳4B][Acc3B][Gyro3B][Angle3B]
    [EMG1-10, 每个8B（8个通道）][电池1B][帧尾1B: 55]
    
    长度字节 = 包含帧尾的负载长度
    总帧长 = 2(头) + 1(长度字节) + length_byte
    有效载荷 = length_byte - 1 (去掉帧尾)
    返回解析结果字典或 None
    """
    if len(frame_bytes) < MIN_FRAME_LEN:
        return None
    
    # 检查帧头和帧尾
    if frame_bytes[:2] != FRAME_HEADER or frame_bytes[-1] != FRAME_TAIL:
        return None

    length_byte = frame_bytes[2]
    # 帧总长度 = 头(2) + 长度字节(1) + length_byte内容(包含帧尾)
    expected_total = 2 + 1 + length_byte
    
    if len(frame_bytes) != expected_total:
        return None

    try:
        # 有效载荷 = 从位置3开始，长度为 length_byte - 1（去掉帧尾0x55）
        payload = frame_bytes[3:3+(length_byte-1)]
        offset = 0

        # 时间戳 4B 大端 unsigned int
        timestamp, = struct.unpack_from('>I', payload, offset)
        offset += 4

        # 加速度 AccX, AccY, AccZ (3个unsigned byte, 0-255)
        acc_x, acc_y, acc_z = struct.unpack_from('>3B', payload, offset)
        offset += 3

        # 陀螺仪 GyroX, GyroY, GyroZ (3个unsigned byte, 0-255)
        gyro_x, gyro_y, gyro_z = struct.unpack_from('>3B', payload, offset)
        offset += 3

        # 角度 Pitch, Roll, Yaw (3个unsigned byte, 0-255)
        pitch, roll, yaw = struct.unpack_from('>3B', payload, offset)
        offset += 3

        # EMG 10个数据包，每个8字节（对应8个通道）
        emg_data = []  # 列表，包含10个EMG数据包，每个包含8个通道
        for emg_idx in range(10):
            # 每个EMG包包含8个字节，每个字节是一个通道的数据
            emg_channels = struct.unpack_from('>8B', payload, offset)
            emg_data.append(list(emg_channels))
            offset += 8

        # 电池电量百分比 (1B，0-100)
        battery = payload[offset] if offset < len(payload) else 0
        
        return {
            'timestamp': timestamp,
            'acc': {'x': acc_x, 'y': acc_y, 'z': acc_z},
            'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},
            'angle': {'pitch': pitch, 'roll': roll, 'yaw': yaw},
            'emg': emg_data,  # 10个EMG包，每个包含8个通道
            'battery': battery,
            'length': length_byte
        }
    except Exception as e:
        return None


def find_frame_boundary(buffer, debug=False):
    """
    在缓冲区中查找有效的帧边界
    帧格式：[AA AA][长度1B][有效载荷][55]
    长度字节包含帧尾，总帧长 = 2(头) + 1(长度字节) + length_byte
    返回 (帧起始索引, 帧长度) 或 (None, None)
    """
    if len(buffer) < MIN_FRAME_LEN:
        return None, None
    
    # 查找帧头
    idx = 0
    while idx < len(buffer) - 2:
        if buffer[idx:idx+2] == FRAME_HEADER:
            if idx + 2 < len(buffer):
                length_byte = buffer[idx + 2]
                
                # 帧总长度 = 头(2) + 长度字节(1) + length_byte(包含帧尾)
                frame_len = 2 + 1 + length_byte
                
                # 检查是否有足够的数据
                if idx + frame_len <= len(buffer):
                    # 验证帧尾
                    tail_pos = idx + frame_len - 1
                    if buffer[tail_pos] == FRAME_TAIL:
                        if debug:
                            frame = buffer[idx:idx+frame_len]
                            print(f"[DEBUG] 找到帧 @ idx={idx}, 长度字节={length_byte}, 帧长={frame_len}")
                        return idx, frame_len
                    else:
                        # 帧尾错误，继续查找下一个可能的帧头
                        idx += 1
                else:
                    # 数据不足，需要更多数据
                    if debug and idx == 0:
                        print(f"[DEBUG] 缓冲区不足: 有 {len(buffer)} 字节，需要 {frame_len} 字节")
                    return None, None
        else:
            idx += 1
    
    return None, None


def open_serial_port(port, baudrate, timeout=TIMEOUT):
    """
    打开串口并返回连接对象，带异常处理
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"✓ 串口已连接: {port} @ {baudrate} bps")
        return ser
    except serial.SerialException as e:
        print(f"✗ 无法打开串口 {port}: {e}")
        return None
    except Exception as e:
        print(f"✗ 未知错误: {e}")
        return None




def analyze_raw_data(data, max_lines=10):
    """
    分析原始数据，寻找规律
    """
    print("\n" + "="*70)
    print("数据分析：")
    print("="*70)
    
    # 查找所有可能的帧头
    print(f"\n总长度: {len(data)} 字节")
    print(f"\n十六进制（前500字节）:")
    for i in range(0, min(500, len(data)), 16):
        hex_str = data[i:i+16].hex()
        hex_str = ' '.join([hex_str[j:j+2] for j in range(0, len(hex_str), 2)])
        ascii_str = ''.join([chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16]])
        print(f"  {i:04d}: {hex_str:<48} {ascii_str}")
    
    # 查找重复的字节序列
    print(f"\n查找可能的帧头候选:")
    candidates = {}
    for i in range(len(data) - 1):
        pair = data[i:i+2]
        candidates[pair] = candidates.get(pair, 0) + 1
    
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_cands[:10]:
        print(f"  {pair.hex()}: {count:4d} 次")
    
    # 查找可能的帧尾候选
    print(f"\n单字节重复高频:")
    single_freq = {}
    for b in data:
        single_freq[b] = single_freq.get(b, 0) + 1
    sorted_single = sorted(single_freq.items(), key=lambda x: x[1], reverse=True)
    for byte_val, count in sorted_single[:10]:
        print(f"  0x{byte_val:02x}: {count:5d} 次 ({chr(byte_val) if 32 <= byte_val < 127 else '?'})")


def print_frame_data(parsed):
    """
    格式化输出一行数据（十六进制格式）
    EMG1-EMG10：每个包含8个通道的数据
    """
    timestamp = parsed['timestamp']
    acc = parsed['acc']
    gyro = parsed['gyro']
    angle = parsed['angle']
    emg = parsed['emg']  # 10个EMG包，每个包含8个通道
    battery = parsed['battery']
    
    time_str = time.strftime('%H:%M:%S')
    
    # 加速度数据：AccX, AccY, AccZ
    acc_str = f"Acc({acc['x']:4d},{acc['y']:4d},{acc['z']:4d})"
    
    # 陀螺仪数据：GyroX, GyroY, GyroZ
    gyro_str = f"Gyro({gyro['x']:4d},{gyro['y']:4d},{gyro['z']:4d})"
    
    # 角度数据：Pitch, Roll, Yaw
    angle_str = f"Angle(P:{angle['pitch']:4d},R:{angle['roll']:4d},Y:{angle['yaw']:4d})"
    
    # EMG 数据输出
    print(f"[{time_str}] TS:{timestamp:10d} | {acc_str} | {gyro_str} | {angle_str} | Bat:{battery:3d}%")
    
    # 输出10个EMG包的数据（十六进制格式）
    for emg_idx, emg_channels in enumerate(emg, 1):
        hex_data = ' '.join(f'{ch:02X}' for ch in emg_channels)
        print(f"  EMG{emg_idx}: {hex_data}  (十进制: {emg_channels})")
    
    print()


def read_serial_data(ser):
    """
    从串口读取数据并以十六进制格式输出（原始数据查看）
    """
    print("\n=== 原始数据读取模式（十六进制显示）===\n")
    try:
        frame_count = 0
        while True:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)  # 读取串口数据
                if data:
                    frame_count += 1
                    hex_data = ' '.join(f'{byte:02X}' for byte in data)
                    print(f"[帧{frame_count}] {hex_data}")
    except KeyboardInterrupt:
        print("\n终止数据读取。")
    except Exception as e:
        print(f"读取数据时出错: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("串口已关闭。")


def main():
    """
    主程序：连接串口并持续接收解析数据
    """
    ser = None
    buffer = bytearray()
    last_print = time.time()
    last_stats = time.time()

    print("=" * 80)
    print("EMG 臂带串口数据接收器 v2.0")
    print("=" * 80)
    print(f"目标端口: {PORT} | 波特率: {BAUDRATE} bps | 超时: {TIMEOUT}s")
    print("按 Ctrl+C 退出...\n")

    while True:
        # 重连机制
        if ser is None:
            ser = open_serial_port(PORT, BAUDRATE, TIMEOUT)
            if ser is None:
                time.sleep(RECONNECT_DELAY)
                continue

        try:
            # 读取数据
            data = ser.read(256)  # 一次读取256字节
            #print(data)
            if data:
                buffer.extend(data)
                stats.frames_received += len(data)
                
                # 防止缓冲区溢出
                if len(buffer) > MAX_BUFFER_SIZE:
                    print(f"⚠ 缓冲区溢出，清空缓冲区")
                    buffer.clear()
                
                # 处理帧
                while True:
                    frame_start, frame_len = find_frame_boundary(buffer, debug=False)
                    
                    if frame_start is None:
                        # 没有找到有效帧
                        if len(buffer) > 500:
                            # 缓冲区太大但找不到有效帧，丢弃前100字节
                            buffer = buffer[100:]
                            stats.sync_errors += 100
                        break
                    else:
                        # 找到有效帧
                        if frame_start > 0:
                            # 丢弃帧之前的无效数据
                            stats.sync_errors += frame_start
                            buffer = buffer[frame_start:]
                        
                        frame = bytes(buffer[:frame_len])
                        parsed = parse_frame(frame)
                        
                        if parsed:
                            stats.frames_parsed += 1
                            
                            # 定期输出数据（每0.5秒）
                            now = time.time()
                            if now - last_print > 0.5:
                                print_frame_data(parsed)
                                last_print = now
                        else:
                            stats.frames_failed += 1
                        
                        # 移除已处理的帧
                        buffer = buffer[frame_len:]
                
                # 定期输出统计（每10秒）
                if time.time() - last_stats > 10:
                    stats.report()
                    last_stats = time.time()

        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("用户中断，正在关闭...")
            print("=" * 80)
            stats.report()
            break
        
        except serial.SerialException as e:
            print(f"✗ 串口错误: {e}")
            if ser:
                ser.close()
            ser = None
            time.sleep(RECONNECT_DELAY)
        
        except Exception as e:
            print(f"✗ 运行错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    # 清理资源
    if ser and ser.is_open:
        ser.close()
        print("✓ 串口已关闭\n")


if __name__ == '__main__':
    main()