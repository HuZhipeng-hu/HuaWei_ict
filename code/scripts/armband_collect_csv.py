# 文件名: emg_to_csv.py
# 功能: 从EMG臂带串口接收数据并保存为CSV文件
# 依赖: pip install pyserial numpy
# 用法: python emg_to_csv.py

import serial
import time
import struct
import csv
import os
import sys

# ==================== 配置 ====================
PORT = 'COM4'
BAUDRATE = 115200
TIMEOUT = 0.5
RECONNECT_DELAY = 2
MAX_BUFFER_SIZE = 10000

# CSV 配置
CSV_FILENAME = 'emg_data.csv'
CSV_HEADERS = [
    'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8',
    'acc_x', 'acc_y', 'acc_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'angle_pitch', 'angle_roll', 'angle_yaw'
]

# 采集配置
RECORD_SECONDS = 30  # 默认采集时长（秒），0表示无限采集直到Ctrl+C

# 帧格式常数
FRAME_HEADER = b'\xAA\xAA'
FRAME_TAIL = 0x55
MIN_FRAME_LEN = 6


# ==================== 帧解析 ====================
def parse_frame(frame_bytes):
    """
    解析完整的帧数据
    帧格式：[AA AA][长度1B][时间戳4B][Acc3B][Gyro3B][Angle3B]
            [EMG1-10, 每个8B][电池1B][帧尾: 55]
    长度字节包含帧尾，总帧长 = 2(头) + 1(长度字节) + length_byte
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

        # 时间戳 4B
        timestamp, = struct.unpack_from('>I', payload, offset)
        offset += 4

        # 加速度 3B (unsigned, 0-255)
        acc_x, acc_y, acc_z = struct.unpack_from('>3B', payload, offset)
        offset += 3

        # 陀螺仪 3B (unsigned, 0-255)
        gyro_x, gyro_y, gyro_z = struct.unpack_from('>3B', payload, offset)
        offset += 3

        # 角度 3B (unsigned, 0-255)
        pitch, roll, yaw = struct.unpack_from('>3B', payload, offset)
        offset += 3

        # EMG 10个包，每个8字节（8个通道，uint8）
        emg_data = []
        for _ in range(10):
            emg_channels = struct.unpack_from('>8B', payload, offset)
            emg_data.append(list(emg_channels))
            offset += 8

        # 电池
        battery = payload[offset] if offset < len(payload) else 0

        return {
            'timestamp': timestamp,
            'acc': {'x': acc_x, 'y': acc_y, 'z': acc_z},
            'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},
            'angle': {'pitch': pitch, 'roll': roll, 'yaw': yaw},
            'emg': emg_data,
            'battery': battery,
        }
    except Exception:
        return None


def find_frame_boundary(buffer):
    """
    在缓冲区中查找有效的帧边界
    返回 (帧起始索引, 帧长度) 或 (None, None)
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
                    tail_pos = idx + frame_len - 1
                    if buffer[tail_pos] == FRAME_TAIL:
                        return idx, frame_len
                    else:
                        idx += 1
                else:
                    return None, None
        else:
            idx += 1

    return None, None


def frame_to_csv_rows(parsed):
    """
    将解析后的帧数据转换为多行CSV数据
    每帧包含10个EMG包(EMG1-EMG10)，每个包8个通道
    每个EMG包展开为一行：emg1~emg8 对应该包的8个通道
    acc/gyro/angle 在同一帧的10行中相同
    返回10行数据的列表
    """
    acc = parsed['acc']
    gyro = parsed['gyro']
    angle = parsed['angle']

    rows = []
    for emg_pack in parsed['emg']:  # 遍历EMG1~EMG10共10个包
        row = [
            emg_pack[0], emg_pack[1], emg_pack[2], emg_pack[3],
            emg_pack[4], emg_pack[5], emg_pack[6], emg_pack[7],
            acc['x'], acc['y'], acc['z'],
            gyro['x'], gyro['y'], gyro['z'],
            angle['pitch'], angle['roll'], angle['yaw'],
        ]
        rows.append(row)
    return rows


# ==================== 主程序 ====================
def main():
    ser = None
    buffer = bytearray()
    frames_saved = 0
    frames_failed = 0
    sync_errors = 0
    start_time = None

    # 解析命令行参数
    csv_file = CSV_FILENAME
    record_sec = RECORD_SECONDS
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            record_sec = int(sys.argv[2])
        except ValueError:
            pass

    print("=" * 60)
    print("EMG 臂带数据采集 → CSV")
    print("=" * 60)
    print(f"串口: {PORT} @ {BAUDRATE} bps")
    print(f"输出文件: {csv_file}")
    print(f"采集时长: {record_sec}s" if record_sec > 0 else "采集时长: 无限 (Ctrl+C停止)")
    print(f"CSV列: {', '.join(CSV_HEADERS)}")
    print("按 Ctrl+C 可随时停止采集\n")

    # 打开CSV文件
    csv_fp = open(csv_file, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_fp)
    writer.writerow(CSV_HEADERS)

    try:
        # 连接串口
        while ser is None:
            try:
                ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
                print(f"✓ 串口已连接: {PORT}")
            except serial.SerialException as e:
                print(f"✗ 无法打开串口 {PORT}: {e}")
                print(f"  {RECONNECT_DELAY}秒后重试...")
                time.sleep(RECONNECT_DELAY)

        start_time = time.time()
        last_stats = start_time
        last_print = start_time

        print(f"✓ 开始采集数据...\n")

        while True:
            # 检查是否超时
            elapsed = time.time() - start_time
            if record_sec > 0 and elapsed >= record_sec:
                print(f"\n✓ 已达到设定采集时长 {record_sec}s，停止采集")
                break

            # 读取串口数据
            try:
                data = ser.read(256)
            except serial.SerialException as e:
                print(f"✗ 串口读取错误: {e}")
                break

            if data:
                buffer.extend(data)

                # 防止缓冲区溢出
                if len(buffer) > MAX_BUFFER_SIZE:
                    buffer.clear()
                    continue

                # 解析帧
                while True:
                    frame_start, frame_len = find_frame_boundary(buffer)

                    if frame_start is None:
                        if len(buffer) > 500:
                            buffer = buffer[100:]
                            sync_errors += 100
                        break

                    if frame_start > 0:
                        sync_errors += frame_start
                        buffer = buffer[frame_start:]

                    frame = bytes(buffer[:frame_len])
                    parsed = parse_frame(frame)

                    if parsed:
                        rows = frame_to_csv_rows(parsed)
                        for row in rows:
                            writer.writerow(row)
                        frames_saved += 1

                        # 每秒打印进度
                        now = time.time()
                        if now - last_print >= 1.0:
                            remaining = ""
                            if record_sec > 0:
                                left = max(0, record_sec - (now - start_time))
                                remaining = f" | 剩余: {left:.0f}s"
                            print(f"  [采集中] 已保存: {frames_saved} 帧 | "
                                  f"时间: {now - start_time:.1f}s{remaining}")
                            last_print = now
                    else:
                        frames_failed += 1

                    buffer = buffer[frame_len:]

            # 定期刷新文件
            if time.time() - last_stats > 5:
                csv_fp.flush()
                last_stats = time.time()

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断采集")

    finally:
        # 关闭资源
        csv_fp.flush()
        csv_fp.close()

        if ser and ser.is_open:
            ser.close()

        elapsed = time.time() - start_time if start_time else 0
        fps = frames_saved / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("采集完成 - 统计信息")
        print("=" * 60)
        print(f"  运行时间:   {elapsed:.1f}s")
        print(f"  保存帧数:   {frames_saved}")
        print(f"  失败帧数:   {frames_failed}")
        print(f"  同步错误:   {sync_errors}")
        print(f"  平均帧率:   {fps:.1f} fps")
        print(f"  输出文件:   {os.path.abspath(csv_file)}")
        file_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
        print(f"  文件大小:   {file_size / 1024:.1f} KB")
        print("=" * 60)


if __name__ == '__main__':
    main()
