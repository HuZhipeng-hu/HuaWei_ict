"""
EMG armband serial sensor driver.

This implementation supports two frame protocols observed in the project:

1) Legacy fixed-length frame:
   header: 0xAA 0x55
   tail:   0x55 0xAA
   total length: 25 bytes
   EMG bytes: 8 channels at frame[2:10]

2) New variable-length frame (used by scripts/emg_armband.py):
   header: 0xAA 0xAA
   length: 1 byte (includes tail byte)
   payload (without tail): timestamp(4) + acc(3) + gyro(3) + angle(3) + emg(10x8) + battery(1)
   tail: 0x55

For the new protocol, each frame contains 10 EMG packs. We ingest all packs and
then apply downsampling to target rate.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from .base import SensorBase

logger = logging.getLogger(__name__)

try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Shared EMG shape
NUM_EMG_CHANNELS = 8

# Legacy protocol (fixed frame)
LEGACY_HEADER = bytes([0xAA, 0x55])
LEGACY_TAIL = bytes([0x55, 0xAA])
LEGACY_FRAME_LENGTH = 25

# New protocol (length-delimited)
NEW_HEADER = bytes([0xAA, 0xAA])
NEW_TAIL = 0x55
NEW_META_BYTES = 4 + 3 + 3 + 3  # ts + acc + gyro + angle
NEW_PACK_COUNT = 10
NEW_BATTERY_BYTES = 1
NEW_EXPECTED_PAYLOAD_NO_TAIL = NEW_META_BYTES + (NEW_PACK_COUNT * NUM_EMG_CHANNELS) + NEW_BATTERY_BYTES


class ArmbandSensor(SensorBase):
    """
    Serial EMG armband sensor with background receive thread.
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
        self._decimate_ratio = max(1, int(device_sampling_rate // target_sampling_rate))
        self._center_value = center_value

        self._serial = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        self._sample_counter = 0
        self._total_frames = 0
        self._error_frames = 0

    def connect(self) -> bool:
        if not SERIAL_AVAILABLE:
            logger.error("pyserial not installed. Please run: pip install pyserial")
            return False

        port = self._port or self._auto_detect_port()
        if port is None:
            logger.error("No serial port found for armband.")
            return False

        try:
            self._serial = serial.Serial(port=port, baudrate=self._baudrate, timeout=1)
            self._port = port
            time.sleep(0.5)

            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True, name="ArmbandReader")
            self._thread.start()

            logger.info("Armband connected: %s @ %sbps", port, self._baudrate)
            return True
        except serial.SerialException as exc:
            logger.error("Serial connect failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

        logger.info(
            "Armband disconnected (frames=%s, errors=%s)", self._total_frames, self._error_frames
        )

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def read_window(self, window_size: int) -> Optional[np.ndarray]:
        with self._lock:
            if len(self._buffer) < window_size:
                return None
            data = list(self._buffer)[-window_size:]
            return np.array(data, dtype=np.float32)

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open and self._running

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

    def _read_loop(self) -> None:
        raw_buffer = bytearray()

        while self._running:
            try:
                if not self._serial or not self._serial.is_open:
                    break

                # Some USB-serial drivers may report in_waiting=0 intermittently
                # even while data is streaming. Fall back to a blocking read so
                # we do not miss frames in that case.
                available = self._serial.in_waiting
                read_size = available if available > 0 else 256
                chunk = self._serial.read(read_size)
                if not chunk:
                    continue
                raw_buffer.extend(chunk)

                # Parse as many frames as possible.
                while True:
                    parsed = self._extract_frame(raw_buffer)
                    if parsed is None:
                        break

                    consumed, emg_samples, is_error = parsed
                    if consumed <= 0:
                        break

                    if is_error:
                        self._error_frames += 1
                    elif emg_samples:
                        self._total_frames += 1
                        for emg_raw in emg_samples:
                            self._ingest_emg_sample(emg_raw)

                    raw_buffer = raw_buffer[consumed:]
            except Exception as exc:  # pragma: no cover - runtime side guard
                logger.error("Sensor read-loop error: %s", exc)
                time.sleep(0.1)

    def _extract_frame(self, raw_buffer: bytearray):
        """
        Parse one frame from buffer.

        Returns:
            (consumed_bytes, emg_samples, is_error) or None if buffer needs more bytes.
        """
        if len(raw_buffer) < 3:
            return None

        legacy_pos = raw_buffer.find(LEGACY_HEADER)
        new_pos = raw_buffer.find(NEW_HEADER)
        candidates = [p for p in (legacy_pos, new_pos) if p >= 0]

        if not candidates:
            # Keep last 2 bytes for possible partial header.
            drop = max(0, len(raw_buffer) - 2)
            if drop > 0:
                return drop, [], False
            return None

        start = min(candidates)
        if start > 0:
            return start, [], False

        if raw_buffer.startswith(NEW_HEADER):
            return self._extract_new_frame(raw_buffer)
        if raw_buffer.startswith(LEGACY_HEADER):
            return self._extract_legacy_frame(raw_buffer)

        # Unknown sync state; skip one byte.
        return 1, [], True

    def _extract_legacy_frame(self, raw_buffer: bytearray):
        if len(raw_buffer) < LEGACY_FRAME_LENGTH:
            return None

        frame = raw_buffer[:LEGACY_FRAME_LENGTH]
        if frame[-2:] != LEGACY_TAIL:
            return 1, [], True

        emg_raw = np.frombuffer(frame[2 : 2 + NUM_EMG_CHANNELS], dtype=np.uint8).astype(np.float32)
        return LEGACY_FRAME_LENGTH, [emg_raw], False

    def _extract_new_frame(self, raw_buffer: bytearray):
        if len(raw_buffer) < 3:
            return None

        length_byte = int(raw_buffer[2])
        frame_len = 2 + 1 + length_byte
        if frame_len <= 4:
            return 1, [], True

        if len(raw_buffer) < frame_len:
            return None

        frame = raw_buffer[:frame_len]
        if frame[-1] != NEW_TAIL:
            return 1, [], True

        payload = frame[3:-1]
        if len(payload) < NEW_EXPECTED_PAYLOAD_NO_TAIL:
            # Accept partial variants if at least one full EMG pack exists.
            min_needed = NEW_META_BYTES + NUM_EMG_CHANNELS + NEW_BATTERY_BYTES
            if len(payload) < min_needed:
                return frame_len, [], True

        emg_start = NEW_META_BYTES
        emg_bytes = payload[emg_start:-NEW_BATTERY_BYTES]
        pack_count = len(emg_bytes) // NUM_EMG_CHANNELS
        if pack_count <= 0:
            return frame_len, [], True

        samples = []
        for i in range(pack_count):
            s = i * NUM_EMG_CHANNELS
            e = s + NUM_EMG_CHANNELS
            chunk = emg_bytes[s:e]
            if len(chunk) != NUM_EMG_CHANNELS:
                break
            samples.append(np.frombuffer(chunk, dtype=np.uint8).astype(np.float32))

        if not samples:
            return frame_len, [], True
        return frame_len, samples, False

    def _ingest_emg_sample(self, emg_raw: np.ndarray) -> None:
        self._sample_counter += 1
        if self._sample_counter % self._decimate_ratio != 0:
            return

        emg_centered = emg_raw - self._center_value
        with self._lock:
            self._buffer.append(emg_centered)

    @staticmethod
    def _auto_detect_port() -> Optional[str]:
        if not SERIAL_AVAILABLE:
            return None

        ports = serial.tools.list_ports.comports()

        # Prefer Linux stable symlink when available by-id.
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ("ch340", "cp210", "ftdi", "usb serial")):
                logger.info("Auto-detected serial port: %s (%s)", p.device, p.description)
                return p.device

        for p in ports:
            if p.device:
                return p.device
        return None
