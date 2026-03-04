# -*- coding: utf-8 -*-
"""
realtime_ckpt.py - 基于 MindSpore .ckpt 模型的 sEMG 手势实时推理
================================================================
直接加载 NeuroGripNet 的 .ckpt 权重文件，从 EMG 臂环实时采集数据，
经过与训练时一致的预处理流水线（降采样 → 带通滤波 → 整流 → 标准化 → STFT），
输出 6 类手势的实时分类结果。

手势类别: RELAX(放松) | FIST(握拳) | PINCH(捏取) | OK | YE(剪刀手) | SIDEGRIP(侧握)

用法:
  cd code
  python scripts/realtime_ckpt.py
  python scripts/realtime_ckpt.py --port COM3
  python scripts/realtime_ckpt.py --ckpt checkpoints/neurogrip_best.ckpt --threshold 0.6
  python scripts/realtime_ckpt.py --config configs/runtime.yaml

依赖: pip install mindspore pyserial numpy scipy
"""

__version__ = '1.0.0'

import os
import sys
import time
import argparse
import logging
import numpy as np
from collections import deque, Counter
from typing import Optional, Tuple, List

# ── 路径设置：确保能导入 code/ 下的 shared 模块 ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_SCRIPT_DIR)       # code/
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# ── 导入 emg_armband（同目录）──
import emg_armband as emg

# ── 导入项目共享模块 ──
from shared.gestures import GestureType, LABEL_NAME_MAP, NUM_CLASSES
from shared.preprocessing import PreprocessPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realtime_ckpt")

# ── MindSpore 导入 ──
try:
    import mindspore as ms
    from mindspore import Tensor, load_checkpoint, load_param_into_net
    from shared.models import NeuroGripNet, NeuroGripNetLite, create_model
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    logger.warning("MindSpore 未安装，将使用模拟推理模式。安装命令: pip install mindspore")


# =============================================================================
# 常量 / 默认配置
# =============================================================================

# 设备协议
DEVICE_SAMPLING_RATE = 1000      # 臂环原始采样率 Hz
TARGET_SAMPLING_RATE = 200       # 目标采样率 Hz
DECIMATE_RATIO = DEVICE_SAMPLING_RATE // TARGET_SAMPLING_RATE  # 5
NUM_TOTAL_CHANNELS = 8           # 臂环总 EMG 通道数
NUM_USE_CHANNELS = 6             # 模型使用前 6 通道
EMG_CENTER_VALUE = 128.0         # uint8 中心值

# 预处理 / 模型
SEGMENT_LENGTH = 84              # 200Hz 下的窗口长度 (0.42s)
NUM_EMG_PACKS = 10               # emg_armband 每帧 10 个 EMG 包

# 推理后处理
DEFAULT_THRESHOLD = 0.5          # 置信度阈值
DEFAULT_VOTE_WINDOW = 5          # 投票窗口大小
DEFAULT_VOTE_MIN_COUNT = 3       # 最低投票数

# 手势显示名称（带 emoji）
GESTURE_DISPLAY = {
    'relax':     '✋ 放松 (RELAX)',
    'fist':      '✊ 握拳 (FIST)',
    'pinch':     '🤏 捏取 (PINCH)',
    'ok':        '👌 OK',
    'ye':        '✌  剪刀手 (YE)',
    'sidegrip':  '🤜 侧握 (SIDEGRIP)',
    'unknown':   '❓ 未知',
}


# =============================================================================
# Softmax 工具
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# =============================================================================
# 实时推理引擎
# =============================================================================

class NeuroGripRealtimeEngine:
    """
    基于 MindSpore .ckpt 的实时手势推理引擎

    工作流程:
        emg_armband 帧 → 提取 10 packs (1000Hz)
        → 降采样到 200Hz → 累积至 84 个采样点
        → PreprocessPipeline (滤波→整流→标准化→STFT)
        → NeuroGripNet 推理 → softmax → 投票平滑 → 输出手势
    """

    def __init__(
        self,
        ckpt_path: str,
        model_config: Optional[dict] = None,
        preprocess_config: Optional[dict] = None,
        threshold: float = DEFAULT_THRESHOLD,
        vote_window: int = DEFAULT_VOTE_WINDOW,
        vote_min_count: int = DEFAULT_VOTE_MIN_COUNT,
        device: str = "CPU",
    ):
        """
        Args:
            ckpt_path: .ckpt 模型文件路径
            model_config: 模型配置 (可选, 默认使用训练配置)
            preprocess_config: 预处理配置 (可选, 默认使用训练配置)
            threshold: 置信度阈值
            vote_window: 投票窗口大小
            vote_min_count: 输出所需最低投票数
            device: MindSpore 设备 ("CPU" / "GPU" / "Ascend")
        """
        self.ckpt_path = ckpt_path
        self.threshold = threshold
        self.vote_window = vote_window
        self.vote_min_count = vote_min_count

        # ── 模型配置 ──
        if model_config is None:
            model_config = {
                "model_type": "standard",
                "in_channels": NUM_USE_CHANNELS,
                "num_classes": NUM_CLASSES,
                "base_channels": 16,
                "use_se": True,
                "dropout_rate": 0.0,     # 推理时关闭 Dropout
            }
        self.model_config = model_config

        # ── 预处理配置 ──
        if preprocess_config is None:
            preprocess_config = {
                "sampling_rate": TARGET_SAMPLING_RATE,
                "num_channels": NUM_USE_CHANNELS,
                "lowcut": 20.0,
                "highcut": 90.0,
                "filter_order": 4,
                "stft_window_size": 24,
                "stft_hop_size": 12,
                "stft_n_fft": 46,
            }
        self.preprocess_config = preprocess_config

        # ── 构建预处理流水线 ──
        self.pipeline = PreprocessPipeline(
            sampling_rate=preprocess_config["sampling_rate"],
            num_channels=preprocess_config["num_channels"],
            lowcut=preprocess_config["lowcut"],
            highcut=preprocess_config["highcut"],
            filter_order=preprocess_config["filter_order"],
            stft_window_size=preprocess_config["stft_window_size"],
            stft_hop_size=preprocess_config["stft_hop_size"],
            stft_n_fft=preprocess_config["stft_n_fft"],
        )

        # ── 构建 & 加载模型 ──
        self.model = None
        self._mock_mode = False
        self._load_model(ckpt_path, model_config, device)

        # ── 数据缓冲区 ──
        # 存储降采样后的 EMG 数据 (200Hz), 每个元素 shape = (NUM_USE_CHANNELS,)
        self._sample_buffer: deque = deque(maxlen=SEGMENT_LENGTH * 2)
        self._decimate_counter = 0   # 降采样计数器（跨帧连续）

        # ── 投票缓冲区 ──
        self._vote_buffer: deque = deque(maxlen=vote_window)
        self._last_gesture_id = 0    # 默认 RELAX
        self._last_confidence = 0.0

        # ── 统计 ──
        self.inference_count = 0
        self.total_frames = 0
        self._last_infer_ms = 0.0

    def _load_model(self, ckpt_path: str, config: dict, device: str):
        """加载 MindSpore 模型 + ckpt 权重"""
        if not MINDSPORE_AVAILABLE:
            logger.warning("MindSpore 不可用，使用模拟推理")
            self._mock_mode = True
            return

        try:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=device)
        except Exception:
            # CPU fallback
            ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
            logger.info("回退到 CPU 设备")

        # 创建模型
        self.model = create_model(config)
        self.model.set_train(False)

        # 加载权重
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

        param_dict = load_checkpoint(ckpt_path)
        not_loaded, _ = load_param_into_net(self.model, param_dict)

        if not_loaded:
            logger.warning(f"以下参数未加载: {not_loaded}")

        # 获取参数量
        total_params = sum(p.size for p in self.model.trainable_params())
        logger.info(f"模型已加载: {config['model_type']}, "
                    f"参数量: {total_params:,}, 设备: {device}")

    def feed_frame(self, frame_event) -> Optional[Tuple[int, str, float]]:
        """
        输入一帧来自 emg_armband 的完整帧事件

        从 10 个 EMG 包中提取数据，降采样后累积。
        当缓冲区达到 SEGMENT_LENGTH 时执行推理。

        Args:
            frame_event: emg.FrameEvent

        Returns:
            (gesture_id, gesture_name, confidence) 或 None (缓冲区未满)
        """
        self.total_frames += 1

        # ── 提取 10 packs × 8ch → 降采样 → 取前 6ch ──
        emg_packs = frame_event.emg_event.emg  # List[List[int]]: 10×8

        for pack in emg_packs:
            self._decimate_counter += 1

            # 每 DECIMATE_RATIO 个样本保留 1 个 (1000Hz → 200Hz)
            if self._decimate_counter % DECIMATE_RATIO != 0:
                continue

            # 取前 6 通道, 中心化 (uint8 128 → 0)
            sample = np.array(pack[:NUM_USE_CHANNELS], dtype=np.float32)
            sample -= EMG_CENTER_VALUE
            self._sample_buffer.append(sample)

        # ── 检查缓冲区是否够做推理 ──
        if len(self._sample_buffer) < SEGMENT_LENGTH:
            return None

        # ── 取最新窗口 ──
        window = np.array(
            list(self._sample_buffer)[-SEGMENT_LENGTH:],
            dtype=np.float32,
        )  # shape: (84, 6)

        # ── 预处理 → STFT 时频谱图 ──
        try:
            spectrogram = self.pipeline.process_window(window)
            # shape: (6, freq_bins, time_frames) ≈ (6, 24, 13)
        except Exception as e:
            logger.debug(f"预处理异常: {e}")
            return None

        # ── 模型推理 ──
        gesture_id, confidence = self._infer(spectrogram)

        # ── 投票平滑 ──
        if confidence >= self.threshold:
            self._vote_buffer.append(gesture_id)
        else:
            self._vote_buffer.append(0)  # 低置信度视为 RELAX

        voted_id = self._majority_vote()
        voted_name = LABEL_NAME_MAP.get(voted_id, "unknown")

        self._last_gesture_id = voted_id
        self._last_confidence = confidence
        self.inference_count += 1

        return voted_id, voted_name, confidence

    def _infer(self, spectrogram: np.ndarray) -> Tuple[int, float]:
        """
        执行一次模型推理

        Args:
            spectrogram: (C, F, T) float32

        Returns:
            (gesture_id, confidence)
        """
        if self._mock_mode:
            return self._mock_infer()

        # (C, F, T) → (1, C, F, T)
        x = spectrogram[np.newaxis, ...].astype(np.float32)

        t0 = time.perf_counter()
        input_tensor = Tensor(x, ms.float32)
        logits = self.model(input_tensor)
        logits_np = logits.asnumpy()[0]  # (num_classes,)
        self._last_infer_ms = (time.perf_counter() - t0) * 1000

        probs = softmax(logits_np)
        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])

        return gesture_id, confidence

    def _mock_infer(self) -> Tuple[int, float]:
        """模拟推理（无 MindSpore 时）"""
        time.sleep(0.01)
        gesture_id = np.random.randint(0, NUM_CLASSES)
        confidence = np.random.uniform(0.3, 0.95)
        self._last_infer_ms = 10.0
        return int(gesture_id), float(confidence)

    def _majority_vote(self) -> int:
        """对投票缓冲区做多数投票"""
        if not self._vote_buffer:
            return 0

        counts = Counter(self._vote_buffer)
        most_common_id, most_common_count = counts.most_common(1)[0]

        if most_common_count >= self.vote_min_count:
            return most_common_id
        else:
            return self._last_gesture_id  # 维持上一个手势

    def get_buffer_progress(self) -> float:
        """缓冲区填充进度 0~1"""
        return min(1.0, len(self._sample_buffer) / SEGMENT_LENGTH)

    def reset(self):
        """重置所有缓冲区"""
        self._sample_buffer.clear()
        self._vote_buffer.clear()
        self._decimate_counter = 0
        self._last_gesture_id = 0
        self._last_confidence = 0.0

    @property
    def last_gesture(self) -> str:
        return LABEL_NAME_MAP.get(self._last_gesture_id, "unknown")

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def last_latency_ms(self) -> float:
        return self._last_infer_ms


# =============================================================================
# 实时监听器
# =============================================================================

class RealtimeCkptListener(emg.DeviceListener):
    """
    绑定 NeuroGripRealtimeEngine 的 emg_armband 监听器

    每收到一帧数据就喂给推理引擎，并在终端显示结果。
    """

    def __init__(
        self,
        engine: NeuroGripRealtimeEngine,
        print_interval: float = 0.25,
        on_gesture_change=None,
    ):
        self.engine = engine
        self.print_interval = print_interval
        self.on_gesture_change = on_gesture_change
        self._last_print_time = 0.0
        self._last_gesture = None

    def on_connected(self, device):
        logger.info(f"臂环已连接: {device.port}")

    def on_disconnected(self, device):
        logger.info("臂环已断开")

    def on_frame(self, event):
        result = self.engine.feed_frame(event)

        now = time.time()

        if result is None:
            # 缓冲区填充中
            if now - self._last_print_time > 0.5:
                progress = self.engine.get_buffer_progress()
                print(f"\r  缓冲区填充: {progress * 100:.0f}%  "
                      f"(需要 {SEGMENT_LENGTH} 个200Hz样本)",
                      end='', flush=True)
                self._last_print_time = now
            return

        gesture_id, gesture_name, confidence = result

        # 手势变化回调
        if gesture_name != self._last_gesture:
            self._last_gesture = gesture_name
            if self.on_gesture_change:
                self.on_gesture_change(gesture_id, gesture_name, confidence)

        # 定期打印
        if now - self._last_print_time >= self.print_interval:
            display = GESTURE_DISPLAY.get(gesture_name, gesture_name)
            bar_len = int(confidence * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            latency = self.engine.last_latency_ms

            print(
                f"\r  手势: {display:<22s} [{bar}] {confidence:5.1%}  "
                f"推理: {latency:5.1f}ms  "
                f"帧:{self.engine.total_frames:6d}  "
                f"次:{self.engine.inference_count:5d}",
                end='', flush=True,
            )
            self._last_print_time = now


# =============================================================================
# 主程序
# =============================================================================

def load_yaml_config(config_path: str) -> dict:
    """从 YAML 文件加载配置"""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML 未安装，使用默认配置")
        return {}

    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def run_realtime(args):
    """实时推理主入口"""

    # ── 加载配置 ──
    yaml_cfg = {}
    if args.config and os.path.exists(args.config):
        yaml_cfg = load_yaml_config(args.config)
        logger.info(f"已加载配置: {args.config}")

    # 模型配置（命令行 > YAML > 默认值）
    model_cfg = yaml_cfg.get("model", {})
    model_config = {
        "model_type": model_cfg.get("model_type", "standard"),
        "in_channels": model_cfg.get("in_channels", NUM_USE_CHANNELS),
        "num_classes": model_cfg.get("num_classes", NUM_CLASSES),
        "base_channels": model_cfg.get("base_channels", 16),
        "use_se": model_cfg.get("use_se", True),
        "dropout_rate": 0.0,  # 推理时始终关闭 Dropout
    }

    # 预处理配置
    prep_cfg = yaml_cfg.get("preprocess", {})
    preprocess_config = {
        "sampling_rate": prep_cfg.get("sampling_rate", TARGET_SAMPLING_RATE),
        "num_channels": prep_cfg.get("num_channels", NUM_USE_CHANNELS),
        "lowcut": prep_cfg.get("lowcut", 20.0),
        "highcut": prep_cfg.get("highcut", 90.0),
        "filter_order": prep_cfg.get("filter_order", 4),
        "stft_window_size": prep_cfg.get("stft_window_size", 24),
        "stft_hop_size": prep_cfg.get("stft_hop_size", 12),
        "stft_n_fft": prep_cfg.get("stft_n_fft", 46),
    }

    # 后处理配置
    threshold = args.threshold or yaml_cfg.get("confidence_threshold", DEFAULT_THRESHOLD)
    vote_window = args.vote_window or yaml_cfg.get("vote_window_size", DEFAULT_VOTE_WINDOW)
    vote_min = args.vote_min or yaml_cfg.get("vote_min_count", DEFAULT_VOTE_MIN_COUNT)

    # 串口
    port = args.port
    if port is None:
        hw_cfg = yaml_cfg.get("hardware", {})
        port = hw_cfg.get("sensor_port", None) or "COM4"

    # ckpt 路径
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        # 尝试相对于 code/ 目录
        alt = os.path.join(_CODE_DIR, ckpt_path)
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            print(f"✗ 模型文件不存在: {ckpt_path}")
            sys.exit(1)

    # 推理设备
    device = args.device or yaml_cfg.get("inference", {}).get("device", "CPU")
    # 规范化设备名
    device_map = {"NPU": "Ascend", "GPU": "GPU", "CPU": "CPU",
                  "ASCEND": "Ascend", "ascend": "Ascend"}
    device = device_map.get(device, device)

    # ── 打印横幅 ──
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " NeuroGrip sEMG 手势实时推理 (MindSpore .ckpt) ".center(52) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    print(f"  模型文件:  {ckpt_path}")
    print(f"  模型类型:  {model_config['model_type']}  "
          f"(in={model_config['in_channels']}, classes={model_config['num_classes']}, "
          f"base={model_config['base_channels']}, SE={model_config['use_se']})")
    print(f"  推理设备:  {device}")
    print(f"  预处理:    {TARGET_SAMPLING_RATE}Hz, "
          f"STFT({preprocess_config['stft_window_size']}/"
          f"{preprocess_config['stft_hop_size']}/"
          f"{preprocess_config['stft_n_fft']})")
    print(f"  窗口长度:  {SEGMENT_LENGTH} 样本 ({SEGMENT_LENGTH / TARGET_SAMPLING_RATE:.2f}s)")
    print(f"  串口:      {port}")
    print(f"  置信阈值:  {threshold}")
    print(f"  投票窗口:  {vote_window} (最低 {vote_min} 票)")
    print(f"  手势类别:  {', '.join(LABEL_NAME_MAP.values())}")
    print()

    # ── 创建推理引擎 ──
    engine = NeuroGripRealtimeEngine(
        ckpt_path=ckpt_path,
        model_config=model_config,
        preprocess_config=preprocess_config,
        threshold=threshold,
        vote_window=vote_window,
        vote_min_count=vote_min,
        device=device,
    )

    # ── 手势变化回调 ──
    def on_gesture_change(gesture_id, gesture_name, confidence):
        if gesture_name != 'relax':
            display = GESTURE_DISPLAY.get(gesture_name, gesture_name)
            print(f"\n  >>> 手势变化: {display}  (置信度: {confidence:.1%})")
            try:
                import winsound
                winsound.Beep(800, 100)
            except Exception:
                pass

    # ── 创建监听器 ──
    listener = RealtimeCkptListener(
        engine=engine,
        print_interval=0.2,
        on_gesture_change=on_gesture_change,
    )

    # ── 启动 Hub ──
    hub = emg.Hub(port=port)

    print(f"  {'─' * 55}")
    print("  按 Ctrl+C 停止")
    print(f"  {'─' * 55}")
    print()

    try:
        hub.run(listener)
    except KeyboardInterrupt:
        print("\n")
    finally:
        hub.stop()
        print()
        print(f"  ── 运行统计 ──")
        print(f"  总帧数:      {engine.total_frames}")
        print(f"  推理次数:    {engine.inference_count}")
        print(f"  最后手势:    {engine.last_gesture} ({engine.last_confidence:.1%})")
        print(f"  最后延迟:    {engine.last_latency_ms:.1f} ms")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NeuroGrip sEMG 手势实时分类 (.ckpt 模型)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/realtime_ckpt.py
  python scripts/realtime_ckpt.py --ckpt checkpoints/neurogrip_best.ckpt
  python scripts/realtime_ckpt.py --port COM3 --threshold 0.6
  python scripts/realtime_ckpt.py --config configs/runtime.yaml --device CPU
        """,
    )

    parser.add_argument(
        '--ckpt', type=str,
        default='checkpoints/neurogrip_best.ckpt',
        help='MindSpore .ckpt 模型路径 (默认: checkpoints/neurogrip_best.ckpt)',
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/runtime.yaml',
        help='YAML 配置文件路径 (默认: configs/runtime.yaml)',
    )
    parser.add_argument(
        '--port', type=str, default=None,
        help='串口号 (默认: COM4 或配置文件中的值)',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='推理设备: CPU / GPU / Ascend (默认: CPU)',
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help=f'置信度阈值 (默认: {DEFAULT_THRESHOLD})',
    )
    parser.add_argument(
        '--vote-window', type=int, default=None,
        help=f'投票窗口大小 (默认: {DEFAULT_VOTE_WINDOW})',
    )
    parser.add_argument(
        '--vote-min', type=int, default=None,
        help=f'最低投票数 (默认: {DEFAULT_VOTE_MIN_COUNT})',
    )

    args = parser.parse_args()
    run_realtime(args)


if __name__ == '__main__':
    main()
