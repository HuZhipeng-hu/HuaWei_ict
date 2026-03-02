"""
CSV 数据集加载器

从思知瑞臂环采集的 CSV 文件中加载 EMG 数据并转换为训练样本。

数据组织方式（目录即标签）:
    data_dir/
    ├── Relax/       → GestureType.RELAX (0)
    │   ├── 1772246123.csv
    │   └── ...
    ├── fist/        → GestureType.FIST (1)
    ├── Pinch/       → GestureType.PINCH (2)
    ├── ok/          → GestureType.OK (3)
    ├── ye/          → GestureType.YE (4)
    └── Sidegrip/    → GestureType.SIDEGRIP (5)

CSV 格式（每行一个采样点，1000Hz 采样率，共 17 列）:
    emg1,emg2,...,emg8,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,angle_pitch,angle_roll,angle_yaw

处理流程:
    1. 扫描目录，按文件夹名分配标签
    2. 读取 CSV → 提取前 N 通道 EMG → uint8 减 128 归零
    3. 降采样（1000Hz → 200Hz）
    4. 滑动窗口切分为固定长度片段
    5. 对每个片段执行预处理流水线 → 时频谱图
    6. 可选数据增强
    7. 封装为 (spectrogram, label) 对
"""

import csv
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

from shared.gestures import GestureType, FOLDER_TO_GESTURE, NUM_CLASSES
from shared.preprocessing import PreprocessPipeline, SignalWindower

logger = logging.getLogger(__name__)


class CSVDatasetLoader:
    """
    思知瑞臂环 CSV 数据集加载器

    将磁盘上的原始 CSV 文件批量处理为模型可用的训练样本。

    Usage:
        loader = CSVDatasetLoader(data_dir="data/", ...)
        samples, labels = loader.load_all()
        train_data, val_data = loader.split(samples, labels, val_ratio=0.2)

    Args:
        data_dir: 数据根目录（包含各手势子文件夹）
        preprocess: 预处理流水线实例
        num_emg_channels: 使用的 EMG 通道数
        device_sampling_rate: 设备原始采样率 (Hz)
        target_sampling_rate: 目标采样率 (Hz)
        segment_length: 训练样本的采样点数（基于目标采样率）
        segment_stride: 样本间滑动步长
        center_value: 归零中心值（思知瑞臂环 uint8 数据的中心为 128）
    """

    def __init__(
        self,
        data_dir: str,
        preprocess: PreprocessPipeline,
        num_emg_channels: int = 6,
        device_sampling_rate: int = 1000,
        target_sampling_rate: int = 200,
        segment_length: int = 84,
        segment_stride: int = 42,
        center_value: float = 128.0,
    ):
        self.data_dir = Path(data_dir)
        self.preprocess = preprocess
        self.num_emg_channels = num_emg_channels
        self.device_sampling_rate = device_sampling_rate
        self.target_sampling_rate = target_sampling_rate
        self.segment_length = segment_length
        self.segment_stride = segment_stride
        self.center_value = center_value

        # 降采样比
        self.decimate_ratio = device_sampling_rate // target_sampling_rate

        # 验证数据目录
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 滑动窗口切分器（使用共享核心的 SignalWindower）
        self.windower = SignalWindower(
            window_size=segment_length,
            stride=segment_stride,
        )

        # 扫描可用手势文件夹
        self.gesture_folders = self._scan_folders()
        if not self.gesture_folders:
            raise ValueError(f"在 {self.data_dir} 中未找到有效的手势文件夹")

    def _scan_folders(self) -> Dict[GestureType, List[Path]]:
        """
        扫描数据目录，找到所有手势文件夹及其 CSV 文件

        Returns:
            {GestureType: [csv_path1, csv_path2, ...]}
        """
        result: Dict[GestureType, List[Path]] = {}

        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name.lower()
            if folder_name not in FOLDER_TO_GESTURE:
                logger.warning(f"跳过未识别的文件夹: {folder.name}")
                continue

            gesture = FOLDER_TO_GESTURE[folder_name]
            csv_files = sorted(folder.glob("*.csv"))

            if not csv_files:
                logger.warning(f"文件夹 {folder.name} 中没有 CSV 文件")
                continue

            result[gesture] = csv_files
            logger.info(
                f"[{gesture.name}] {folder.name}/ → {len(csv_files)} 个文件"
            )

        return result

    def _read_csv(self, csv_path: Path) -> np.ndarray:
        """
        读取单个 CSV 文件，提取 EMG 数据

        Args:
            csv_path: CSV 文件路径

        Returns:
            emg_data: (num_samples, num_emg_channels) float32
                      已减去 center_value 归零
        """
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # 跳过表头
            for row in reader:
                if len(row) < self.num_emg_channels:
                    continue
                try:
                    # 只取前 num_emg_channels 列（EMG数据）
                    emg_values = [float(row[i]) for i in range(self.num_emg_channels)]
                    rows.append(emg_values)
                except (ValueError, IndexError):
                    continue

        if not rows:
            logger.warning(f"CSV 文件为空或格式错误: {csv_path}")
            return np.empty((0, self.num_emg_channels), dtype=np.float32)

        data = np.array(rows, dtype=np.float32)

        # 归零: uint8 数据中心为 128
        data -= self.center_value

        return data

    def _decimate(self, signal: np.ndarray) -> np.ndarray:
        """
        降采样（简单抽取）

        Args:
            signal: (num_samples, num_channels)

        Returns:
            降采样后的信号
        """
        if self.decimate_ratio <= 1:
            return signal
        return signal[::self.decimate_ratio]

    def _segment(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        将连续信号切分为固定长度的片段（委托给共享核心的 SignalWindower）

        Args:
            signal: (num_samples, num_channels)

        Returns:
            片段列表，每个形状为 (segment_length, num_channels)
        """
        return self.windower.segment(signal)

    def load_file(self, csv_path: Path) -> List[np.ndarray]:
        """
        加载单个 CSV 文件并处理为时频谱图列表

        流程: 读取CSV → 归零 → 降采样 → 切片 → 逐片段预处理

        Args:
            csv_path: CSV 文件路径

        Returns:
            时频谱图列表，每个形状 (num_channels, freq_bins, time_frames)
        """
        # 1. 读取并归零
        emg_data = self._read_csv(csv_path)
        if emg_data.shape[0] == 0:
            return []

        # 2. 降采样
        decimated = self._decimate(emg_data)
        if decimated.shape[0] < self.segment_length:
            logger.debug(
                f"文件 {csv_path.name} 降采样后长度 {decimated.shape[0]} "
                f"< 片段长度 {self.segment_length}，跳过"
            )
            return []

        # 3. 切片
        segments = self._segment(decimated)

        # 4. 逐片段预处理 → 时频谱图
        spectrograms = []
        for segment in segments:
            try:
                spec = self.preprocess.process(segment)
                spectrograms.append(spec)
            except Exception as e:
                logger.debug(f"预处理失败 ({csv_path.name}): {e}")
                continue

        return spectrograms

    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载所有数据

        Returns:
            (samples, labels):
                samples: (N, num_channels, freq_bins, time_frames) float32
                labels:  (N,) int32 手势标签
        """
        all_samples = []
        all_labels = []

        for gesture, csv_files in self.gesture_folders.items():
            gesture_samples = 0

            for csv_path in csv_files:
                spectrograms = self.load_file(csv_path)
                for spec in spectrograms:
                    all_samples.append(spec)
                    all_labels.append(gesture.value)
                    gesture_samples += 1

            logger.info(
                f"[{gesture.name}] 加载完成: "
                f"{len(csv_files)} 个文件 → {gesture_samples} 个样本"
            )

        if not all_samples:
            raise ValueError("未加载到任何有效样本，请检查数据目录和格式")

        samples = np.array(all_samples, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        logger.info(
            f"数据集加载完成: {len(samples)} 个样本, "
            f"形状 {samples.shape}, "
            f"{NUM_CLASSES} 个类别"
        )

        # 打印各类别样本数
        for gesture in GestureType:
            count = np.sum(labels == gesture.value)
            logger.info(f"  {gesture.name}: {count} 个样本")

        return samples, labels

    @staticmethod
    def split(
        samples: np.ndarray,
        labels: np.ndarray,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        将数据集划分为训练集和验证集（分层采样）

        确保每个类别在训练集和验证集中的比例一致。

        Args:
            samples: (N, C, F, T) 所有样本
            labels: (N,) 所有标签
            val_ratio: 验证集占比
            seed: 随机种子（确保可复现）

        Returns:
            ((train_samples, train_labels), (val_samples, val_labels))
        """
        rng = np.random.RandomState(seed)

        train_indices = []
        val_indices = []

        for class_id in range(NUM_CLASSES):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            n_val = max(1, int(len(class_indices) * val_ratio))
            val_indices.extend(class_indices[:n_val])
            train_indices.extend(class_indices[n_val:])

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        return (
            (samples[train_indices], labels[train_indices]),
            (samples[val_indices], labels[val_indices]),
        )

    @staticmethod
    def kfold_split(
        samples: np.ndarray,
        labels: np.ndarray,
        k: int = 5,
        seed: int = 42,
    ):
        """
        K-Fold 分层交叉验证

        将数据按类别均匀分成 K 折，每次用 1 折做验证、其余做训练。
        通过 generator 逐折 yield，调用方可迭代使用。

        Args:
            samples: (N, C, F, T) 所有样本
            labels: (N,) 所有标签
            k: 折数
            seed: 随机种子

        Yields:
            fold_idx, (train_samples, train_labels), (val_samples, val_labels)
        """
        rng = np.random.RandomState(seed)

        # 按类别分组索引并打乱
        class_indices = {}
        for class_id in range(NUM_CLASSES):
            idx = np.where(labels == class_id)[0]
            rng.shuffle(idx)
            class_indices[class_id] = idx

        # K 折划分
        for fold in range(k):
            train_idx = []
            val_idx = []

            for class_id, idx in class_indices.items():
                n = len(idx)
                fold_size = n // k
                start = fold * fold_size
                end = start + fold_size if fold < k - 1 else n

                val_idx.extend(idx[start:end])
                train_idx.extend(idx[:start])
                train_idx.extend(idx[end:])

            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)

            yield (
                fold,
                (samples[train_idx], labels[train_idx]),
                (samples[val_idx], labels[val_idx]),
            )

    def get_stats(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        stats = {"total_files": 0}
        for gesture, files in self.gesture_folders.items():
            stats[gesture.name] = len(files)
            stats["total_files"] += len(files)
        return stats
