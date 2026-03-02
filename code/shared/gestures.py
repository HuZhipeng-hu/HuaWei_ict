"""
手势定义与手指映射

定义所有支持的手势类型、手势标签和对应的义肢手指状态映射。
这是整个系统的"语言"——训练、转换、运行时三个模块都基于此定义通信。

扩展新手势只需：
1. 在 GestureType 枚举中添加新成员
2. 在 GESTURE_FINGER_MAP 中添加对应的手指映射
"""

from enum import IntEnum
from typing import Dict, List, Tuple


# =============================================================================
# 手指状态
# =============================================================================

class FingerState(IntEnum):
    """单根手指的状态"""
    OPEN = 0        # 伸直张开
    HALF = 1        # 半弯曲
    CLOSED = 2      # 完全弯曲（握紧）


# =============================================================================
# 手势定义
# =============================================================================

class GestureType(IntEnum):
    """
    手势类型枚举

    枚举值即为模型输出的类别索引（0~N-1）。
    添加新手势时在末尾追加即可，注意同步更新 GESTURE_FINGER_MAP。
    """
    RELAX     = 0   # 放松 / 静息
    FIST      = 1   # 握拳
    PINCH     = 2   # 捏取（拇指 + 食指）
    OK        = 3   # OK 手势
    YE        = 4   # 剪刀手 / 耶
    SIDEGRIP  = 5   # 侧握


# 类别总数（模型输出维度）
NUM_CLASSES = len(GestureType)

# 手势名称 → 整数标签（供数据加载器使用）
GESTURE_LABEL_MAP: Dict[str, int] = {
    gesture.name.lower(): gesture.value
    for gesture in GestureType
}

# 整数标签 → 手势名称（供日志/可视化使用）
LABEL_NAME_MAP: Dict[int, str] = {
    v: k for k, v in GESTURE_LABEL_MAP.items()
}

# 数据文件夹名 → 手势类型（CSV 数据集目录名到标签的映射）
# 键为 data/ 下的子文件夹名（大小写不敏感），值为 GestureType
FOLDER_TO_GESTURE: Dict[str, GestureType] = {
    "relax":     GestureType.RELAX,
    "fist":      GestureType.FIST,
    "pinch":     GestureType.PINCH,
    "ok":        GestureType.OK,
    "ye":        GestureType.YE,
    "sidegrip":  GestureType.SIDEGRIP,
}


# =============================================================================
# 手指映射
# =============================================================================

# 手指索引定义（5指义肢）
FINGER_THUMB  = 0   # 拇指
FINGER_INDEX  = 1   # 食指
FINGER_MIDDLE = 2   # 中指
FINGER_RING   = 3   # 无名指
FINGER_PINKY  = 4   # 小指

NUM_FINGERS = 5

# EMG 通道数（思知瑞臂环 8 通道）
NUM_EMG_CHANNELS = 8

# 每种手势对应的 5 指状态：[拇指, 食指, 中指, 无名指, 小指]
GESTURE_FINGER_MAP: Dict[GestureType, List[FingerState]] = {
    GestureType.RELAX: [
        FingerState.OPEN,       # 拇指 张开
        FingerState.OPEN,       # 食指 张开
        FingerState.OPEN,       # 中指 张开
        FingerState.OPEN,       # 无名指 张开
        FingerState.OPEN,       # 小指 张开
    ],
    GestureType.FIST: [
        FingerState.CLOSED,     # 拇指 握紧
        FingerState.CLOSED,     # 食指 握紧
        FingerState.CLOSED,     # 中指 握紧
        FingerState.CLOSED,     # 无名指 握紧
        FingerState.CLOSED,     # 小指 握紧
    ],
    GestureType.PINCH: [
        FingerState.HALF,       # 拇指 半弯（捏合）
        FingerState.HALF,       # 食指 半弯（捏合）
        FingerState.OPEN,       # 中指 张开
        FingerState.OPEN,       # 无名指 张开
        FingerState.OPEN,       # 小指 张开
    ],
    GestureType.OK: [
        FingerState.HALF,       # 拇指 弯曲（圈合）
        FingerState.HALF,       # 食指 弯曲（圈合）
        FingerState.OPEN,       # 中指 张开
        FingerState.OPEN,       # 无名指 张开
        FingerState.OPEN,       # 小指 张开
    ],
    GestureType.YE: [
        FingerState.CLOSED,     # 拇指 握紧
        FingerState.OPEN,       # 食指 伸出
        FingerState.OPEN,       # 中指 伸出
        FingerState.CLOSED,     # 无名指 握紧
        FingerState.CLOSED,     # 小指 握紧
    ],
    GestureType.SIDEGRIP: [
        FingerState.HALF,       # 拇指 半弯（侧向）
        FingerState.CLOSED,     # 食指 握紧
        FingerState.CLOSED,     # 中指 握紧
        FingerState.CLOSED,     # 无名指 握紧
        FingerState.CLOSED,     # 小指 握紧
    ],
}


# =============================================================================
# 验证与工具函数
# =============================================================================

def validate_gesture_definitions() -> bool:
    """
    验证手势定义的完整性和一致性。

    检查:
    - 每个 GestureType 都有对应的手指映射
    - 每个手指映射恰好有 NUM_FINGERS 个元素
    - FOLDER_TO_GESTURE 覆盖了所有手势类型

    Returns:
        True 如果所有检查通过
    Raises:
        ValueError 如果发现不一致
    """
    for gesture in GestureType:
        # 检查手指映射完整性
        if gesture not in GESTURE_FINGER_MAP:
            raise ValueError(
                f"手势 {gesture.name} 缺少手指映射，"
                f"请在 GESTURE_FINGER_MAP 中添加"
            )
        finger_states = GESTURE_FINGER_MAP[gesture]
        if len(finger_states) != NUM_FINGERS:
            raise ValueError(
                f"手势 {gesture.name} 的手指映射长度为 {len(finger_states)}，"
                f"期望 {NUM_FINGERS}"
            )

    # 检查文件夹映射覆盖所有手势
    mapped_gestures = set(FOLDER_TO_GESTURE.values())
    all_gestures = set(GestureType)
    missing = all_gestures - mapped_gestures
    if missing:
        raise ValueError(
            f"以下手势缺少文件夹映射: "
            f"{[g.name for g in missing]}，"
            f"请在 FOLDER_TO_GESTURE 中添加"
        )

    return True


def get_finger_angles(
    gesture: GestureType,
    angle_open: float = 0.0,
    angle_half: float = 90.0,
    angle_closed: float = 180.0,
) -> List[float]:
    """
    将手势的手指状态转换为舵机角度。

    Args:
        gesture: 手势类型
        angle_open: OPEN 状态对应的角度（度）
        angle_half: HALF 状态对应的角度（度）
        angle_closed: CLOSED 状态对应的角度（度）

    Returns:
        5 个手指的舵机角度列表
    """
    state_to_angle = {
        FingerState.OPEN: angle_open,
        FingerState.HALF: angle_half,
        FingerState.CLOSED: angle_closed,
    }
    return [
        state_to_angle[state]
        for state in GESTURE_FINGER_MAP[gesture]
    ]
