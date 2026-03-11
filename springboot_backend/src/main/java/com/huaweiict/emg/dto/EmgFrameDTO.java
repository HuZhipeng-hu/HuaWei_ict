package com.huaweiict.emg.dto;

import lombok.Data;
import java.util.List;

/**
 * 单帧 EMG 数据 DTO
 */
@Data
public class EmgFrameDTO {
    /** 设备时间戳 */
    private Long device_ts;
    /** 服务器时间 ISO 格式 */
    private String serverTime;
    /** EMG 10×8 二维数组 */
    private List<List<Integer>> emg;
    /** 加速度 [x, y, z] */
    private List<Integer> acc;
    /** 陀螺仪 [x, y, z] */
    private List<Integer> gyro;
    /** 姿态角 [pitch, roll, yaw] */
    private List<Integer> angle;
    /** 电池电量 */
    private Integer battery;
    /** 手势名称（可选） */
    private String gesture;
    /** 置信度（可选） */
    private Float confidence;
}
