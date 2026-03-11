package com.huaweiict.emg.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * EMG 帧数据实体
 */
@Data
@TableName("emg_frame")
public class EmgFrame {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String deviceId;

    private Long deviceTs;

    private LocalDateTime serverTime;

    /** EMG 10x8 JSON 字符串 */
    private String emgData;

    /** EMG Pack1 的8通道，逗号分隔 */
    private String emgPack1;

    private Integer accX;
    private Integer accY;
    private Integer accZ;

    private Integer gyroX;
    private Integer gyroY;
    private Integer gyroZ;

    private Integer pitch;
    private Integer roll;
    private Integer yaw;

    private Integer battery;

    private String gesture;
    private Float confidence;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
}
