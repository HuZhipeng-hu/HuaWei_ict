package com.huaweiict.emg.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * 手势事件实体
 */
@Data
@TableName("gesture_event")
public class GestureEvent {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String deviceId;
    private LocalDateTime eventTime;
    private String gesture;
    private Float confidence;
    private Integer durationMs;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
}
