package com.huaweiict.emg.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 标注会话实体
 * 
 * 对应表: annotation_session
 * 用途: 记录用户的标注会话（批量标注场景）
 */
@Data
@TableName("annotation_session")
public class AnnotationSession {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String sessionName;
    private String deviceId;
    private String annotator;
    
    // 时间范围
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    
    // 统计
    private Integer totalFrames;
    private String gestureLabel;
    
    // 元数据
    private LocalDateTime createdTime;
}
