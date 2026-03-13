package com.huaweiict.emg.dto;

import lombok.Data;

import java.time.LocalDateTime;

/**
 * 数据标注请求DTO
 */
@Data
public class AnnotationRequest {
    
    /** 设备ID */
    private String deviceId;
    
    /** 开始时间 */
    private LocalDateTime startTime;
    
    /** 结束时间 */
    private LocalDateTime endTime;
    
    /** 手势标签 */
    private String gestureLabel;
    
    /** 标注人 */
    private String annotator;
    
    /** 标注备注 */
    private String annotationNote;
    
    /** 是否计算质量分数 */
    private Boolean calculateQuality = true;
}
