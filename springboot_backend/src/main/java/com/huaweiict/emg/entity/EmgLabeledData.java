package com.huaweiict.emg.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * EMG 标注数据实体
 * 
 * 对应表: emg_labeled_data
 * 用途: 存储用户通过App标注的EMG数据，用于模型训练
 */
@Data
@TableName("emg_labeled_data")
public class EmgLabeledData {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String deviceId;
    
    // 时间信息
    private LocalDateTime captureTime;
    private Long deviceTs;
    
    // EMG数据（JSON字符串类型）
    // 存储为JSON字符串，使用FastJSON2序列化/反序列化
    private String emgData;      // 10x8数组
    private String accData;      // [x,y,z]
    private String gyroData;     // [x,y,z]
    private String angleData;    // [pitch,roll,yaw]
    private Integer battery;
    
    // 标注信息
    private String gestureLabel;   // fist/ok/pinch/relax/sidegrip/ye
    private String annotator;
    private LocalDateTime annotationTime;
    private String annotationNote;
    
    // 训练相关
    private Boolean isUsedForTraining;
    private String splitType;      // train/val/test
    private Long trainingTaskId;
    
    // 数据质量
    private Float qualityScore;
    private Float signalNoiseRatio;
    private Boolean isValid;
    
    // 元数据
    private LocalDateTime createdTime;
    private LocalDateTime updatedTime;
}
