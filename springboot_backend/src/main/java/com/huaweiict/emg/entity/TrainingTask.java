package com.huaweiict.emg.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 训练任务实体
 * 
 * 对应表: training_task
 * 用途: 管理模型训练任务的生命周期
 */
@Data
@TableName("training_task")
public class TrainingTask {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String taskName;
    private String taskStatus;    // pending/running/completed/failed/cancelled
    
    // 训练参数配置（JSON字符串）
    private String config;
    private String dataFilter;
    
    // 数据集统计
    private Integer totalSamples;
    private Integer trainSamples;
    private Integer valSamples;
    private Integer testSamples;
    private String gestureDistribution;
    
    // 训练进度
    private Integer currentEpoch;
    private Integer totalEpochs;
    private Float progressPercent;
    
    // 训练结果
    private String modelPath;
    private Float modelSizeMb;
    
    private Float trainAccuracy;
    private Float valAccuracy;
    private Float testAccuracy;
    
    private Float finalTrainLoss;
    private Float finalValLoss;
    
    private Integer bestEpoch;
    private Float bestValAccuracy;
    
    private String metrics;  // 详细指标（precision, recall, f1, confusion_matrix等，JSON字符串）
    
    // 时间信息
    private LocalDateTime createdTime;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer durationSeconds;
    
    // 日志与错误
    private String logFile;
    private String errorMessage;
    
    // 创建者
    private String createdBy;
}
