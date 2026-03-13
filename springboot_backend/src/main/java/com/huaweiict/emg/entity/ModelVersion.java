package com.huaweiict.emg.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 模型版本实体
 * 
 * 对应表: model_version
 * 用途: 管理训练完成的模型版本
 */
@Data
@TableName("model_version")
public class ModelVersion {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String version;        // v1.0.0, v1.0.1
    private String modelName;
    
    // 模型文件信息
    private String modelPath;
    private String modelFormat;    // pytorch/onnx/tflite/mindspore
    private Float modelSizeMb;
    private String checksum;       // MD5/SHA256
    
    // 训练信息
    private Long trainingTaskId;
    private Integer trainedSamples;
    private String gestureClasses;   // ["fist", "ok", ...] JSON字符串
    private Integer numClasses;
    
    // 模型架构
    private String modelArchitecture;  // CNN/LSTM/CNN_LSTM/Transformer
    private String inputShape;       // {window_size: 150, channels: 8} JSON字符串
    
    // 性能指标
    private Float accuracy;
    private Float precisionScore;
    private Float recallScore;
    private Float f1Score;
    
    private Float inferenceTimeMs;
    private Long modelParamsCount;
    
    // 部署状态
    private Boolean isActive;
    private String deployedTo;   // ["orangepi_01", "cloud"] JSON字符串
    private LocalDateTime deployTime;
    
    // 元数据
    private String description;
    private String tags;         // ["baseline", "optimized", "production"] JSON字符串
    private LocalDateTime createdTime;
    private String createdBy;
}
