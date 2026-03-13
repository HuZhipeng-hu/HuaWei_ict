package com.huaweiict.emg.dto;

import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * 训练任务创建请求DTO
 */
@Data
public class TrainingTaskCreateRequest {
    
    /** 任务名称 */
    private String taskName;
    
    /** 训练配置参数 */
    private TrainingConfig config;
    
    /** 数据筛选条件 */
    private DataFilter dataFilter;
    
    /** 创建者 */
    private String createdBy;
    
    /**
     * 训练配置
     */
    @Data
    public static class TrainingConfig {
        private Integer epochs = 50;
        private Integer batchSize = 32;
        private Float learningRate = 0.001f;
        private Integer windowSize = 150;
        private String modelType = "cnn_lstm";  // cnn/lstm/cnn_lstm
        private String optimizer = "adam";
        private Float valRatio = 0.2f;
        private Float testRatio = 0.1f;
    }
    
    /**
     * 数据筛选条件
     */
    @Data
    public static class DataFilter {
        private List<String> gestures;      // 要训练的手势列表
        private Float minQualityScore = 0.8f;
        private String dateFrom;            // 日期范围
        private String dateTo;
        private List<String> deviceIds;     // 设备ID过滤
        private List<String> annotators;    // 标注人过滤
    }
}
