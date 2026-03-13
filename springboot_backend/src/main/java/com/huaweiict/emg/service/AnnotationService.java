package com.huaweiict.emg.service;

import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.huaweiict.emg.dto.AnnotationRequest;
import com.huaweiict.emg.entity.AnnotationSession;
import com.huaweiict.emg.entity.EmgLabeledData;
import com.huaweiict.emg.mapper.EmgLabeledDataMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 标注服务实现
 * 
 * 功能：
 * 1. 从缓存中获取EMG数据
 * 2. 保存标注数据到MySQL
 * 3. 统计标注信息
 */
@Slf4j
@Service
public class AnnotationService {

    @Autowired
    private EmgLabeledDataMapper labeledDataMapper;

    @Autowired
    private EmgDataService emgDataService;  // 用于访问缓存

    /**
     * 从缓存获取指定时间范围的数据（用于预览）
     */
    public Map<String, Object> getCacheData(String deviceId, LocalDateTime startTime, LocalDateTime endTime) {
        // 从EmgDataService的缓冲区获取数据
        // TODO: 实现从Redis或内存缓冲区获取数据
        List<Map<String, Object>> frames = emgDataService.getCachedFrames(deviceId, startTime, endTime);
        
        Map<String, Object> result = new HashMap<>();
        result.put("frames", frames);
        result.put("count", frames.size());
        result.put("start_time", startTime);
        result.put("end_time", endTime);
        
        log.debug("从缓存获取数据: 设备={}, 帧数={}", deviceId, frames.size());
        return result;
    }

    /**
     * 保存标注数据
     */
    @Transactional
    public Map<String, Object> saveAnnotation(AnnotationRequest request) {
        // 1. 从缓存获取数据
        List<Map<String, Object>> frames = emgDataService.getCachedFrames(
                request.getDeviceId(), 
                request.getStartTime(), 
                request.getEndTime()
        );
        
        if (frames.isEmpty()) {
            throw new RuntimeException("缓存中没有找到该时间段的数据，可能已过期");
        }
        
        // 2. 转换为标注数据实体并保存
        List<EmgLabeledData> labeledDataList = new ArrayList<>();
        for (Map<String, Object> frame : frames) {
            EmgLabeledData data = new EmgLabeledData();
            data.setDeviceId(request.getDeviceId());
            data.setCaptureTime(request.getStartTime()); // TODO: 使用实际帧时间
            data.setDeviceTs((Long) frame.get("device_ts"));
            
            // 保存JSON数据
            data.setEmgData(JSON.toJSONString(frame.get("emg")));
            data.setAccData(JSON.toJSONString(frame.get("acc")));
            data.setGyroData(JSON.toJSONString(frame.get("gyro")));
            data.setAngleData(JSON.toJSONString(frame.get("angle")));
            data.setBattery((Integer) frame.get("battery"));
            
            // 标注信息
            data.setGestureLabel(request.getGestureLabel());
            data.setAnnotator(request.getAnnotator());
            data.setAnnotationTime(LocalDateTime.now());
            data.setAnnotationNote(request.getAnnotationNote());
            
            // 计算质量分数（如果需要）
            if (Boolean.TRUE.equals(request.getCalculateQuality())) {
                data.setQualityScore(calculateQualityScore(frame));
            }
            
            data.setIsValid(true);
            data.setIsUsedForTraining(false);
            
            labeledDataList.add(data);
        }
        
        // 3. 批量插入
        int savedCount = 0;
        for (EmgLabeledData data : labeledDataList) {
            labeledDataMapper.insert(data);
            savedCount++;
        }
        
        log.info("标注保存成功: 手势={}, 帧数={}", request.getGestureLabel(), savedCount);
        
        // 4. 返回结果
        Map<String, Object> result = new HashMap<>();
        result.put("saved_count", savedCount);
        result.put("gesture_label", request.getGestureLabel());
        result.put("annotator", request.getAnnotator());
        
        return result;
    }

    /**
     * 获取标注统计信息
     */
    public Map<String, Object> getAnnotationStatistics() {
        // 1. 使用视图查询统计信息
        List<Map<String, Object>> stats = labeledDataMapper.getAnnotationStatistics();
        
        // 2. 总计
        int totalSamples = 0;
        Map<String, Integer> gestureDistribution = new HashMap<>();
        
        for (Map<String, Object> stat : stats) {
            String gesture = (String) stat.get("gesture_label");
            Long count = (Long) stat.get("sample_count");
            gestureDistribution.put(gesture, count.intValue());
            totalSamples += count;
        }
        
        // 3. 获取时间范围
        LambdaQueryWrapper<EmgLabeledData> wrapper = new LambdaQueryWrapper<>();
        wrapper.select(EmgLabeledData::getCaptureTime)
                .orderByAsc(EmgLabeledData::getCaptureTime)
                .last("LIMIT 1");
        EmgLabeledData earliest = labeledDataMapper.selectOne(wrapper);
        
        wrapper = new LambdaQueryWrapper<>();
        wrapper.select(EmgLabeledData::getCaptureTime)
                .orderByDesc(EmgLabeledData::getCaptureTime)
                .last("LIMIT 1");
        EmgLabeledData latest = labeledDataMapper.selectOne(wrapper);
        
        // 4. 组装结果
        Map<String, Object> result = new HashMap<>();
        result.put("total_samples", totalSamples);
        result.put("gesture_distribution", gestureDistribution);
        result.put("date_range", Map.of(
                "start", earliest != null ? earliest.getCaptureTime() : null,
                "end", latest != null ? latest.getCaptureTime() : null
        ));
        result.put("gesture_count", gestureDistribution.size());
        
        return result;
    }

    /**
     * 删除标注数据
     */
    public void deleteAnnotation(Long id) {
        labeledDataMapper.deleteById(id);
        log.info("删除标注数据: id={}", id);
    }

    /**
     * 查询标注历史
     */
    public List<EmgLabeledData> getHistory(String gesture, String annotator, int limit) {
        LambdaQueryWrapper<EmgLabeledData> wrapper = new LambdaQueryWrapper<>();
        
        if (gesture != null && !gesture.isEmpty()) {
            wrapper.eq(EmgLabeledData::getGestureLabel, gesture);
        }
        if (annotator != null && !annotator.isEmpty()) {
            wrapper.eq(EmgLabeledData::getAnnotator, annotator);
        }
        
        wrapper.orderByDesc(EmgLabeledData::getAnnotationTime)
                .last("LIMIT " + limit);
        
        return labeledDataMapper.selectList(wrapper);
    }

    /**
     * 计算数据质量分数
     * 基于信号强度、变异性等指标
     */
    private Float calculateQualityScore(Map<String, Object> frame) {
        try {
            // TODO: 实现质量评估算法
            // 考虑因素：
            // 1. 信号强度（EMG幅值）
            // 2. 信噪比
            // 3. 信号稳定性
            // 4. 是否有明显的伪迹
            
            // 简单实现：基于EMG数据的标准差
            Object emgObj = frame.get("emg");
            if (emgObj instanceof List) {
                @SuppressWarnings("unchecked")
                List<List<Integer>> emg = (List<List<Integer>>) emgObj;
                
                // 计算所有通道的平均值
                double sum = 0;
                int count = 0;
                for (List<Integer> row : emg) {
                    for (Integer val : row) {
                        sum += val;
                        count++;
                    }
                }
                double mean = sum / count;
                
                // 如果平均值在合理范围内，给予较高分数
                if (mean > 20 && mean < 200) {
                    return 0.9f;
                } else if (mean > 10 && mean < 250) {
                    return 0.7f;
                } else {
                    return 0.5f;
                }
            }
            
            return 0.8f;  // 默认分数
        } catch (Exception e) {
            log.warn("计算质量分数失败: {}", e.getMessage());
            return 0.5f;
        }
    }
}
