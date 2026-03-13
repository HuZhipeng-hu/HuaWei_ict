package com.huaweiict.emg.controller;

import com.huaweiict.emg.dto.AnnotationRequest;
import com.huaweiict.emg.service.AnnotationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 数据标注API控制器
 * 
 * 提供：
 * 1. 获取缓存数据（用于预览）
 * 2. 保存标注数据
 * 3. 查询标注统计信息
 */
@Slf4j
@RestController
@RequestMapping("/api/annotation")
public class AnnotationController {

    @Autowired
    private AnnotationService annotationService;

    /**
     * 获取缓存数据（用于预览）
     * GET /api/annotation/cache-data?device_id=orangepi_01&start_time=xxx&end_time=xxx
     */
    @GetMapping("/cache-data")
    public Map<String, Object> getCacheData(
            @RequestParam(required = false, defaultValue = "orangepi_01") String deviceId,
            @RequestParam(required = false) String startTime,
            @RequestParam(required = false) String endTime) {
        
        Map<String, Object> result = new HashMap<>();
        try {
            LocalDateTime start = startTime != null ? LocalDateTime.parse(startTime) : LocalDateTime.now().minusMinutes(1);
            LocalDateTime end = endTime != null ? LocalDateTime.parse(endTime) : LocalDateTime.now();
            
            Map<String, Object> cacheData = annotationService.getCacheData(deviceId, start, end);
            result.put("code", 200);
            result.put("data", cacheData);
        } catch (Exception e) {
            log.error("获取缓存数据失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 保存标注数据
     * POST /api/annotation/save
     * Body: { deviceId, startTime, endTime, gestureLabel, annotator }
     */
    @PostMapping("/save")
    public Map<String, Object> saveAnnotation(@RequestBody AnnotationRequest request) {
        Map<String, Object> result = new HashMap<>();
        try {
            log.info("收到标注请求: 设备={}, 手势={}, 时间={}~{}", 
                    request.getDeviceId(), request.getGestureLabel(), 
                    request.getStartTime(), request.getEndTime());
            
            Map<String, Object> saveResult = annotationService.saveAnnotation(request);
            
            result.put("code", 200);
            result.put("message", "标注保存成功");
            result.put("data", saveResult);
            
            log.info("标注保存成功: 保存了 {} 帧数据", saveResult.get("saved_count"));
        } catch (Exception e) {
            log.error("保存标注失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", "保存失败: " + e.getMessage());
        }
        return result;
    }

    /**
     * 查询标注数据统计
     * GET /api/annotation/statistics
     */
    @GetMapping("/statistics")
    public Map<String, Object> getStatistics() {
        Map<String, Object> result = new HashMap<>();
        try {
            Map<String, Object> stats = annotationService.getAnnotationStatistics();
            result.put("code", 200);
            result.put("data", stats);
        } catch (Exception e) {
            log.error("获取统计信息失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 删除标注数据
     * DELETE /api/annotation/{id}
     */
    @DeleteMapping("/{id}")
    public Map<String, Object> deleteAnnotation(@PathVariable Long id) {
        Map<String, Object> result = new HashMap<>();
        try {
            annotationService.deleteAnnotation(id);
            result.put("code", 200);
            result.put("message", "删除成功");
        } catch (Exception e) {
            log.error("删除标注失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 查询标注历史
     * GET /api/annotation/history?gesture=fist&limit=100
     */
    @GetMapping("/history")
    public Map<String, Object> getHistory(
            @RequestParam(required = false) String gesture,
            @RequestParam(required = false) String annotator,
            @RequestParam(defaultValue = "100") int limit) {
        
        Map<String, Object> result = new HashMap<>();
        try {
            result.put("code", 200);
            result.put("data", annotationService.getHistory(gesture, annotator, limit));
        } catch (Exception e) {
            log.error("查询历史失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }
}
