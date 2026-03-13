package com.huaweiict.emg.controller;

import com.huaweiict.emg.dto.TrainingTaskCreateRequest;
import com.huaweiict.emg.entity.TrainingTask;
import com.huaweiict.emg.service.TrainingService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 训练任务API控制器
 * 
 * 提供：
 * 1. 创建训练任务
 * 2. 查询任务状态
 * 3. 获取训练日志
 * 4. 获取训练结果
 */
@Slf4j
@RestController
@RequestMapping("/api/training")
public class TrainingController {

    @Autowired
    private TrainingService trainingService;

    /**
     * 创建训练任务
     * POST /api/training/create
     */
    @PostMapping("/create")
    public Map<String, Object> createTask(@RequestBody TrainingTaskCreateRequest request) {
        Map<String, Object> result = new HashMap<>();
        try {
            log.info("收到训练任务创建请求: {}", request.getTaskName());
            
            Long taskId = trainingService.createTask(request);
            
            result.put("code", 200);
            result.put("message", "训练任务创建成功");
            result.put("data", Map.of(
                "task_id", taskId,
                "estimated_time_minutes", 30  // TODO: 根据数据量估算
            ));
            
            log.info("训练任务创建成功: taskId={}", taskId);
        } catch (Exception e) {
            log.error("创建训练任务失败: {}", e.getMessage(), e);
            result.put("code", 500);
            result.put("message", "创建失败: " + e.getMessage());
        }
        return result;
    }

    /**
     * 查询训练任务状态
     * GET /api/training/task/{taskId}
     */
    @GetMapping("/task/{taskId}")
    public Map<String, Object> getTaskStatus(@PathVariable Long taskId) {
        Map<String, Object> result = new HashMap<>();
        try {
            TrainingTask task = trainingService.getTask(taskId);
            if (task == null) {
                result.put("code", 404);
                result.put("message", "任务不存在");
                return result;
            }
            
            Map<String, Object> data = new HashMap<>();
            data.put("task_id", task.getId());
            data.put("task_name", task.getTaskName());
            data.put("status", task.getTaskStatus());
            data.put("progress", task.getProgressPercent());
            data.put("current_epoch", task.getCurrentEpoch());
            data.put("total_epochs", task.getTotalEpochs());
            data.put("train_accuracy", task.getTrainAccuracy());
            data.put("val_accuracy", task.getValAccuracy());
            data.put("train_loss", task.getFinalTrainLoss());
            data.put("val_loss", task.getFinalValLoss());
            data.put("created_time", task.getCreatedTime());
            data.put("start_time", task.getStartTime());
            
            // 计算剩余时间
            if ("running".equals(task.getTaskStatus()) && task.getCurrentEpoch() != null && task.getTotalEpochs() != null) {
                int elapsed = task.getDurationSeconds() != null ? task.getDurationSeconds() : 0;
                if (task.getCurrentEpoch() > 0) {
                    int avgTimePerEpoch = elapsed / task.getCurrentEpoch();
                    int remaining = avgTimePerEpoch * (task.getTotalEpochs() - task.getCurrentEpoch());
                    data.put("estimated_remaining_seconds", remaining);
                }
            }
            
            result.put("code", 200);
            result.put("data", data);
        } catch (Exception e) {
            log.error("查询任务状态失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 获取训练日志
     * GET /api/training/task/{taskId}/logs?lines=100
     */
    @GetMapping("/task/{taskId}/logs")
    public Map<String, Object> getTaskLogs(
            @PathVariable Long taskId,
            @RequestParam(defaultValue = "100") int lines) {
        
        Map<String, Object> result = new HashMap<>();
        try {
            List<String> logs = trainingService.getTaskLogs(taskId, lines);
            result.put("code", 200);
            result.put("data", Map.of("logs", logs));
        } catch (Exception e) {
            log.error("获取训练日志失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 获取训练结果
     * GET /api/training/task/{taskId}/result
     */
    @GetMapping("/task/{taskId}/result")
    public Map<String, Object> getTaskResult(@PathVariable Long taskId) {
        Map<String, Object> result = new HashMap<>();
        try {
            Map<String, Object> taskResult = trainingService.getTaskResult(taskId);
            result.put("code", 200);
            result.put("data", taskResult);
        } catch (Exception e) {
            log.error("获取训练结果失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 获取所有训练任务列表
     * GET /api/training/tasks?status=completed&limit=20
     */
    @GetMapping("/tasks")
    public Map<String, Object> listTasks(
            @RequestParam(required = false) String status,
            @RequestParam(defaultValue = "20") int limit) {
        
        Map<String, Object> result = new HashMap<>();
        try {
            List<TrainingTask> tasks = trainingService.listTasks(status, limit);
            result.put("code", 200);
            result.put("data", tasks);
        } catch (Exception e) {
            log.error("获取任务列表失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 取消训练任务
     * POST /api/training/task/{taskId}/cancel
     */
    @PostMapping("/task/{taskId}/cancel")
    public Map<String, Object> cancelTask(@PathVariable Long taskId) {
        Map<String, Object> result = new HashMap<>();
        try {
            trainingService.cancelTask(taskId);
            result.put("code", 200);
            result.put("message", "任务已取消");
        } catch (Exception e) {
            log.error("取消任务失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }
}
