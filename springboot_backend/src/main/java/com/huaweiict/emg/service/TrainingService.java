package com.huaweiict.emg.service;

import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.huaweiict.emg.dto.TrainingTaskCreateRequest;
import com.huaweiict.emg.entity.EmgLabeledData;
import com.huaweiict.emg.entity.TrainingTask;
import com.huaweiict.emg.mapper.EmgLabeledDataMapper;
import com.huaweiict.emg.mapper.TrainingTaskMapper;
import com.huaweiict.emg.websocket.EmgWebSocketHandler;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 训练服务实现
 * 
 * 功能：
 * 1. 创建训练任务
 * 2. 导出标注数据
 * 3. 执行Python训练脚本
 * 4. 更新训练状态
 * 5. 保存训练结果
 */
@Slf4j
@Service
public class TrainingService {

    @Autowired
    private TrainingTaskMapper taskMapper;

    @Autowired
    private EmgLabeledDataMapper labeledDataMapper;

    @Autowired
    private ModelService modelService;
    
    @Autowired
    private EmgWebSocketHandler webSocketHandler;

    @Value("${training.workspace:/opt/emg/training}")
    private String trainingWorkspace;

    @Value("${training.python.executable:python}")
    private String pythonExecutable;

    @Value("${training.script.path:/opt/emg/scripts/training_server.py}")
    private String trainingScriptPath;

    /**
     * 创建训练任务
     */
    @Transactional
    public Long createTask(TrainingTaskCreateRequest request) {
        // 1. 创建任务记录
        TrainingTask task = new TrainingTask();
        task.setTaskName(request.getTaskName());
        task.setTaskStatus("pending");
        task.setCreatedBy(request.getCreatedBy());
        task.setCreatedTime(LocalDateTime.now());
        
        // 2. 保存配置
        task.setConfig(JSON.toJSONString(request.getConfig()));
        task.setDataFilter(JSON.toJSONString(request.getDataFilter()));
        
        if (request.getConfig() != null) {
            task.setTotalEpochs(request.getConfig().getEpochs());
        }
        
        // 3. 统计数据集
        Map<String, Object> datasetStats = calculateDatasetStats(request.getDataFilter());
        task.setTotalSamples((Integer) datasetStats.get("total_samples"));
        task.setGestureDistribution(JSON.toJSONString(datasetStats.get("gesture_distribution")));
        
        // 4. 保存任务
        taskMapper.insert(task);
        
        log.info("训练任务创建成功: taskId={}, 样本数={}", task.getId(), task.getTotalSamples());
        
        // 5. 异步执行训练
        executeTrainingAsync(task.getId());
        
        return task.getId();
    }

    /**
     * 执行训练任务（异步）
     */
    @Async
    public void executeTrainingAsync(Long taskId) {
        log.info("开始执行训练任务: taskId={}", taskId);
        
        try {
            // 1. 更新状态为running
            TrainingTask task = taskMapper.selectById(taskId);
            task.setTaskStatus("running");
            task.setStartTime(LocalDateTime.now());
            taskMapper.updateById(task);
            
            // 2. 导出训练数据
            String dataPath = exportTrainingData(task);
            
            // 3. 准备训练配置文件
            String configPath = prepareTrainingConfig(task);
            
            // 4. 执行Python训练脚本
            executeTrainingProcess(task, dataPath, configPath);
            
            // 5. 读取训练结果
            processTrainingResults(task);
            
            // 6. 更新状态为completed
            task.setTaskStatus("completed");
            task.setEndTime(LocalDateTime.now());
            task.setDurationSeconds((int) ChronoUnit.SECONDS.between(task.getStartTime(), task.getEndTime()));
            taskMapper.updateById(task);
            
            // 7. 通知App训练完成
            notifyTrainingCompleted(task);
            
            log.info("训练任务完成: taskId={}", taskId);
            
        } catch (Exception e) {
            log.error("训练任务失败: taskId={}, error={}", taskId, e.getMessage(), e);
            
            // 更新状态为failed
            TrainingTask task = taskMapper.selectById(taskId);
            task.setTaskStatus("failed");
            task.setErrorMessage(e.getMessage());
            task.setEndTime(LocalDateTime.now());
            taskMapper.updateById(task);
            
            // 通知App训练失败
            notifyTrainingFailed(task, e.getMessage());
        }
    }

    /**
     * 导出训练数据到CSV
     */
    private String exportTrainingData(TrainingTask task) throws IOException {
        log.info("导出训练数据: taskId={}", task.getId());
        
        // 1. 创建输出目录
        Path taskDir = Paths.get(trainingWorkspace, "task_" + task.getId());
        Files.createDirectories(taskDir);
        
        // 2. 根据dataFilter查询标注数据
        LambdaQueryWrapper<EmgLabeledData> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EmgLabeledData::getIsValid, true);
        
        // TODO: 添加更多过滤条件（根据task.getDataFilter()）
        
        List<EmgLabeledData> data = labeledDataMapper.selectList(wrapper);
        
        // 3. 按手势分组导出CSV文件
        Map<String, List<EmgLabeledData>> grouped = data.stream()
                .collect(Collectors.groupingBy(EmgLabeledData::getGestureLabel));
        
        for (Map.Entry<String, List<EmgLabeledData>> entry : grouped.entrySet()) {
            String gesture = entry.getKey();
            Path gestureDir = taskDir.resolve(gesture);
            Files.createDirectories(gestureDir);
            
            // 为每个样本创建一个CSV文件
            int index = 0;
            for (EmgLabeledData sample : entry.getValue()) {
                Path csvFile = gestureDir.resolve(sample.getId() + ".csv");
                exportSampleToCsv(sample, csvFile);
                index++;
            }
            
            log.info("导出手势 {} 的数据: {} 个样本", gesture, index);
        }
        
        return taskDir.toString();
    }

    /**
     * 导出单个样本到CSV
     */
    private void exportSampleToCsv(EmgLabeledData sample, Path csvFile) throws IOException {
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(csvFile))) {
            // CSV表头
            writer.println("timestamp,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z");
            
            // EMG数据（假设是 10x8 数组）
            Object emgData = sample.getEmgData();
            // TODO: 解析JSON数据并写入CSV
            
            // 简化示例：写入一行数据
            writer.println(String.format("%d,0,0,0,0,0,0,0,0,0,0,0,0,0,0", 
                    sample.getDeviceTs()));
        }
    }

    /**
     * 准备训练配置文件
     */
    private String prepareTrainingConfig(TrainingTask task) throws IOException {
        Path configPath = Paths.get(trainingWorkspace, "task_" + task.getId(), "config.json");
        
        Map<String, Object> config = new HashMap<>();
        config.put("task_id", task.getId());
        config.put("task_name", task.getTaskName());
        config.put("epochs", task.getTotalEpochs());
        // TODO: 添加更多配置参数
        
        String json = JSON.toJSONString(config);
        Files.writeString(configPath, json);
        
        return configPath.toString();
    }

    /**
     * 执行训练进程
     */
    private void executeTrainingProcess(TrainingTask task, String dataPath, String configPath) 
            throws IOException, InterruptedException {
        
        log.info("启动训练进程: taskId={}", task.getId());
        
        // 构建命令
        List<String> command = Arrays.asList(
                pythonExecutable,
                trainingScriptPath,
                "--data", dataPath,
                "--config", configPath,
                "--task-id", String.valueOf(task.getId())
        );
        
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File(trainingWorkspace));
        pb.redirectErrorStream(true);
        
        // 日志文件
        Path logFile = Paths.get(trainingWorkspace, "task_" + task.getId(), "training.log");
        task.setLogFile(logFile.toString());
        taskMapper.updateById(task);
        
        pb.redirectOutput(ProcessBuilder.Redirect.to(logFile.toFile()));
        
        // 启动进程
        Process process = pb.start();
        
        // 等待进程完成
        int exitCode = process.waitFor();
        
        if (exitCode != 0) {
            throw new RuntimeException("训练进程异常退出，退出码: " + exitCode);
        }
        
        log.info("训练进程完成: taskId={}", task.getId());
    }

    /**
     * 处理训练结果
     */
    private void processTrainingResults(TrainingTask task) throws IOException {
        // 读取结果文件
        Path resultFile = Paths.get(trainingWorkspace, "task_" + task.getId(), "result.json");
        
        if (!Files.exists(resultFile)) {
            throw new RuntimeException("训练结果文件不存在");
        }
        
        String resultJson = Files.readString(resultFile);
        Map<String, Object> result = JSON.parseObject(resultJson, Map.class);
        
        // 更新任务记录
        task.setTestAccuracy(((Number) result.get("test_accuracy")).floatValue());
        task.setTrainAccuracy(((Number) result.get("train_accuracy")).floatValue());
        task.setValAccuracy(((Number) result.get("val_accuracy")).floatValue());
        task.setModelPath((String) result.get("model_path"));
        // TODO: 更新更多结果字段
        
        taskMapper.updateById(task);
        
        // 创建模型版本记录
        modelService.createModelVersion(task, result);
    }

    /**
     * 通知App训练完成
     */
    private void notifyTrainingCompleted(TrainingTask task) {
        Map<String, Object> message = new HashMap<>();
        message.put("type", "training_completed");
        message.put("task_id", task.getId());
        message.put("task_name", task.getTaskName());
        message.put("accuracy", task.getTestAccuracy());
        
        webSocketHandler.broadcastToApps(JSON.toJSONString(message));
    }

    /**
     * 通知App训练失败
     */
    private void notifyTrainingFailed(TrainingTask task, String error) {
        Map<String, Object> message = new HashMap<>();
        message.put("type", "training_failed");
        message.put("task_id", task.getId());
        message.put("error", error);
        
        webSocketHandler.broadcastToApps(JSON.toJSONString(message));
    }

    /**
     * 计算数据集统计信息
     */
    private Map<String, Object> calculateDatasetStats(TrainingTaskCreateRequest.DataFilter filter) {
        // 查询符合条件的数据
        LambdaQueryWrapper<EmgLabeledData> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EmgLabeledData::getIsValid, true);
        
        if (filter != null && filter.getGestures() != null && !filter.getGestures().isEmpty()) {
            wrapper.in(EmgLabeledData::getGestureLabel, filter.getGestures());
        }
        
        List<EmgLabeledData> data = labeledDataMapper.selectList(wrapper);
        
        // 统计手势分布
        Map<String, Long> distribution = data.stream()
                .collect(Collectors.groupingBy(EmgLabeledData::getGestureLabel, Collectors.counting()));
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("total_samples", data.size());
        stats.put("gesture_distribution", distribution);
        
        return stats;
    }

    // ============= 查询接口 =============

    public TrainingTask getTask(Long taskId) {
        return taskMapper.selectById(taskId);
    }

    public List<TrainingTask> listTasks(String status, int limit) {
        LambdaQueryWrapper<TrainingTask> wrapper = new LambdaQueryWrapper<>();
        
        if (status != null && !status.isEmpty()) {
            wrapper.eq(TrainingTask::getTaskStatus, status);
        }
        
        wrapper.orderByDesc(TrainingTask::getCreatedTime)
                .last("LIMIT " + limit);
        
        return taskMapper.selectList(wrapper);
    }

    public List<String> getTaskLogs(Long taskId, int lines) throws IOException {
        TrainingTask task = taskMapper.selectById(taskId);
        if (task == null || task.getLogFile() == null) {
            return Collections.emptyList();
        }
        
        Path logFile = Paths.get(task.getLogFile());
        if (!Files.exists(logFile)) {
            return Collections.emptyList();
        }
        
        List<String> allLines = Files.readAllLines(logFile);
        
        // 返回最后N行
        int start = Math.max(0, allLines.size() - lines);
        return allLines.subList(start, allLines.size());
    }

    public Map<String, Object> getTaskResult(Long taskId) {
        TrainingTask task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new RuntimeException("任务不存在");
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("task_id", task.getId());
        result.put("task_name", task.getTaskName());
        result.put("status", task.getTaskStatus());
        result.put("accuracy", task.getTestAccuracy());
        result.put("train_accuracy", task.getTrainAccuracy());
        result.put("val_accuracy", task.getValAccuracy());
        result.put("model_path", task.getModelPath());
        
       // TODO: 添加更多结果信息（混淆矩阵、训练曲线等）
        
        return result;
    }

    public void cancelTask(Long taskId) {
        TrainingTask task = taskMapper.selectById(taskId);
        if (task != null && "running".equals(task.getTaskStatus())) {
            task.setTaskStatus("cancelled");
            task.setEndTime(LocalDateTime.now());
            taskMapper.updateById(task);
            
            log.info("取消训练任务: taskId={}", taskId);
        }
    }
}
