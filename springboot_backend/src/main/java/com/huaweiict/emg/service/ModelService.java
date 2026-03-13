package com.huaweiict.emg.service;

import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.huaweiict.emg.dto.ModelDeployRequest;
import com.huaweiict.emg.entity.ModelVersion;
import com.huaweiict.emg.entity.TrainingTask;
import com.huaweiict.emg.mapper.ModelVersionMapper;
import com.huaweiict.emg.websocket.EmgWebSocketHandler;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 模型管理服务
 * 
 * 功能：
 * 1. 创建模型版本记录
 * 2. 查询模型版本
 * 3. 部署模型
 * 4. 激活/停用模型
 */
@Slf4j
@Service
public class ModelService {

    @Autowired
    private ModelVersionMapper modelVersionMapper;

    @Autowired
    private EmgWebSocketHandler webSocketHandler;

    @Value("${model.storage.path:/opt/emg/models}")
    private String modelStoragePath;

    /**
     * 创建模型版本记录（从训练任务）
     */
    @Transactional
    public ModelVersion createModelVersion(TrainingTask task, Map<String, Object> trainingResult) {
        log.info("创建模型版本: taskId={}", task.getId());
        
        // 1. 生成版本号
        String version = generateVersionNumber();
        
        // 2. 创建模型版本记录
        ModelVersion model = new ModelVersion();
        model.setVersion(version);
        model.setModelName("NeuroGrip_" + task.getTaskName());
        model.setTrainingTaskId(task.getId());
        
        // 模型文件信息
        model.setModelPath((String) trainingResult.get("model_path"));
        model.setModelFormat("pytorch");  // TODO: 从配置读取
        
        // 计算模型文件大小
        try {
            Path modelFile = Paths.get(model.getModelPath());
            if (Files.exists(modelFile)) {
                long sizeBytes = Files.size(modelFile);
                model.setModelSizeMb(sizeBytes / 1024f / 1024f);
            }
        } catch (Exception e) {
            log.warn("无法读取模型文件大小: {}", e.getMessage());
        }
        
        // 训练信息
        model.setTrainedSamples(task.getTotalSamples());
        model.setGestureClasses(task.getGestureDistribution()); // 手势类别
        
        // 性能指标
        model.setAccuracy(task.getTestAccuracy());
        // TODO: precision, recall, f1_score等从trainingResult读取
        
        // 模型架构
        // TODO: 从训练配置读取
        model.setModelArchitecture("CNN_LSTM");
        
        // 状态
        model.setIsActive(false);  // 新模型默认不激活
        model.setCreatedTime(LocalDateTime.now());
        model.setCreatedBy(task.getCreatedBy());
        
        // 3. 保存记录
        modelVersionMapper.insert(model);
        
        log.info("模型版本创建成功: version={}, accuracy={}", version, model.getAccuracy());
        
        return model;
    }

    /**
     * 部署模型
     */
    public void deployModel(ModelDeployRequest request) {
        log.info("部署模型: version={}, target={}", request.getVersion(), request.getTargetType());
        
        // 1. 查询模型版本
        LambdaQueryWrapper<ModelVersion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ModelVersion::getVersion, request.getVersion());
        ModelVersion model = modelVersionMapper.selectOne(wrapper);
        
        if (model == null) {
            throw new RuntimeException("模型版本不存在: " + request.getVersion());
        }
        
        // 2. 根据目标类型部署
        if ("orangepi".equals(request.getTargetType())) {
            deployToOrangePi(model, request.getTargetDeviceId());
        } else if ("cloud".equals(request.getTargetType())) {
            deployToCloud(model);
        } else {
            throw new RuntimeException("不支持的部署目标: " + request.getTargetType());
        }
        
        // 3. 更新部署状态
        model.setDeployTime(LocalDateTime.now());
        // TODO: 更新deployedTo字段
        
        // 4. 如果设置为激活，则激活此版本
        if (Boolean.TRUE.equals(request.getSetAsActive())) {
            activateModel(request.getVersion());
        }
        
        modelVersionMapper.updateById(model);
        
        // 5. 通知App部署完成
        notifyModelDeployed(model, request);
        
        log.info("模型部署成功: version={}", request.getVersion());
    }

    /**
     * 部署到OrangePi
     */
    private void deployToOrangePi(ModelVersion model, String deviceId) {
        // TODO: 实现推送模型到OrangePi的逻辑
        // 方案1: 通过WebSocket发送模型文件
        // 方案2: OrangePi主动HTTP下载模型
        // 方案3: 通过SCP/SFTP上传
        
        log.info("推送模型到OrangePi: device={}, version={}", deviceId, model.getVersion());
        
        // 示例：通过WebSocket通知OrangePi下载模型
        Map<String, Object> message = new HashMap<>();
        message.put("type", "model_deploy");
        message.put("version", model.getVersion());
        message.put("download_url", "/api/model/download/" + model.getVersion());
        
        // TODO: 发送到特定设备的WebSocket连接
        webSocketHandler.broadcastToApps(JSON.toJSONString(message));
    }

    /**
     * 部署到云端
     */
    private void deployToCloud(ModelVersion model) {
        // TODO: 实现云端部署逻辑
        // 可能需要启动推理服务容器等
        
        log.info("部署模型到云端: version={}", model.getVersion());
    }

    /**
     * 激活模型版本
     */
    @Transactional
    public void activateModel(String version) {
        log.info("激活模型: version={}", version);
        
        // 1. 停用所有当前激活的模型
        LambdaQueryWrapper<ModelVersion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ModelVersion::getIsActive, true);
        List<ModelVersion> activeModels = modelVersionMapper.selectList(wrapper);
        
        for (ModelVersion model : activeModels) {
            model.setIsActive(false);
            modelVersionMapper.updateById(model);
        }
        
        // 2. 激活指定版本
        wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ModelVersion::getVersion, version);
        ModelVersion model = modelVersionMapper.selectOne(wrapper);
        
        if (model == null) {
            throw new RuntimeException("模型版本不存在: " + version);
        }
        
        model.setIsActive(true);
        modelVersionMapper.updateById(model);
        
        log.info("模型激活成功: version={}", version);
    }

    /**
     * 获取激活的模型
     */
    public ModelVersion getActiveModel() {
        return modelVersionMapper.selectActiveModel();
    }

    /**
     * 查询模型版本列表
     */
    public List<ModelVersion> listVersions(String format, int limit) {
        LambdaQueryWrapper<ModelVersion> wrapper = new LambdaQueryWrapper<>();
        
        if (format != null && !format.isEmpty()) {
            wrapper.eq(ModelVersion::getModelFormat, format);
        }
        
        wrapper.orderByDesc(ModelVersion::getCreatedTime)
                .last("LIMIT " + limit);
        
        return modelVersionMapper.selectList(wrapper);
    }

    /**
     * 根据版本号获取模型
     */
    public ModelVersion getModelByVersion(String version) {
        LambdaQueryWrapper<ModelVersion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ModelVersion::getVersion, version);
        return modelVersionMapper.selectOne(wrapper);
    }

    /**
     * 获取模型文件路径
     */
    public String getModelPath(String version) {
        ModelVersion model = getModelByVersion(version);
        if (model == null) {
            throw new RuntimeException("模型版本不存在: " + version);
        }
        return model.getModelPath();
    }

    /**
     * 删除模型
     */
    @Transactional
    public void deleteModel(String version) {
        ModelVersion model = getModelByVersion(version);
        if (model == null) {
            throw new RuntimeException("模型版本不存在: " + version);
        }
        
        if (Boolean.TRUE.equals(model.getIsActive())) {
            throw new RuntimeException("无法删除激活的模型");
        }
        
        // 删除数据库记录
        modelVersionMapper.deleteById(model.getId());
        
        // TODO: 删除模型文件
        
        log.info("模型删除成功: version={}", version);
    }

    /**
     * 通知App模型已部署
     */
    private void notifyModelDeployed(ModelVersion model, ModelDeployRequest request) {
        Map<String, Object> message = new HashMap<>();
        message.put("type", "model_deployed");
        message.put("version", model.getVersion());
        message.put("target_type", request.getTargetType());
        message.put("target_device", request.getTargetDeviceId());
        message.put("status", "success");
        
        webSocketHandler.broadcastToApps(JSON.toJSONString(message));
    }

    /**
     * 生成版本号
     * 格式: v1.0.{序号}
     */
    private String generateVersionNumber() {
        // 查询最新版本
        LambdaQueryWrapper<ModelVersion> wrapper = new LambdaQueryWrapper<>();
        wrapper.orderByDesc(ModelVersion::getId).last("LIMIT 1");
        ModelVersion latest = modelVersionMapper.selectOne(wrapper);
        
        if (latest == null) {
            return "v1.0.0";
        }
        
        // 解析版本号并递增
        String version = latest.getVersion();
        if (version.startsWith("v")) {
            version = version.substring(1);
        }
        
        String[] parts = version.split("\\.");
        if (parts.length == 3) {
            int patch = Integer.parseInt(parts[2]);
            return String.format("v%s.%s.%d", parts[0], parts[1], patch + 1);
        }
        
        return "v1.0.0";
    }
}
