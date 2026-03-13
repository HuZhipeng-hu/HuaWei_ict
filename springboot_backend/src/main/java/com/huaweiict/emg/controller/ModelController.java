package com.huaweiict.emg.controller;

import com.huaweiict.emg.dto.ModelDeployRequest;
import com.huaweiict.emg.entity.ModelVersion;
import com.huaweiict.emg.service.ModelService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 模型管理API控制器
 * 
 * 提供：
 * 1. 查询模型版本列表
 * 2. 部署模型
 * 3. 下载模型文件
 * 4. 激活/停用模型
 */
@Slf4j
@RestController
@RequestMapping("/api/model")
public class ModelController {

    @Autowired
    private ModelService modelService;

    /**
     * 获取所有模型版本
     * GET /api/model/versions?format=onnx&limit=20
     */
    @GetMapping("/versions")
    public Map<String, Object> getVersions(
            @RequestParam(required = false) String format,
            @RequestParam(defaultValue = "20") int limit) {
        
        Map<String, Object> result = new HashMap<>();
        try {
            List<ModelVersion> versions = modelService.listVersions(format, limit);
            result.put("code", 200);
            result.put("data", versions);
        } catch (Exception e) {
            log.error("获取模型版本列表失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 获取当前激活的模型
     * GET /api/model/active
     */
    @GetMapping("/active")
    public Map<String, Object> getActiveModel() {
        Map<String, Object> result = new HashMap<>();
        try {
            ModelVersion activeModel = modelService.getActiveModel();
            if (activeModel == null) {
                result.put("code", 404);
                result.put("message", "没有激活的模型");
            } else {
                result.put("code", 200);
                result.put("data", activeModel);
            }
        } catch (Exception e) {
            log.error("获取激活模型失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 部署模型
     * POST /api/model/deploy
     */
    @PostMapping("/deploy")
    public Map<String, Object> deployModel(@RequestBody ModelDeployRequest request) {
        Map<String, Object> result = new HashMap<>();
        try {
            log.info("收到模型部署请求: version={}, target={}", 
                    request.getVersion(), request.getTargetType());
            
            modelService.deployModel(request);
            
            result.put("code", 200);
            result.put("message", "模型部署成功");
            
            log.info("模型部署成功: {}", request.getVersion());
        } catch (Exception e) {
            log.error("模型部署失败: {}", e.getMessage(), e);
            result.put("code", 500);
            result.put("message", "部署失败: " + e.getMessage());
        }
        return result;
    }

    /**
     * 下载模型文件
     * GET /api/model/download/{version}
     */
    @GetMapping("/download/{version}")
    public ResponseEntity<Resource> downloadModel(@PathVariable String version) {
        try {
            log.info("收到模型下载请求: version={}", version);
            
            String modelPath = modelService.getModelPath(version);
            Resource resource = new FileSystemResource(modelPath);
            
            if (!resource.exists()) {
                log.error("模型文件不存在: {}", modelPath);
                return ResponseEntity.notFound().build();
            }
            
            HttpHeaders headers = new HttpHeaders();
            headers.add(HttpHeaders.CONTENT_DISPOSITION, 
                    "attachment; filename=" + version + "_model.pth");
            
            return ResponseEntity.ok()
                    .headers(headers)
                    .contentLength(resource.contentLength())
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .body(resource);
                    
        } catch (Exception e) {
            log.error("下载模型失败: {}", e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * 激活模型版本
     * POST /api/model/{version}/activate
     */
    @PostMapping("/{version}/activate")
    public Map<String, Object> activateModel(@PathVariable String version) {
        Map<String, Object> result = new HashMap<>();
        try {
            modelService.activateModel(version);
            result.put("code", 200);
            result.put("message", "模型已激活");
        } catch (Exception e) {
            log.error("激活模型失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 删除模型版本
     * DELETE /api/model/{version}
     */
    @DeleteMapping("/{version}")
    public Map<String, Object> deleteModel(@PathVariable String version) {
        Map<String, Object> result = new HashMap<>();
        try {
            modelService.deleteModel(version);
            result.put("code", 200);
            result.put("message", "模型已删除");
        } catch (Exception e) {
            log.error("删除模型失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }

    /**
     * 获取模型详情
     * GET /api/model/{version}/details
     */
    @GetMapping("/{version}/details")
    public Map<String, Object> getModelDetails(@PathVariable String version) {
        Map<String, Object> result = new HashMap<>();
        try {
            ModelVersion model = modelService.getModelByVersion(version);
            if (model == null) {
                result.put("code", 404);
                result.put("message", "模型不存在");
            } else {
                result.put("code", 200);
                result.put("data", model);
            }
        } catch (Exception e) {
            log.error("获取模型详情失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("message", e.getMessage());
        }
        return result;
    }
}
