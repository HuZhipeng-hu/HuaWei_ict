package com.huaweiict.emg.dto;

import lombok.Data;

/**
 * 模型部署请求DTO
 */
@Data
public class ModelDeployRequest {
    
    /** 模型版本号 */
    private String version;
    
    /** 部署目标类型: orangepi/cloud/edge */
    private String targetType;
    
    /** 目标设备ID（针对orangepi部署） */
    private String targetDeviceId;
    
    /** 是否激活为生产版本 */
    private Boolean setAsActive = false;
    
    /** 部署方式: auto_download/manual_upload/docker */
    private String deployMethod = "auto_download";
    
    /** 操作人 */
    private String deployedBy;
}
