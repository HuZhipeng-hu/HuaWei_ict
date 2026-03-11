package com.huaweiict.emg.dto;

import lombok.Data;
import java.util.List;

/**
 * OrangePi 批量上报请求体
 */
@Data
public class EmgBatchRequest {
    /** 设备标识 */
    private String deviceId;
    /** 帧数据列表 */
    private List<EmgFrameDTO> frames;
    /** 帧数量 */
    private Integer count;
    /** 上报时间 */
    private String uploadTime;
}
