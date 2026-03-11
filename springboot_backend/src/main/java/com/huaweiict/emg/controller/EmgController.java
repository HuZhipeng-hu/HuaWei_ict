package com.huaweiict.emg.controller;

import com.huaweiict.emg.dto.EmgBatchRequest;
import com.huaweiict.emg.dto.EmgFrameDTO;
import com.huaweiict.emg.entity.EmgFrame;
import com.huaweiict.emg.entity.GestureEvent;
import com.huaweiict.emg.service.EmgDataService;
import com.huaweiict.emg.websocket.EmgWebSocketHandler;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * EMG 数据 REST API 控制器
 *
 * 提供给：
 * 1. OrangePi - HTTP 批量上报（备用通道，WebSocket 不可用时使用）
 * 2. HarmonyOS App - REST 查询接口
 */
@Slf4j
@RestController
@RequestMapping("/api/emg")
public class EmgController {

    @Autowired
    private EmgDataService emgDataService;

    @Autowired
    private EmgWebSocketHandler webSocketHandler;

    // ======================== OrangePi 上报接口 ========================

    /**
     * 批量上报 EMG 数据
     * POST /api/emg/batch
     *
     * 供 OrangePi 通过 HTTP 批量上传（当 WebSocket 不稳定时的备用通道）
     */
    @PostMapping("/batch")
    public Map<String, Object> batchUpload(@RequestBody EmgBatchRequest request) {
        Map<String, Object> result = new HashMap<>();
        try {
            emgDataService.processBatch(request.getDeviceId(), request.getFrames());
            result.put("code", 200);
            result.put("msg", "ok");
            result.put("received", request.getFrames().size());
        } catch (Exception e) {
            log.error("批量上报失败: {}", e.getMessage());
            result.put("code", 500);
            result.put("msg", e.getMessage());
        }
        return result;
    }

    /**
     * 单帧上报
     * POST /api/emg/frame
     */
    @PostMapping("/frame")
    public Map<String, Object> singleFrame(
            @RequestParam(defaultValue = "default") String deviceId,
            @RequestBody EmgFrameDTO dto) {
        Map<String, Object> result = new HashMap<>();
        try {
            emgDataService.processFrame(deviceId, dto);
            result.put("code", 200);
            result.put("msg", "ok");
        } catch (Exception e) {
            result.put("code", 500);
            result.put("msg", e.getMessage());
        }
        return result;
    }

    // ======================== App 查询接口 ========================

    /**
     * 获取最新一帧数据
     * GET /api/emg/latest
     *
     * App 可以轮询此接口获取最新数据（WebSocket 的备用方案）
     */
    @GetMapping("/latest")
    public Map<String, Object> getLatest() {
        return emgDataService.getLatestFrame();
    }

    /**
     * 查询历史数据
     * GET /api/emg/history?deviceId=orangepi_01&limit=100
     */
    @GetMapping("/history")
    public List<EmgFrame> getHistory(
            @RequestParam(required = false) String deviceId,
            @RequestParam(defaultValue = "100") int limit) {
        return emgDataService.getHistory(deviceId, limit);
    }

    /**
     * 查询手势事件
     * GET /api/emg/gestures?deviceId=orangepi_01&limit=50
     */
    @GetMapping("/gestures")
    public List<GestureEvent> getGestures(
            @RequestParam(required = false) String deviceId,
            @RequestParam(defaultValue = "50") int limit) {
        return emgDataService.getGestureEvents(deviceId, limit);
    }

    /**
     * 服务器状态
     * GET /api/emg/status
     */
    @GetMapping("/status")
    public Map<String, Object> status() {
        Map<String, Object> status = new HashMap<>();
        status.put("running", true);
        status.put("connections", webSocketHandler.getConnectionStats());
        status.put("serverTime", System.currentTimeMillis());
        return status;
    }
}
