package com.huaweiict.emg.service;

import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.huaweiict.emg.dto.EmgFrameDTO;
import com.huaweiict.emg.entity.EmgFrame;
import com.huaweiict.emg.entity.GestureEvent;
import com.huaweiict.emg.mapper.EmgFrameMapper;
import com.huaweiict.emg.mapper.GestureEventMapper;
import com.huaweiict.emg.websocket.EmgWebSocketHandler;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * EMG 数据服务
 * - 接收 OrangePi 上报的数据
 * - 批量写入 RDS MySQL
 * - 通过 WebSocket 推送至 HarmonyOS App
 */
@Slf4j
@Service
public class EmgDataService extends ServiceImpl<EmgFrameMapper, EmgFrame> {

    @Autowired
    private EmgFrameMapper emgFrameMapper;

    @Autowired
    private GestureEventMapper gestureEventMapper;

    @Autowired
    private EmgWebSocketHandler webSocketHandler;

    /** 写入缓冲队列 */
    private final ConcurrentLinkedQueue<EmgFrame> writeBuffer = new ConcurrentLinkedQueue<>();

    /** 上一次检测到的手势（用于事件去重） */
    private volatile String lastGesture = "";

    /** 最新一帧数据缓存（供 REST API 查询） */
    private volatile Map<String, Object> latestFrame = new HashMap<>();

    // ======================== 数据接收 ========================

    /**
     * 处理单帧数据（来自 WebSocket 或 HTTP）
     */
    public void processFrame(String deviceId, EmgFrameDTO dto) {
        // 1. 构建实体
        EmgFrame frame = convertToEntity(deviceId, dto);

        // 2. 加入写入缓冲（异步批量写入）
        writeBuffer.add(frame);

        // 3. 更新最新帧缓存
        Map<String, Object> frameMap = new HashMap<>();
        frameMap.put("deviceId", deviceId);
        frameMap.put("deviceTs", dto.getDevice_ts());
        frameMap.put("emg", dto.getEmg());
        frameMap.put("acc", dto.getAcc());
        frameMap.put("gyro", dto.getGyro());
        frameMap.put("angle", dto.getAngle());
        frameMap.put("battery", dto.getBattery());
        frameMap.put("gesture", dto.getGesture());
        frameMap.put("confidence", dto.getConfidence());
        frameMap.put("serverTime", LocalDateTime.now().toString());
        latestFrame = frameMap;

        // 4. 实时推送至所有连接的 App 客户端
        webSocketHandler.broadcastToApps(JSON.toJSONString(frameMap));

        // 5. 手势事件检测
        if (dto.getGesture() != null && !dto.getGesture().equals(lastGesture)
                && !"unknown".equals(dto.getGesture())) {
            saveGestureEvent(deviceId, dto.getGesture(), dto.getConfidence());
            lastGesture = dto.getGesture();
        }
    }

    /**
     * 批量处理帧数据（HTTP batch 接口）
     */
    public void processBatch(String deviceId, List<EmgFrameDTO> frames) {
        for (EmgFrameDTO dto : frames) {
            processFrame(deviceId, dto);
        }
    }

    // ======================== 定时批量写入 ========================

    /**
     * 每 500ms 将缓冲区数据批量写入 MySQL
     */
    @Scheduled(fixedRate = 500)
    public void flushToDatabase() {
        List<EmgFrame> batch = new ArrayList<>();
        EmgFrame frame;
        while ((frame = writeBuffer.poll()) != null) {
            batch.add(frame);
            if (batch.size() >= 100) break; // 每次最多写 100 条
        }

        if (!batch.isEmpty()) {
            try {
                saveBatch(batch, batch.size());
                log.debug("批量写入 {} 帧到 MySQL", batch.size());
            } catch (Exception e) {
                log.error("MySQL 批量写入失败: {}", e.getMessage());
                // 写入失败放回队列
                writeBuffer.addAll(batch);
            }
        }
    }

    // ======================== 手势事件 ========================

    private void saveGestureEvent(String deviceId, String gesture, Float confidence) {
        GestureEvent event = new GestureEvent();
        event.setDeviceId(deviceId);
        event.setGesture(gesture);
        event.setConfidence(confidence);
        event.setEventTime(LocalDateTime.now());
        try {
            gestureEventMapper.insert(event);
        } catch (Exception e) {
            log.error("写入手势事件失败: {}", e.getMessage());
        }
    }

    // ======================== 查询接口 ========================

    /**
     * 获取最新一帧
     */
    public Map<String, Object> getLatestFrame() {
        return latestFrame;
    }

    /**
     * 查询历史数据
     */
    public List<EmgFrame> getHistory(String deviceId, int limit) {
        LambdaQueryWrapper<EmgFrame> wrapper = new LambdaQueryWrapper<>();
        if (deviceId != null) {
            wrapper.eq(EmgFrame::getDeviceId, deviceId);
        }
        wrapper.orderByDesc(EmgFrame::getId).last("LIMIT " + Math.min(limit, 1000));
        return emgFrameMapper.selectList(wrapper);
    }

    /**
     * 查询手势事件
     */
    public List<GestureEvent> getGestureEvents(String deviceId, int limit) {
        LambdaQueryWrapper<GestureEvent> wrapper = new LambdaQueryWrapper<>();
        if (deviceId != null) {
            wrapper.eq(GestureEvent::getDeviceId, deviceId);
        }
        wrapper.orderByDesc(GestureEvent::getId).last("LIMIT " + Math.min(limit, 500));
        return gestureEventMapper.selectList(wrapper);
    }

    // ======================== 工具方法 ========================

    private EmgFrame convertToEntity(String deviceId, EmgFrameDTO dto) {
        EmgFrame frame = new EmgFrame();
        frame.setDeviceId(deviceId);
        frame.setDeviceTs(dto.getDevice_ts());
        frame.setServerTime(LocalDateTime.now());
        frame.setEmgData(JSON.toJSONString(dto.getEmg()));

        // Pack1 的 8 通道
        if (dto.getEmg() != null && !dto.getEmg().isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (Integer ch : dto.getEmg().get(0)) {
                if (sb.length() > 0) sb.append(",");
                sb.append(ch);
            }
            frame.setEmgPack1(sb.toString());
        }

        if (dto.getAcc() != null && dto.getAcc().size() >= 3) {
            frame.setAccX(dto.getAcc().get(0));
            frame.setAccY(dto.getAcc().get(1));
            frame.setAccZ(dto.getAcc().get(2));
        }
        if (dto.getGyro() != null && dto.getGyro().size() >= 3) {
            frame.setGyroX(dto.getGyro().get(0));
            frame.setGyroY(dto.getGyro().get(1));
            frame.setGyroZ(dto.getGyro().get(2));
        }
        if (dto.getAngle() != null && dto.getAngle().size() >= 3) {
            frame.setPitch(dto.getAngle().get(0));
            frame.setRoll(dto.getAngle().get(1));
            frame.setYaw(dto.getAngle().get(2));
        }

        frame.setBattery(dto.getBattery());
        frame.setGesture(dto.getGesture());
        frame.setConfidence(dto.getConfidence());

        return frame;
    }

    // ======================== 定时清理旧数据 ========================

    /**
     * 每天凌晨 3 点清理 7 天前的数据
     */
    @Scheduled(cron = "0 0 3 * * ?")
    public void cleanupOldData() {
        LocalDateTime cutoff = LocalDateTime.now().minusDays(7);
        LambdaQueryWrapper<EmgFrame> wrapper = new LambdaQueryWrapper<>();
        wrapper.lt(EmgFrame::getServerTime, cutoff);
        int deleted = emgFrameMapper.delete(wrapper);
        log.info("清理旧数据: 删除 {} 条帧记录", deleted);
    }
}
