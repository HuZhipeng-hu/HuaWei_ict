package com.huaweiict.emg.websocket;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.huaweiict.emg.dto.EmgFrameDTO;
import com.huaweiict.emg.service.EmgDataService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.*;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * WebSocket 处理器
 *
 * 两个端点：
 * 1. /ws/emg    - OrangePi 上报数据（设备端）
 * 2. /ws/app    - HarmonyOS App 接收数据（客户端）
 *
 * 数据流：OrangePi → /ws/emg → processFrame() → /ws/app → HarmonyOS App
 */
@Slf4j
@Component
public class EmgWebSocketHandler extends TextWebSocketHandler {

    /** OrangePi 设备端连接 (deviceId → session) */
    private final Map<String, WebSocketSession> deviceSessions = new ConcurrentHashMap<>();

    /** App 客户端连接 (sessionId → session) */
    private final Map<String, WebSocketSession> appSessions = new ConcurrentHashMap<>();

    /** 当前处理的路径类型 (session → "device" | "app") */
    private final Map<String, String> sessionTypes = new ConcurrentHashMap<>();

    /** App 客户端当前选择的训练手势 (sessionId → gesture) */
    private final Map<String, String> selectedGestures = new ConcurrentHashMap<>();

    @Autowired
    @Lazy
    private EmgDataService emgDataService;

    // ======================== 连接管理 ========================

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        String path = session.getUri() != null ? session.getUri().getPath() : "";
        String sessionId = session.getId();

        if (path.contains("/ws/emg")) {
            // OrangePi 设备端
            sessionTypes.put(sessionId, "device");
            log.info("[WS] 设备端连入: {} ({})", sessionId, session.getRemoteAddress());
        } else if (path.contains("/ws/app")) {
            // App 客户端
            sessionTypes.put(sessionId, "app");
            appSessions.put(sessionId, session);
            log.info("[WS] App客户端连入: {} ({}) | 当前App连接数: {}",
                    sessionId, session.getRemoteAddress(), appSessions.size());

            // 立即发送最新一帧
            try {
                Map<String, Object> latest = emgDataService.getLatestFrame();
                if (!latest.isEmpty()) {
                    session.sendMessage(new TextMessage(JSON.toJSONString(latest)));
                }
            } catch (Exception e) {
                log.error("发送最新帧失败: {}", e.getMessage());
            }
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        String sessionId = session.getId();
        String type = sessionTypes.remove(sessionId);

        if ("device".equals(type)) {
            deviceSessions.values().remove(session);
            log.info("[WS] 设备端断开: {}", sessionId);
        } else if ("app".equals(type)) {
            appSessions.remove(sessionId);
            selectedGestures.remove(sessionId); // 清理手势选择信息
            log.info("[WS] App断开: {} | 剩余App连接: {}", sessionId, appSessions.size());
        }
    }

    // ======================== 消息处理 ========================

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        String type = sessionTypes.get(session.getId());

        if ("device".equals(type)) {
            handleDeviceMessage(session, message.getPayload());
        } else if ("app".equals(type)) {
            handleAppMessage(session, message.getPayload());
        }
    }

    /**
     * 处理 OrangePi 设备端消息
     */
    private void handleDeviceMessage(WebSocketSession session, String payload) {
        try {
            JSONObject json = JSON.parseObject(payload);
            String msgType = json.getString("type");

            if ("register".equals(msgType)) {
                // 设备注册
                String deviceId = json.getString("deviceId");
                deviceSessions.put(deviceId, session);
                log.info("[WS] 设备注册: {}", deviceId);

                // 回复确认
                JSONObject ack = new JSONObject();
                ack.put("type", "register_ack");
                ack.put("status", "ok");
                session.sendMessage(new TextMessage(ack.toJSONString()));

            } else if ("emg_frame".equals(msgType)) {
                // EMG 帧数据
                String deviceId = json.getString("deviceId");
                JSONObject data = json.getJSONObject("data");

                EmgFrameDTO dto = data.toJavaObject(EmgFrameDTO.class);
                emgDataService.processFrame(deviceId, dto);
            }

        } catch (Exception e) {
            log.error("[WS] 解析设备消息失败: {}", e.getMessage());
        }
    }

    /**
     * 处理 App 客户端消息（获取最新数据、选择训练手势等）
     */
    private void handleAppMessage(WebSocketSession session, String payload) {
        try {
            JSONObject json = JSON.parseObject(payload);
            String action = json.getString("action");

            if ("get_latest".equals(action)) {
                // 请求最新数据
                Map<String, Object> latest = emgDataService.getLatestFrame();
                session.sendMessage(new TextMessage(JSON.toJSONString(latest)));
                
            } else if ("select_gesture".equals(action)) {
                // 选择训练手势
                String gesture = json.getString("gesture");
                String sessionId = session.getId();
                selectedGestures.put(sessionId, gesture);
                log.info("[WS] App [{}] 选择手势: {}", sessionId, gesture);
                
                // 发送确认消息
                JSONObject ack = new JSONObject();
                ack.put("type", "gesture_selected");
                ack.put("gesture", gesture);
                ack.put("timestamp", System.currentTimeMillis());
                session.sendMessage(new TextMessage(ack.toJSONString()));
                
            } else if ("start_training".equals(action)) {
                // 开始训练（可选：记录训练状态）
                String gesture = selectedGestures.getOrDefault(session.getId(), "unknown");
                log.info("[WS] App [{}] 开始训练手势: {}", session.getId(), gesture);
                
            } else if ("stop_training".equals(action)) {
                // 停止训练
                log.info("[WS] App [{}] 停止训练", session.getId());
            }
            
            // 可扩展更多 App 端交互命令
        } catch (Exception e) {
            log.error("[WS] 解析App消息失败: {}", e.getMessage());
        }
    }

    // ======================== 广播推送 ========================

    /**
     * 向所有 App 客户端推送数据（由 EmgDataService 调用）
     */
    public void broadcastToApps(String jsonMessage) {
        if (appSessions.isEmpty()) return;

        TextMessage msg = new TextMessage(jsonMessage);
        appSessions.forEach((id, session) -> {
            try {
                if (session.isOpen()) {
                    session.sendMessage(msg);
                }
            } catch (IOException e) {
                log.warn("推送App失败 [{}]: {}", id, e.getMessage());
                appSessions.remove(id);
            }
        });
    }

    // ======================== 工具方法 ========================

    /**
     * 获取指定会话当前选择的手势
     */
    public String getSelectedGesture(String sessionId) {
        return selectedGestures.get(sessionId);
    }

    /**
     * 向特定App客户端发送消息
     */
    public void sendToApp(String sessionId, String message) {
        WebSocketSession session = appSessions.get(sessionId);
        if (session != null && session.isOpen()) {
            try {
                session.sendMessage(new TextMessage(message));
            } catch (IOException e) {
                log.warn("发送消息到App [{}] 失败: {}", sessionId, e.getMessage());
            }
        }
    }

    /**
     * 获取连接统计
     */
    public Map<String, Integer> getConnectionStats() {
        Map<String, Integer> stats = new ConcurrentHashMap<>();
        stats.put("deviceCount", deviceSessions.size());
        stats.put("appCount", appSessions.size());
        return stats;
    }
}
