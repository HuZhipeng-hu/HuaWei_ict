package com.huaweiict.emg.config;

import com.huaweiict.emg.websocket.EmgWebSocketHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

/**
 * WebSocket 配置
 *
 * 注册两个端点：
 * - /ws/emg : OrangePi 设备端上报数据
 * - /ws/app : HarmonyOS App 接收实时数据
 */
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Autowired
    private EmgWebSocketHandler emgWebSocketHandler;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(emgWebSocketHandler, "/ws/emg", "/ws/app")
                .setAllowedOrigins("*");  // 开发阶段允许所有来源，生产环境请限制
    }
}
