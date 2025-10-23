"""
MQTT 客户端 - 简化的 MQTT 客户端
"""

import paho.mqtt.client as mqtt
import json
import time
from loguru import logger
from typing import Callable, Optional


class MQTTClient:
    """MQTT 客户端"""
    
    def __init__(self, config=None):
        if config:
            self.host = config.mqtt['host']
            self.port = config.mqtt['port']
            self.username = config.mqtt['username']
            self.password = config.mqtt['password']
            self.keepalive = config.mqtt['keepalive']
            self.region_id = getattr(config, 'region_id', 'unknown')
        else:
            # 兼容旧版本
            self.host = self._get_env('MQTT_BROKER_HOST', 'localhost')
            self.port = int(self._get_env('MQTT_BROKER_PORT', '1883'))
            self.username = self._get_env('MQTT_USER', 'mqtt')
            self.password = self._get_env('MQTT_PASSWORD', 'mqtt2024#')
            self.keepalive = int(self._get_env('MQTT_KEEPALIVE', '60'))
            self.region_id = self._get_env('REGION_ID', 'unknown')
        
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.running = False
        self.message_callback: Optional[Callable] = None
    
    def _get_env(self, key: str, default: str) -> str:
        """获取环境变量"""
        import os
        return os.environ.get(key, default)
    
    def connect(self):
        """连接到 MQTT Broker"""
        if self.client and self.connected:
            logger.info("MQTT 客户端已连接，跳过重复连接")
            return
            
        try:
            # 先关闭现有连接
            if self.client:
                self.close()
            
            # 使用 Client ID 避免重复连接问题
            import uuid
            client_id = f"regional_node_{self.region_id}_{uuid.uuid4().hex[:8]}"
            
            # 使用正确的协议版本创建客户端
            self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
            self.client.username_pw_set(self.username, self.password)
            
            # 设置连接参数
            self.client.keepalive = self.keepalive
            
            # 设置回调
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_subscribe = self._on_subscribe
            self.client.on_log = self._on_log
            
            # 连接
            logger.info(f"Connecting to MQTT Broker: {self.host}:{self.port}")
            self.client.connect(self.host, self.port, self.keepalive)
            self.client.loop_start()
            
            # 等待连接
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            if self.connected:
                self.running = True
                # 不输出连接成功日志，减少冗余
            else:
                raise Exception("MQTT Connection Timeout")
                
        except Exception as e:
            logger.error(f"MQTT Broker Connection Failed: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            # 不输出连接成功日志，减少冗余
        else:
            logger.error(f"MQTT Connection Failed - Error Code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        logger.warning(f"MQTT Connection Disconnected - Error Code: {rc}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """订阅回调"""
        logger.debug(f"MQTT Subscription Successful - Message ID: {mid}, QoS: {granted_qos}")
    
    def _on_log(self, client, userdata, level, buf):
        """日志回调 - 用于调试"""
        if level == mqtt.MQTT_LOG_ERR:
            logger.error(f"MQTT Error: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            logger.warning(f"MQTT Warning: {buf}")
        # 其他级别的日志不输出，避免过多日志
    
    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.info(f"Received MQTT Message: {topic} -> {payload}")
            
            # 解析消息
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                message = payload
            
            # 调用用户回调
            if self.message_callback:
                self.message_callback(topic, message)
                
        except UnicodeDecodeError as e:
            logger.error(f"MQTT Message Decode Failed: {e}")
        except Exception as e:
            logger.error(f"MQTT Message Processing Failed: {e}")
            # 记录原始消息用于调试
            logger.debug(f"Raw Message: topic={msg.topic}, payload={msg.payload}")
    
    def subscribe(self, topic: str):
        """订阅主题"""
        if not self.client or not self.connected:
            logger.error("MQTT Client Not Connected")
            return
        
        try:
            result = self.client.subscribe(topic, qos=1)  # 使用 QoS 1 提高可靠性
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                # logger.info(f"已订阅主题: {topic}")
                pass
            else:
                # logger.error(f"订阅主题失败: {topic}, 错误码: {result[0]}")
                pass
        except Exception as e:
            logger.error(f"Topic Subscription Failed: {e}")
    
    def publish(self, topic: str, message: str):
        """发布消息"""
        if not self.client or not self.connected:
            logger.error("MQTT Client Not Connected")
            return
        
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published Message to {topic}: {message}")
            else:
                logger.error(f"Message Publishing Failed - Error Code: {result.rc}")
        except Exception as e:
            logger.error(f"Message Publishing Failed: {e}")
    
    def set_message_callback(self, callback: Callable):
        """设置消息回调"""
        self.message_callback = callback
    
    def loop_forever(self):
        """保持连接 - 使用非阻塞方式"""
        if self.client and self.connected:
            logger.info("MQTT Starting Message Loop...")
            # 使用 loop_start 而不是 loop_forever 避免阻塞
            # loop_start 已经在 connect 中调用了
            # 这里只需要保持连接状态
            while self.connected and self.running:
                time.sleep(1)
        else:
            logger.error("MQTT Client Not Connected - Cannot Start Loop")
    
    def close(self):
        """关闭连接"""
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("MQTT Connection Closed")