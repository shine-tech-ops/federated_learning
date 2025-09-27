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
            # 使用 Client ID 避免重复连接问题
            import uuid
            client_id = f"regional_node_{self.region_id}_{uuid.uuid4().hex[:8]}"
            self.client = mqtt.Client(client_id=client_id)
            self.client.username_pw_set(self.username, self.password)
            
            # 设置连接参数，避免版本兼容性问题
            self.client._protocol_version = mqtt.MQTTv311
            
            # 设置回调
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_subscribe = self._on_subscribe
            self.client.on_log = self._on_log
            
            # 连接
            self.client.connect(self.host, self.port, self.keepalive)
            self.client.loop_start()
            
            # 等待连接
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            if self.connected:
                logger.info(f"已连接到 MQTT Broker: {self.host}:{self.port}")
            else:
                raise Exception("MQTT 连接超时")
                
        except Exception as e:
            logger.error(f"连接 MQTT Broker 失败: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT 连接成功")
        else:
            logger.error(f"MQTT 连接失败，错误码: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        logger.warning(f"MQTT 连接断开，错误码: {rc}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """订阅回调"""
        logger.debug(f"MQTT 订阅成功，消息ID: {mid}, QoS: {granted_qos}")
    
    def _on_log(self, client, userdata, level, buf):
        """日志回调 - 用于调试"""
        if level == mqtt.MQTT_LOG_ERR:
            logger.error(f"MQTT 错误: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            logger.warning(f"MQTT 警告: {buf}")
        # 其他级别的日志不输出，避免过多日志
    
    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.info(f"收到 MQTT 消息: {topic} -> {payload}")
            
            # 解析消息
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                message = payload
            
            # 调用用户回调
            if self.message_callback:
                self.message_callback(topic, message)
                
        except UnicodeDecodeError as e:
            logger.error(f"MQTT 消息解码失败: {e}")
        except Exception as e:
            logger.error(f"处理 MQTT 消息失败: {e}")
            # 记录原始消息用于调试
            logger.debug(f"原始消息: topic={msg.topic}, payload={msg.payload}")
    
    def subscribe(self, topic: str):
        """订阅主题"""
        if not self.client or not self.connected:
            logger.error("MQTT 客户端未连接")
            return
        
        try:
            result = self.client.subscribe(topic, qos=1)  # 使用 QoS 1 提高可靠性
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"已订阅主题: {topic}")
            else:
                logger.error(f"订阅主题失败: {topic}, 错误码: {result[0]}")
        except Exception as e:
            logger.error(f"订阅主题失败: {e}")
    
    def publish(self, topic: str, message: str):
        """发布消息"""
        if not self.client or not self.connected:
            logger.error("MQTT 客户端未连接")
            return
        
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"已发布消息到 {topic}: {message}")
            else:
                logger.error(f"发布消息失败，错误码: {result.rc}")
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
    
    def set_message_callback(self, callback: Callable):
        """设置消息回调"""
        self.message_callback = callback
    
    def loop_forever(self):
        """保持连接 - 使用 loop_forever 会阻塞线程"""
        if self.client and self.connected:
            logger.info("MQTT 开始循环监听...")
            self.client.loop_forever()
        else:
            logger.error("MQTT 客户端未连接，无法开始循环")
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("MQTT 连接已关闭")