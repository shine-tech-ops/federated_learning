"""
设备端 MQTT 处理器
"""

import paho.mqtt.client as mqtt
import json
import time
from loguru import logger
from typing import Callable, Optional


class MQTTHandler:
    """设备端 MQTT 处理器"""
    
    def __init__(self, device_id: str, config: dict):
        self.device_id = device_id
        self.host = config['host']
        self.port = config['port']
        self.username = config['username']
        self.password = config['password']
        self.keepalive = config['keepalive']
        
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.message_callback: Optional[Callable] = None
    
    def connect(self):
        """连接到 MQTT Broker"""
        try:
            self.client = mqtt.Client(client_id=f"device_{self.device_id}")
            self.client.username_pw_set(self.username, self.password)
            
            # 设置回调
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            # 连接
            self.client.connect(self.host, self.port, self.keepalive)
            self.client.loop_start()
            
            # 等待连接
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            if self.connected:
                pass
            else:
                raise Exception("MQTT 连接超时")
                
        except Exception as e:
            logger.error(f"连接 MQTT Broker 失败: {e}")
            raise
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"设备 {self.device_id} MQTT 连接成功")
        else:
            logger.error(f"设备 {self.device_id} MQTT 连接失败，错误码: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        logger.warning(f"设备 {self.device_id} MQTT 连接断开，错误码: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.info(f"设备 {self.device_id} 收到 MQTT 消息: {topic}")
            
            # 解析消息
            try:
                message = json.loads(payload)
            except json.JSONDecodeError:
                message = payload
            
            # 调用用户回调
            if self.message_callback:
                self.message_callback(topic, message)
                
        except Exception as e:
            logger.error(f"处理 MQTT 消息失败: {e}")
    
    def subscribe_device_commands(self):
        """订阅设备专属命令主题"""
        if not self.client or not self.connected:
            logger.error("MQTT 客户端未连接")
            return
        
        # 订阅设备专属主题
        command_topic = f"federated_task_{self.device_id}/task_start"
        self.client.subscribe(command_topic)
        logger.info(f"设备 {self.device_id} 已订阅命令主题: {command_topic}")
    
    def publish_status(self, status_data: dict):
        """发布设备状态"""
        if not self.client or not self.connected:
            logger.error("MQTT 客户端未连接")
            return
        
        topic = f"region/1/devices/{self.device_id}/status"
        message = json.dumps(status_data)
        
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"设备 {self.device_id} 状态发布成功")
            else:
                logger.error(f"设备 {self.device_id} 状态发布失败，错误码: {result.rc}")
        except Exception as e:
            logger.error(f"发布状态失败: {e}")
    
    def publish_training_result(self, result_data: dict):
        """发布训练结果"""
        if not self.client or not self.connected:
            logger.error("MQTT 客户端未连接")
            return
        
        topic = f"region/1/devices/{self.device_id}/result"
        message = json.dumps(result_data)
        
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"设备 {self.device_id} 训练结果发布成功")
            else:
                logger.error(f"设备 {self.device_id} 训练结果发布失败，错误码: {result.rc}")
        except Exception as e:
            logger.error(f"发布训练结果失败: {e}")
    
    def set_message_callback(self, callback: Callable):
        """设置消息回调"""
        self.message_callback = callback
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info(f"设备 {self.device_id} MQTT 连接已关闭")
