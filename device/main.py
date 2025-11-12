#!/usr/bin/env python3
"""
边缘设备主程序
模拟联邦学习客户端
"""

import json
import time
import threading
import sys
import os
from loguru import logger

# 添加共享模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from mqtt_handler import MQTTHandler
from flower_client import FlowerClient
from mnist_trainer import MNISTTrainer
from http_client import HTTPClient


class EdgeDevice:
    """边缘设备主类"""
    
    def __init__(self, device_id: str, mqtt_config: dict, http_config: dict = None):
        self.device_id = device_id
        self.mqtt_handler = MQTTHandler(device_id, mqtt_config)
        self.flower_client = None
        self.trainer = None
        self.current_task = None
        self.running = False
        self.region_id = None  # 区域节点ID，从注册信息或配置中获取
        
        # HTTP客户端配置（用于发送心跳到中央服务器）
        http_base_url = http_config.get('base_url', 'http://localhost:8000') if http_config else 'http://localhost:8000'
        self.http_client = HTTPClient(base_url=http_base_url)
        
        # 设置 MQTT 消息回调
        self.mqtt_handler.set_message_callback(self._handle_mqtt_message)
    
    def start(self):
        """启动设备"""
        logger.info(f"启动边缘设备: {self.device_id}")
        
        try:
            # 连接 MQTT
            self.mqtt_handler.connect()
            
            # 订阅设备专属主题
            self.mqtt_handler.subscribe_device_commands()
            
            # 设置运行状态
            self.running = True
            
            # 主循环
            self._main_loop()
            
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭设备...")
        except Exception as e:
            logger.error(f"设备运行错误: {e}")
        finally:
            self.stop()
    
    def _handle_mqtt_message(self, topic: str, message: dict):
        """处理 MQTT 消息"""
        try:
            action = message.get('action')
            logger.info(f"收到 MQTT 消息: {action}")
            
            if action == 'task_start':
                self._handle_task_start(message)
            elif action == 'task_pause':
                self._handle_task_pause(message)
            elif action == 'task_resume':
                self._handle_task_resume(message)
            elif action == 'task_stop':
                self._handle_task_stop(message)
            else:
                logger.warning(f"未知动作: {action}")
                
        except Exception as e:
            logger.error(f"处理 MQTT 消息错误: {e}")
    
    def _handle_task_start(self, message: dict):
        """处理任务开始"""
        logger.info(f"开始处理任务: {message['task_id']}")
        
        try:
            # 保存任务信息
            self.current_task = message
            
            # 获取 Flower 服务器信息
            flower_server = message.get('flower_server', {})
            if not flower_server:
                logger.error("缺少 Flower 服务器信息")
                return
            
            # 创建训练器
            self.trainer = MNISTTrainer(
                device_id=self.device_id,
                task_data=message
            )
            
            # 创建 Flower 客户端
            self.flower_client = FlowerClient(
                device_id=self.device_id,
                trainer=self.trainer,
                #server_address=f"{flower_server['host']}:{flower_server['port']}"
                server_address=f"192.168.1.4:{flower_server['port']}"
            )
            
            # 在后台线程启动联邦学习
            fl_thread = threading.Thread(
                target=self._start_federated_learning,
                daemon=True
            )
            fl_thread.start()
            
            logger.info(f"设备 {self.device_id} 开始联邦学习")
            
        except Exception as e:
            logger.error(f"处理任务开始错误: {e}")
    
    def _handle_task_pause(self, message: dict):
        """处理任务暂停"""
        logger.info(f"暂停任务: {message['task_id']}")
        if self.flower_client:
            self.flower_client.pause()
    
    def _handle_task_resume(self, message: dict):
        """处理任务恢复"""
        logger.info(f"恢复任务: {message['task_id']}")
        if self.flower_client:
            self.flower_client.resume()
    
    def _handle_task_stop(self, message: dict):
        """处理任务停止"""
        logger.info(f"停止任务: {message['task_id']}")
        if self.flower_client:
            self.flower_client.stop()
        self.current_task = None
    
    def _start_federated_learning(self):
        """启动联邦学习"""
        try:
            if self.flower_client:
                self.flower_client.start()
        except Exception as e:
            logger.error(f"联邦学习错误: {e}")
    
    def _main_loop(self):
        """主循环"""
        logger.info("进入主循环...")
        
        while self.running:
            try:
                # 发送心跳
                self._send_heartbeat()
                
                # 休眠
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(5)
    
    def _send_heartbeat(self):
        """发送心跳（同时发送MQTT心跳和HTTP心跳到中央服务器）"""
        try:
            heartbeat_data = {
                "device_id": self.device_id,
                "status": "online",
                "timestamp": time.time(),
                "current_task": self.current_task['task_id'] if self.current_task else None
            }
            
            # 发送MQTT心跳（给区域节点）
            self.mqtt_handler.publish_status(heartbeat_data)
            
            # 发送HTTP心跳到中央服务器
            if self.region_id:
                device_context = {
                    "status": "online",
                    "timestamp": time.time(),
                    "current_task": self.current_task['task_id'] if self.current_task else None
                }
                self.http_client.send_heartbeat(
                    device_id=self.device_id,
                    region_id=self.region_id,
                    device_context=device_context
                )
            else:
                logger.warning(f"设备 {self.device_id} 未设置region_id，跳过HTTP心跳")
            
        except Exception as e:
            logger.error(f"发送心跳失败: {e}")
    
    def stop(self):
        """停止设备"""
        logger.info("正在停止边缘设备...")
        self.running = False
        
        if self.flower_client:
            self.flower_client.stop()
        
        if self.mqtt_handler:
            self.mqtt_handler.close()
        
        if self.http_client:
            self.http_client.close()
        
        logger.info("边缘设备已停止")


if __name__ == "__main__":
    # 配置日志
    logger.add(
        f"logs/device_{sys.argv[1] if len(sys.argv) > 1 else 'default'}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        level="INFO",
        encoding="utf-8",
    )
    
    # 获取设备ID和参数
    device_id = sys.argv[1] if len(sys.argv) > 1 else "device_001"
    region_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # 默认region_id为1
    central_server_url = sys.argv[3] if len(sys.argv) > 3 else 'http://localhost:8000'
    
    # MQTT 配置
    mqtt_config = {
        'host': 'localhost',
        'port': 1883,
        'username': 'mqtt',
        'password': 'mqtt2024#',
        'keepalive': 60
    }
    
    # HTTP 配置（用于发送心跳到中央服务器）
    http_config = {
        'base_url': central_server_url
    }
    
    # 启动设备
    device = EdgeDevice(device_id, mqtt_config, http_config)
    device.region_id = region_id  # 设置区域节点ID
    device.start()
