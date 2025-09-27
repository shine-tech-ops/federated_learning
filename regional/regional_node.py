#!/usr/bin/env python3
"""
Regional Node - 区域节点服务
- RabbitMQ: 与中央服务器通讯，接收任务指令和状态上报
- MQTT: 与边缘设备通讯，发送任务指令和接收设备状态
"""

import json
import time
import threading
from loguru import logger
from typing import Dict, Any

from config import config, validate_config
from app.utils.rabbitmq_client import RabbitMQClient
from app.utils.mqtt_client import MQTTClient
from app.utils.http_client import HTTPClient
from app.service.task_manager import TaskManager
from app.flower.server_manager import FlowerServerManager


class RegionalNode:
    """区域节点主服务类
    
    职责分离：
    - RabbitMQ: 与中央服务器通讯，接收任务指令，上报任务状态
    - MQTT: 与边缘设备通讯，发送任务指令，接收设备状态和训练结果
    """
    
    def __init__(self):
        # 验证配置
        if not validate_config():
            raise ValueError("配置验证失败，请检查环境变量")
        
        # 使用配置文件
        self.config = config
        self.region_id = self.config.region_id
        self.task_manager = TaskManager()
        
        # 通讯客户端
        self.rabbitmq_client = RabbitMQClient(self.config)  # 接收中央服务器指令
        self.mqtt_client = MQTTClient(self.config)          # 与边缘设备通讯
        self.http_client = HTTPClient(self.config.central_server)  # 向中央服务器上报状态
        
        # Flower 服务器管理器
        self.flower_server = FlowerServerManager(self.region_id)
        
        # 运行状态
        self.running = False
    
    def start(self):
        """启动区域节点服务"""
        logger.info(f"启动区域节点服务: {self.region_id}")
        
        try:
            # 初始化连接
            self._init_connections()
            
            # 启动消息队列消费者
            self._start_consumers()
            
            # 设置运行状态
            self.running = True
            
            # 主循环
            self._main_loop()
            
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭服务...")
        except Exception as e:
            logger.error(f"服务运行错误: {e}")
        finally:
            self.stop()
    
    def _init_connections(self):
        """初始化各种连接"""
        logger.info("初始化连接...")
        
        # 连接 RabbitMQ (与中央服务器通讯)
        logger.info("连接 RabbitMQ (中央服务器通讯)...")
        self.rabbitmq_client.connect()
        
        # 连接 MQTT (与边缘设备通讯)
        logger.info("连接 MQTT (边缘设备通讯)...")
        self.mqtt_client.connect()
        
        logger.info("所有连接初始化完成")
    
    def _start_consumers(self):
        """启动消息队列消费者"""
        logger.info("启动消息队列消费者...")
        
        # 启动 RabbitMQ 消费者线程 (接收中央服务器指令)
        rabbitmq_thread = threading.Thread(
            target=self._consume_rabbitmq_messages,
            daemon=True,
            name="RabbitMQ-Consumer"
        )
        rabbitmq_thread.start()
        
        # 启动 MQTT 消费者线程 (接收边缘设备状态和训练结果)
        mqtt_thread = threading.Thread(
            target=self._consume_mqtt_messages,
            daemon=True,
            name="MQTT-Consumer"
        )
        mqtt_thread.start()
        
        logger.info("消息队列消费者启动完成")
    
    def _consume_rabbitmq_messages(self):
        """消费 RabbitMQ 消息 (接收中央服务器指令)"""
        exchange_name = self.config.get_rabbitmq_exchange()
        queue_name = self.config.get_rabbitmq_queue()
        
        logger.info(f"开始消费 RabbitMQ 消息 (中央服务器指令): {exchange_name}")
        
        try:
            self.rabbitmq_client.consumer(
                exchange=exchange_name,
                queue=queue_name,
                callback=self._handle_rabbitmq_message
            )
        except Exception as e:
            logger.error(f"RabbitMQ 消费者错误: {e}")
    
    def _consume_mqtt_messages(self):
        """消费 MQTT 消息 (接收边缘设备状态和训练结果)"""
        logger.info(f"开始消费 MQTT 消息 (边缘设备通讯): {self.config.mqtt['topic_prefix']}")
        
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                # 等待 MQTT 连接建立
                timeout = 30
                while not self.mqtt_client.connected and timeout > 0 and self.running:
                    time.sleep(0.1)
                    timeout -= 0.1
                
                if not self.mqtt_client.connected:
                    logger.error("MQTT 连接超时，无法开始消费")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"等待 5 秒后重试 ({retry_count}/{max_retries})...")
                        time.sleep(5)
                    continue
                
                # 设置 MQTT 消息回调
                self.mqtt_client.set_message_callback(self._handle_mqtt_message)
                
                # 订阅设备状态相关主题
                self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('status'))
                self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('training'))
                self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('heartbeat'))
                self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('result'))
                
                logger.info("MQTT 订阅完成，开始监听消息...")
                
                # 使用 loop_forever 保持连接
                self.mqtt_client.loop_forever()
                
            except Exception as e:
                logger.error(f"MQTT 消费者错误: {e}")
                retry_count += 1
                
                if retry_count < max_retries:
                    logger.info(f"等待 5 秒后重试 ({retry_count}/{max_retries})...")
                    time.sleep(5)
                else:
                    logger.error(f"MQTT 消费者重试次数已达上限，停止重试")
                    break
    
    def _handle_rabbitmq_message(self, ch, method, properties, body):
        """处理 RabbitMQ 消息 (来自中央服务器的指令)"""
        try:
            message = json.loads(body)
            logger.info(f"收到中央服务器指令: {message}")
            
            # 根据消息类型处理
            message_type = message.get('message_type')
            
            if message_type == 'federated_task_start':
                self._handle_task_start(message)
            elif message_type == 'federated_task_pause':
                self._handle_task_pause(message)
            elif message_type == 'federated_task_resume':
                self._handle_task_resume(message)
            elif message_type == 'federated_task_stop':
                self._handle_task_stop(message)
            else:
                logger.warning(f"未知的中央服务器指令类型: {message_type}")
                
        except Exception as e:
            logger.error(f"处理 RabbitMQ 消息错误: {e}")
        finally:
            # 确认消息
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    def _handle_task_start(self, task_data: Dict[str, Any]):
        """处理任务开始消息"""
        logger.info(f"开始处理任务: {task_data['task_id']}")
        
        try:
            # 通知任务管理器
            self.task_manager.start_task(task_data)
            
            # 通过 MQTT 通知边缘设备
            self._notify_devices_task_start(task_data)
            
            # 上报任务状态到中央服务器
            self._report_task_status_to_central_server(
                task_data['task_id'], 
                'started', 
                {'region_id': self.region_id}
            )
            
            logger.info(f"任务 {task_data['task_id']} 处理完成")
            
        except Exception as e:
            logger.error(f"处理任务开始错误: {e}")
            # 上报错误状态
            self._report_task_status_to_central_server(
                task_data['task_id'], 
                'error', 
                {'error': str(e)}
            )
    
    def _handle_task_pause(self, task_data: Dict[str, Any]):
        """处理任务暂停消息"""
        logger.info(f"暂停任务: {task_data['task_id']}")
        self.task_manager.pause_task(task_data['task_id'])
        self._notify_devices_task_pause(task_data)
        
        # 上报任务状态到中央服务器
        self._report_task_status_to_central_server(
            task_data['task_id'], 
            'paused', 
            {'region_id': self.region_id}
        )
    
    def _handle_task_resume(self, task_data: Dict[str, Any]):
        """处理任务恢复消息"""
        logger.info(f"恢复任务: {task_data['task_id']}")
        self.task_manager.resume_task(task_data['task_id'])
        self._notify_devices_task_resume(task_data)
        
        # 上报任务状态到中央服务器
        self._report_task_status_to_central_server(
            task_data['task_id'], 
            'resumed', 
            {'region_id': self.region_id}
        )
    
    def _handle_task_stop(self, task_data: Dict[str, Any]):
        """处理任务停止消息"""
        logger.info(f"停止任务: {task_data['task_id']}")
        self.task_manager.stop_task(task_data['task_id'])
        self._notify_devices_task_stop(task_data)
        
        # 上报任务状态到中央服务器
        self._report_task_status_to_central_server(
            task_data['task_id'], 
            'stopped', 
            {'region_id': self.region_id}
        )
    
    def _handle_mqtt_message(self, topic: str, message: Any):
        """处理 MQTT 消息 (来自边缘设备的状态和训练结果)"""
        try:
            logger.info(f"收到边缘设备消息: {topic} -> {message}")
            
            # 解析主题获取设备ID和动作类型
            topic_parts = topic.split('/')
            if len(topic_parts) >= 4:
                device_id = topic_parts[2]
                action = topic_parts[3]
                
                if action == 'status':
                    self._handle_device_status(device_id, message)
                elif action == 'heartbeat':
                    self._handle_device_heartbeat(device_id, message)
                elif action == 'training':
                    self._handle_device_training(device_id, message)
                elif action == 'result':
                    self._handle_device_result(device_id, message)
                else:
                    logger.warning(f"未知的设备消息类型: {action}")
            else:
                logger.warning(f"无法解析 MQTT 主题: {topic}")
                
        except Exception as e:
            logger.error(f"处理 MQTT 消息错误: {e}")
    
    def _handle_device_status(self, device_id: str, status_data: Any):
        """处理设备状态消息"""
        logger.info(f"设备 {device_id} 状态更新: {status_data}")
        # 更新设备状态到任务管理器
        self.task_manager.update_device_status(device_id, status_data)
    
    def _handle_device_heartbeat(self, device_id: str, heartbeat_data: Any):
        """处理设备心跳消息"""
        logger.debug(f"设备 {device_id} 心跳: {heartbeat_data}")
        # 更新设备心跳时间
        self.task_manager.update_device_heartbeat(device_id, heartbeat_data)
    
    def _handle_device_training(self, device_id: str, training_data: Any):
        """处理设备训练进度消息"""
        logger.info(f"设备 {device_id} 训练进度: {training_data}")
        # 更新设备训练进度
        self.task_manager.update_device_training_progress(device_id, training_data)
    
    def _handle_device_result(self, device_id: str, result_data: Any):
        """处理设备训练结果消息"""
        logger.info(f"设备 {device_id} 训练结果: {result_data}")
        # 处理设备训练结果
        self.task_manager.handle_device_result(device_id, result_data)
        
        # 通过 RabbitMQ 上报结果到中央服务器
        self._report_result_to_central_server(device_id, result_data)
    
    def _notify_devices_task_start(self, task_data: Dict[str, Any]):
        """通知边缘设备任务开始"""
        # 1. 先启动 Flower 服务器
        logger.info("启动 Flower 服务器...")
        flower_server_info = self.flower_server.start_server(task_data)
        
        # 2. 获取边缘设备列表
        edge_devices = task_data.get('edge_devices', [])
        
        if not edge_devices:
            logger.warning("没有找到在线的边缘设备")
            return
        
        # 3. 为每个设备发送任务开始指令（包含 Flower 服务器信息）
        for device in edge_devices:
            device_id = device.get('device_id')
            if not device_id:
                logger.warning(f"设备缺少 device_id: {device}")
                continue
                
            topic = self.config.get_mqtt_device_command_topic(device_id, 'task_start')
            message = {
                "action": "task_start",
                "task_id": task_data['task_id'],
                "task_name": task_data['task_name'],
                "model_info": task_data['model_info'],
                "model_version": task_data['model_version'],
                "rounds": task_data['rounds'],
                "aggregation_method": task_data['aggregation_method'],
                "device_info": device,  # 包含设备信息
                "flower_server": flower_server_info  # 添加 Flower 服务器信息
            }
            
            self.mqtt_client.publish(topic, json.dumps(message))
            logger.info(f"已通知设备 {device_id} 任务开始: {task_data['task_id']}")
        
        logger.info(f"已通知 {len(edge_devices)} 个设备任务开始: {task_data['task_id']}")
    
    def _notify_devices_task_pause(self, task_data: Dict[str, Any]):
        """通知边缘设备任务暂停"""
        self._notify_devices_by_action(task_data, 'task_pause')
    
    def _notify_devices_task_resume(self, task_data: Dict[str, Any]):
        """通知边缘设备任务恢复"""
        self._notify_devices_by_action(task_data, 'task_resume')
    
    def _notify_devices_task_stop(self, task_data: Dict[str, Any]):
        """通知边缘设备任务停止"""
        # 先停止 Flower 服务器
        self.flower_server.stop_server()
        
        # 然后通知设备
        self._notify_devices_by_action(task_data, 'task_stop')
    
    def _notify_devices_by_action(self, task_data: Dict[str, Any], action: str):
        """通用方法：向所有边缘设备发送指定动作"""
        # 获取边缘设备列表
        edge_devices = task_data.get('edge_devices', [])
        
        if not edge_devices:
            logger.warning(f"没有找到在线的边缘设备，无法发送 {action} 指令")
            return
        
        # 为每个设备发送指令
        for device in edge_devices:
            device_id = device.get('device_id')
            if not device_id:
                logger.warning(f"设备缺少 device_id: {device}")
                continue
                
            topic = self.config.get_mqtt_device_command_topic(device_id, action)
            message = {
                "action": action,
                "task_id": task_data['task_id'],
                "device_info": device
            }
            
            self.mqtt_client.publish(topic, json.dumps(message))
            logger.info(f"已通知设备 {device_id} {action}: {task_data['task_id']}")
        
        logger.info(f"已通知 {len(edge_devices)} 个设备 {action}: {task_data['task_id']}")
    
    def _report_result_to_central_server(self, device_id: str, result_data: Any):
        """通过 HTTP API 上报设备结果到中央服务器"""
        try:
            success = self.http_client.report_device_result(device_id, result_data, self.region_id)
            if success:
                logger.info(f"已上报设备 {device_id} 结果到中央服务器")
            else:
                logger.error(f"上报设备 {device_id} 结果失败")
        except Exception as e:
            logger.error(f"上报设备结果到中央服务器失败: {e}")
    
    def _report_task_status_to_central_server(self, task_id: str, status: str, details: Dict[str, Any] = None):
        """通过 HTTP API 上报任务状态到中央服务器"""
        try:
            success = self.http_client.report_task_status(task_id, status, self.region_id, details)
            if success:
                logger.info(f"已上报任务 {task_id} 状态 {status} 到中央服务器")
            else:
                logger.error(f"上报任务 {task_id} 状态失败")
        except Exception as e:
            logger.error(f"上报任务状态到中央服务器失败: {e}")
    
    def _main_loop(self):
        """主循环 - 监控任务和设备状态"""
        logger.info("进入主循环...")
        
        while self.running:
            try:
                # 检查任务状态 (通过 RabbitMQ 与中央服务器同步)
                self._check_task_status()
                
                # 检查设备状态 (通过 MQTT 与边缘设备同步)
                self._check_device_status()
                
                # 休眠
                time.sleep(self.config.task['status_check_interval'])
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(5)
    
    def _check_task_status(self):
        """检查任务状态 (与中央服务器同步)"""
        # 实现任务状态检查逻辑
        # 通过 RabbitMQ 与中央服务器同步任务状态
        pass
    
    def _check_device_status(self):
        """检查设备状态 (与边缘设备同步)"""
        # 实现设备状态检查逻辑
        # 通过 MQTT 与边缘设备同步设备状态
        pass
    
    def stop(self):
        """停止服务"""
        logger.info("正在停止区域节点服务...")
        self.running = False
        
        # 关闭连接
        if hasattr(self, 'rabbitmq_client'):
            logger.info("关闭 RabbitMQ 连接 (中央服务器通讯)...")
            self.rabbitmq_client.close()
        if hasattr(self, 'mqtt_client'):
            logger.info("关闭 MQTT 连接 (边缘设备通讯)...")
            self.mqtt_client.close()
        if hasattr(self, 'http_client'):
            logger.info("关闭 HTTP 客户端 (状态上报)...")
            self.http_client.close()
        if hasattr(self, 'flower_server'):
            logger.info("关闭 Flower 服务器...")
            self.flower_server.stop_server()
        
        logger.info("区域节点服务已停止")


if __name__ == "__main__":
    # 配置日志
    logger.add(
        "logs/regional.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        level="INFO",
        encoding="utf-8",
    )
    
    # 启动服务
    node = RegionalNode()
    node.start()
