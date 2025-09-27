#!/usr/bin/env python3
"""
Regional Node - 简化的区域节点服务
只负责消息队列通信，不提供 HTTP 接口
"""

import json
import time
import threading
from loguru import logger
from typing import Dict, Any

from config import config, validate_config
from app.utils.rabbitmq_client import RabbitMQClient
from app.utils.mqtt_client import MQTTClient
from app.service.task_manager import TaskManager


class RegionalNode:
    """区域节点主服务类"""
    
    def __init__(self):
        # 验证配置
        if not validate_config():
            raise ValueError("配置验证失败，请检查环境变量")
        
        # 使用配置文件
        self.config = config
        self.region_id = self.config.region_id
        self.task_manager = TaskManager()
        self.rabbitmq_client = RabbitMQClient(self.config)
        self.mqtt_client = MQTTClient(self.config)
        
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
        
        # 连接 MQTT
        self.mqtt_client.connect()
        
        # 连接 RabbitMQ
        self.rabbitmq_client.connect()
        
        logger.info("所有连接初始化完成")
    
    def _start_consumers(self):
        """启动消息队列消费者"""
        logger.info("启动消息队列消费者...")
        
        # 启动 RabbitMQ 消费者线程
        rabbitmq_thread = threading.Thread(
            target=self._consume_rabbitmq_messages,
            daemon=True
        )
        rabbitmq_thread.start()
        
        # 启动 MQTT 消费者线程
        mqtt_thread = threading.Thread(
            target=self._consume_mqtt_messages,
            daemon=True
        )
        mqtt_thread.start()
        
        logger.info("消息队列消费者启动完成")
    
    def _consume_rabbitmq_messages(self):
        """消费 RabbitMQ 消息"""
        exchange_name = self.config.get_rabbitmq_exchange()
        queue_name = self.config.get_rabbitmq_queue()
        
        logger.info(f"开始消费 RabbitMQ 消息: {exchange_name}")
        
        try:
            self.rabbitmq_client.consumer(
                exchange=exchange_name,
                queue=queue_name,
                callback=self._handle_rabbitmq_message
            )
        except Exception as e:
            logger.error(f"RabbitMQ 消费者错误: {e}")
    
    def _consume_mqtt_messages(self):
        """消费 MQTT 消息"""
        logger.info(f"开始消费 MQTT 消息: {self.config.mqtt['topic_prefix']}")
        
        try:
            # 订阅设备状态相关主题
            self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('status'))
            self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('training'))
            self.mqtt_client.subscribe(self.config.get_mqtt_wildcard_topic('heartbeat'))
            
            self.mqtt_client.loop_forever()
        except Exception as e:
            logger.error(f"MQTT 消费者错误: {e}")
    
    def _handle_rabbitmq_message(self, ch, method, properties, body):
        """处理 RabbitMQ 消息"""
        try:
            message = json.loads(body)
            logger.info(f"收到任务消息: {message}")
            
            # 根据消息类型处理
            message_type = message.get('message_type')
            
            if message_type == 'federated_task_start':
                self._handle_task_start(message)
            elif message_type == 'federated_task_pause':
                self._handle_task_pause(message)
            elif message_type == 'federated_task_resume':
                self._handle_task_resume(message)
            else:
                logger.warning(f"未知消息类型: {message_type}")
                
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
            
            logger.info(f"任务 {task_data['task_id']} 处理完成")
            
        except Exception as e:
            logger.error(f"处理任务开始错误: {e}")
    
    def _handle_task_pause(self, task_data: Dict[str, Any]):
        """处理任务暂停消息"""
        logger.info(f"暂停任务: {task_data['task_id']}")
        self.task_manager.pause_task(task_data['task_id'])
        self._notify_devices_task_pause(task_data)
    
    def _handle_task_resume(self, task_data: Dict[str, Any]):
        """处理任务恢复消息"""
        logger.info(f"恢复任务: {task_data['task_id']}")
        self.task_manager.resume_task(task_data['task_id'])
        self._notify_devices_task_resume(task_data)
    
    def _notify_devices_task_start(self, task_data: Dict[str, Any]):
        """通知边缘设备任务开始"""
        topic = self.config.get_mqtt_wildcard_topic('task_start')
        message = {
            "action": "task_start",
            "task_id": task_data['task_id'],
            "task_name": task_data['task_name'],
            "model_info": task_data['model_info'],
            "model_version": task_data['model_version'],
            "rounds": task_data['rounds'],
            "aggregation_method": task_data['aggregation_method']
        }
        
        self.mqtt_client.publish(topic, json.dumps(message))
        logger.info(f"已通知设备任务开始: {task_data['task_id']}")
    
    def _notify_devices_task_pause(self, task_data: Dict[str, Any]):
        """通知边缘设备任务暂停"""
        topic = self.config.get_mqtt_wildcard_topic('task_pause')
        message = {
            "action": "task_pause",
            "task_id": task_data['task_id']
        }
        
        self.mqtt_client.publish(topic, json.dumps(message))
        logger.info(f"已通知设备任务暂停: {task_data['task_id']}")
    
    def _notify_devices_task_resume(self, task_data: Dict[str, Any]):
        """通知边缘设备任务恢复"""
        topic = self.config.get_mqtt_wildcard_topic('task_resume')
        message = {
            "action": "task_resume",
            "task_id": task_data['task_id']
        }
        
        self.mqtt_client.publish(topic, json.dumps(message))
        logger.info(f"已通知设备任务恢复: {task_data['task_id']}")
    
    def _main_loop(self):
        """主循环"""
        logger.info("进入主循环...")
        
        while self.running:
            try:
                # 检查任务状态
                self._check_task_status()
                
                # 检查设备状态
                self._check_device_status()
                
                # 休眠
                time.sleep(self.config.task['status_check_interval'])
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(5)
    
    def _check_task_status(self):
        """检查任务状态"""
        # 实现任务状态检查逻辑
        pass
    
    def _check_device_status(self):
        """检查设备状态"""
        # 实现设备状态检查逻辑
        pass
    
    def stop(self):
        """停止服务"""
        logger.info("正在停止区域节点服务...")
        self.running = False
        
        # 关闭连接
        if hasattr(self, 'rabbitmq_client'):
            self.rabbitmq_client.close()
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.close()
        
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
