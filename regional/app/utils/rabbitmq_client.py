"""
RabbitMQ 客户端 - 简化的消息队列客户端
"""

import pika
import json
import time
from loguru import logger
from typing import Callable, Optional


class RabbitMQClient:
    """RabbitMQ 客户端"""
    
    def __init__(self, config=None):
        if config:
            self.host = config.rabbitmq['host']
            self.port = config.rabbitmq['port']
            self.username = config.rabbitmq['username']
            self.password = config.rabbitmq['password']
            self.virtual_host = config.rabbitmq['virtual_host']
        else:
            # 兼容旧版本
            self.host = self._get_env('RABBITMQ_HOST', 'localhost')
            self.port = int(self._get_env('RABBITMQ_PORT', '5672'))
            self.username = self._get_env('RABBITMQ_USER', 'rabbitmq')
            self.password = self._get_env('RABBITMQ_PASSWORD', 'rabbitmq')
            self.virtual_host = self._get_env('RABBITMQ_VHOST', '/')
        
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
    
    def _get_env(self, key: str, default: str) -> str:
        """获取环境变量"""
        import os
        return os.environ.get(key, default)
    
    def connect(self):
        """连接到 RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                virtual_host=self.virtual_host
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            logger.info(f"已连接到 RabbitMQ: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"连接 RabbitMQ 失败: {e}")
            raise
    
    def consumer(self, exchange: str, queue: str, callback: Callable):
        """消费消息 - 自动创建 Exchange 如果不存在"""
        if not self.connection or self.connection.is_closed:
            self.connect()
        
        try:
            # 检查 Exchange 是否存在，如果不存在则自动创建
            try:
                self.channel.exchange_declare(
                    exchange=exchange,
                    passive=True  # 只检查是否存在
                )
                logger.info(f"Exchange {exchange} 已存在，开始消费")
            except pika.exceptions.AMQPChannelError:
                # Exchange 不存在，自动创建
                logger.info(f"Exchange {exchange} 不存在，正在自动创建...")
                self.channel.exchange_declare(
                    exchange=exchange,
                    exchange_type='fanout',  # 与中央服务器保持一致
                    durable=True
                )
                logger.info(f"✅ Exchange {exchange} 创建成功")
            
            # 声明队列
            result = self.channel.queue_declare(
                queue=queue,
                durable=True
            )
            queue_name = result.method.queue
            
            # 绑定队列到 exchange
            self.channel.queue_bind(
                exchange=exchange,
                queue=queue_name
            )
            
            # 设置消费者
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=False
            )
            
            logger.info(f"开始消费消息: {exchange} -> {queue_name}")
            
            # 开始消费
            self.channel.start_consuming()
            
        except Exception as e:
            logger.error(f"消费消息失败: {e}")
            raise
    
    def publish(self, exchange: str, message: dict):
        """发布消息 - 区域节点不通过 RabbitMQ 上报，此方法已废弃"""
        logger.warning("区域节点不应通过 RabbitMQ 发布消息，请使用 HTTP API 上报状态")
        raise NotImplementedError("区域节点应使用 HTTP API 上报状态，不通过 RabbitMQ")
    
    def close(self):
        """关闭连接"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.stop_consuming()
                self.channel.close()
            
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            
            logger.info("RabbitMQ 连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭 RabbitMQ 连接失败: {e}")
