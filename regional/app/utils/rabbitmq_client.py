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
    
    def delete_exchange(self, exchange: str) -> bool:
        """删除 Exchange（需要管理员权限）"""
        if not self.connection or self.connection.is_closed:
            self.connect()
        
        try:
            self.channel.exchange_delete(exchange=exchange, if_unused=False)
            logger.info(f"成功删除 Exchange: {exchange}")
            return True
        except pika.exceptions.ChannelClosedByBroker as e:
            if "ACCESS_REFUSED" in str(e) or "access refused" in str(e).lower():
                logger.warning(f"没有权限删除 Exchange '{exchange}'，需要管理员权限")
                return False
            logger.error(f"删除 Exchange '{exchange}' 失败: {e}")
            return False
        except Exception as e:
            logger.error(f"删除 Exchange '{exchange}' 失败: {e}")
            return False
    
    def consumer(self, exchange: str, queue: str, callback: Callable, routing_key: str = "task"):
        """
        消费消息 - 自动创建 Exchange 如果不存在
        
        Args:
            exchange: Exchange 名称
            queue: 队列名称
            callback: 消息回调函数
            routing_key: 路由键，默认为 "task"，必须与发布消息时使用的 routing_key 一致
        """
        if not self.connection or self.connection.is_closed:
            self.connect()
        
        try:
            # 声明 Exchange，类型必须与后端保持一致（direct）
            # 后端在 backend/utils/rabbitmq_client.py 中使用 exchange_type="direct"
            try:
                self.channel.exchange_declare(
                    exchange=exchange,
                    exchange_type="direct",
                    durable=True
                )
                logger.debug(f"Exchange '{exchange}' 声明成功（类型: direct, routing_key: {routing_key}）")
            except pika.exceptions.ChannelClosedByBroker as e:
                error_str = str(e)
                # 检查是否是类型或 durable 属性不匹配
                if "inequivalent arg" in error_str:
                    if "type" in error_str:
                        logger.warning(
                            f"Exchange '{exchange}' 已存在但类型不匹配，尝试自动删除并重建..."
                        )
                    elif "durable" in error_str:
                        logger.warning(
                            f"Exchange '{exchange}' 已存在但 durable 属性不匹配，尝试自动删除并重建..."
                        )
                    else:
                        logger.warning(
                            f"Exchange '{exchange}' 已存在但属性不匹配，尝试自动删除并重建..."
                        )
                    
                    # 尝试删除旧的 exchange
                    if self.delete_exchange(exchange):
                        # 重新连接（因为删除操作可能关闭了 channel）
                        if self.connection.is_closed:
                            self.connect()
                        # 重新声明 exchange
                        self.channel.exchange_declare(
                            exchange=exchange,
                            exchange_type="direct",
                            durable=True
                        )
                        logger.info(f"Exchange '{exchange}' 已成功删除并重建（类型: direct, durable: True）")
                    else:
                        # 如果删除失败，提供详细的手动删除指南
                        logger.error(
                            f"\n{'='*60}\n"
                            f"无法自动删除 Exchange '{exchange}'（权限不足）\n"
                            f"请使用以下方法之一手动删除：\n\n"
                            f"方法 1: 通过 RabbitMQ 管理界面删除\n"
                            f"  1. 访问 http://localhost:15672\n"
                            f"  2. 登录（用户名: {self.username}）\n"
                            f"  3. 进入 Exchanges 页面\n"
                            f"  4. 找到 '{exchange}' 并删除\n\n"
                            f"方法 2: 使用 rabbitmqctl 命令（需要管理员权限）\n"
                            f"  docker exec -it <rabbitmq_container> rabbitmqctl delete_exchange {exchange}\n\n"
                            f"方法 3: 使用 Python 脚本（需要管理员权限）\n"
                            f"  运行: python -c \"import pika; c=pika.BlockingConnection("
                            f"pika.ConnectionParameters(host='{self.host}', "
                            f"credentials=pika.PlainCredentials('{self.username}', '{self.password}'))); "
                            f"c.channel().exchange_delete('{exchange}')\"\n"
                            f"{'='*60}\n"
                        )
                        raise
                else:
                    raise
            # 声明队列
            result = self.channel.queue_declare(
                queue=queue,
                durable=True
            )
            queue_name = result.method.queue
            
            # 使用 routing_key 绑定队列到 exchange（direct 类型必须指定 routing_key）
            self.channel.queue_bind(
                exchange=exchange,
                queue=queue_name,
                routing_key=routing_key
            )
            logger.debug(f"队列 '{queue_name}' 已绑定到 Exchange '{exchange}' (routing_key: {routing_key})")
            
            # 设置消费者
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=False
            )
            
            
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
