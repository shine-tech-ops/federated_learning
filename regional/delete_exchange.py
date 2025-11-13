#!/usr/bin/env python3
"""
删除 RabbitMQ Exchange 工具脚本
用于删除类型不匹配的 exchange
"""

import sys
import pika
from loguru import logger
from config import config

def delete_exchange(exchange_name: str):
    """删除指定的 exchange"""
    try:
        # 连接到 RabbitMQ
        credentials = pika.PlainCredentials(
            config.rabbitmq['username'],
            config.rabbitmq['password']
        )
        parameters = pika.ConnectionParameters(
            host=config.rabbitmq['host'],
            port=config.rabbitmq['port'],
            credentials=credentials,
            virtual_host=config.rabbitmq['virtual_host']
        )
        
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        logger.info(f"正在删除 Exchange: {exchange_name}")
        
        # 删除 exchange
        channel.exchange_delete(exchange=exchange_name, if_unused=False)
        
        logger.info(f"✅ 成功删除 Exchange: {exchange_name}")
        
        connection.close()
        return True
        
    except pika.exceptions.ChannelClosedByBroker as e:
        if "ACCESS_REFUSED" in str(e) or "access refused" in str(e).lower():
            logger.error(
                f"❌ 权限不足，无法删除 Exchange '{exchange_name}'\n"
                f"请确保用户 '{config.rabbitmq['username']}' 具有管理员权限\n\n"
                f"解决方法：\n"
                f"1. 通过 RabbitMQ 管理界面删除: http://localhost:15672\n"
                f"2. 使用管理员账户运行此脚本\n"
                f"3. 使用 docker exec 命令: docker exec -it <rabbitmq_container> rabbitmqctl delete_exchange {exchange_name}"
            )
        else:
            logger.error(f"❌ 删除失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 删除 Exchange 时发生错误: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python delete_exchange.py <exchange_name>")
        print("示例: python delete_exchange.py federated_task_region_3")
        sys.exit(1)
    
    exchange_name = sys.argv[1]
    success = delete_exchange(exchange_name)
    sys.exit(0 if success else 1)

