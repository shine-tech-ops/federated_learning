#!/usr/bin/env python3
"""
Regional Node 启动脚本
简化的区域节点，只负责消息队列通信
"""

from regional_node import RegionalNode
from loguru import logger
import os


def main():
    """主函数"""
    from config import config
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 配置日志
    logger.add(
        config.logging['file'],
        format=config.logging['format'],
        rotation=config.logging['max_size'],
        level=config.logging['level'],
        encoding="utf-8",
    )
    
    # 显示配置信息
    logger.info(f"=== Regional Node 启动 ===")
    logger.info(f"区域ID: {config.region_id}")
    logger.info(f"节点名称: {config.node_name}")
    logger.info(f"RabbitMQ: {config.rabbitmq['host']}:{config.rabbitmq['port']}")
    logger.info(f"MQTT: {config.mqtt['host']}:{config.mqtt['port']}")
    logger.info(f"调试模式: {config.debug['enabled']}")
    
    # 启动区域节点
    node = RegionalNode()
    node.start()


if __name__ == "__main__":
    main()