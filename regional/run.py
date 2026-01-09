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
    
    
    # 启动区域节点
    node = RegionalNode()
    node.start()


if __name__ == "__main__":
    main()