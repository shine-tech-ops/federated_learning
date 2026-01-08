#!/usr/bin/env python3
"""
设备客户端示例
演示如何创建和运行一个联邦学习设备客户端
"""

import os
import sys
import time
from loguru import logger

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from main import EdgeDevice
from config import Config


def create_example_device(
    device_id: str = "example_device_001",
    region_id: int = 3,
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
    mqtt_user: str = "mqtt",
    mqtt_password: str = "mqtt2024#",
    central_server_url: str = "http://localhost:8085",
    heartbeat_interval: int = 30
):
    """
    创建一个示例设备客户端
    
    Args:
        device_id: 设备唯一标识
        region_id: 区域节点ID
        mqtt_host: MQTT Broker 地址
        mqtt_port: MQTT Broker 端口
        mqtt_user: MQTT 用户名
        mqtt_password: MQTT 密码
        central_server_url: 中央服务器地址
        heartbeat_interval: 心跳间隔（秒）
    
    Returns:
        EdgeDevice: 配置好的设备实例
    """
    # MQTT 配置
    mqtt_config = {
        "host": mqtt_host,
        "port": mqtt_port,
        "username": mqtt_user,
        "password": mqtt_password,
        "keepalive": 60,
    }
    
    # HTTP 配置（用于发送心跳到中央服务器）
    http_config = {
        "base_url": central_server_url,
        "timeout": 10,
    }
    
    # 创建设备实例
    device = EdgeDevice(
        device_id=device_id,
        mqtt_config=mqtt_config,
        http_config=http_config,
        heartbeat_interval=heartbeat_interval
    )
    
    # 设置区域节点ID
    device.region_id = region_id
    
    return device


def run_example_device():
    """运行示例设备客户端"""
    
    # 配置日志
    log_file = "logs/example_device.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        level="INFO",
        encoding="utf-8",
    )
    
    logger.info("=" * 60)
    logger.info("启动示例设备客户端")
    logger.info("=" * 60)
    
    # 方式 1: 使用默认配置创建设备
    logger.info("\n方式 1: 使用默认配置")
    device1 = create_example_device()
    logger.info(f"设备ID: {device1.device_id}")
    logger.info(f"区域ID: {device1.region_id}")
    logger.info(f"MQTT Broker: {device1.mqtt_handler.host}:{device1.mqtt_handler.port}")
    logger.info(f"中央服务器: {device1.http_client.base_url}")
    
    # 方式 2: 使用自定义配置创建设备
    logger.info("\n方式 2: 使用自定义配置")
    device2 = create_example_device(
        device_id="custom_device_002",
        region_id=1,
        mqtt_host="192.168.1.100",
        mqtt_port=1883,
        central_server_url="http://192.168.1.10:8085",
        heartbeat_interval=60
    )
    logger.info(f"设备ID: {device2.device_id}")
    logger.info(f"区域ID: {device2.region_id}")
    logger.info(f"MQTT Broker: {device2.mqtt_handler.host}:{device2.mqtt_handler.port}")
    logger.info(f"中央服务器: {device2.http_client.base_url}")
    logger.info(f"心跳间隔: {device2.heartbeat_interval}秒")
    
    # 方式 3: 从环境变量或命令行参数创建
    logger.info("\n方式 3: 从命令行参数创建")
    if len(sys.argv) >= 2:
        device_id = sys.argv[1]
        region_id = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        central_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8085"
        
        device3 = create_example_device(
            device_id=device_id,
            region_id=region_id,
            central_server_url=central_url
        )
        
        logger.info(f"从命令行参数创建设备: {device3.device_id}")
        logger.info("启动设备...")
        
        # 启动设备（这会阻塞，直到收到停止信号）
        try:
            device3.start()
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        except Exception as e:
            logger.error(f"设备运行错误: {e}")
        finally:
            device3.stop()
    else:
        logger.info("未提供命令行参数，跳过方式 3")
        logger.info("使用方法: python example_client.py <device_id> [region_id] [central_server_url]")
        logger.info("示例: python example_client.py device_001 3 http://localhost:8085")


def example_minimal_client():
    """
    最小化示例：创建一个最简单的设备客户端
    """
    logger.info("\n" + "=" * 60)
    logger.info("最小化示例")
    logger.info("=" * 60)
    
    # 最简单的配置
    device = EdgeDevice(
        device_id="minimal_device",
        mqtt_config={
            "host": "localhost",
            "port": 1883,
            "username": "mqtt",
            "password": "mqtt2024#",
            "keepalive": 60,
        },
        http_config={
            "base_url": "http://localhost:8085",
            "timeout": 10,
        }
    )
    device.region_id = 3
    
    logger.info(f"创建最小化设备: {device.device_id}")
    logger.info("设备已创建，可以调用 device.start() 启动")
    
    return device


def example_with_custom_callbacks():
    """
    示例：使用自定义回调函数
    """
    logger.info("\n" + "=" * 60)
    logger.info("自定义回调示例")
    logger.info("=" * 60)
    
    def custom_message_handler(topic: str, message: dict):
        """自定义 MQTT 消息处理器"""
        logger.info(f"收到自定义消息: {topic}")
        logger.info(f"消息内容: {message}")
        
        # 可以在这里添加自定义处理逻辑
        action = message.get('action')
        if action == 'custom_action':
            logger.info("处理自定义动作")
    
    def custom_log_handler(log_data: dict):
        """自定义日志处理器"""
        logger.info(f"训练日志: {log_data}")
        # 可以在这里添加自定义日志处理逻辑，比如发送到其他系统
    
    # 创建设备
    device = create_example_device(device_id="callback_device")
    
    # 设置自定义消息回调
    device.mqtt_handler.set_message_callback(custom_message_handler)
    
    # 注意：训练日志回调需要在创建 FlowerClient 时设置
    # 这里只是演示如何自定义回调
    
    logger.info("设备已配置自定义回调")
    return device


def example_multiple_devices():
    """
    示例：创建多个设备客户端（用于测试）
    """
    logger.info("\n" + "=" * 60)
    logger.info("多设备示例")
    logger.info("=" * 60)
    
    devices = []
    
    # 创建 3 个设备
    for i in range(1, 4):
        device_id = f"test_device_{i:03d}"
        device = create_example_device(
            device_id=device_id,
            region_id=3,
            heartbeat_interval=30
        )
        devices.append(device)
        logger.info(f"创建设备: {device_id}")
    
    logger.info(f"共创建 {len(devices)} 个设备")
    logger.info("注意：实际运行中，每个设备应该在独立的进程中运行")
    
    return devices


if __name__ == "__main__":
    # 配置控制台日志
    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # 运行主示例
    run_example_device()
    
    # 运行其他示例（取消注释以运行）
    # example_minimal_client()
    # example_with_custom_callbacks()
    # example_multiple_devices()




