#!/usr/bin/env python3
"""
设备启动脚本
"""

import sys
import os
from loguru import logger
from main import EdgeDevice
from config import config

def main():
    device_id = sys.argv[1] if len(sys.argv) > 1 else config.device_id
    region_id = int(sys.argv[2]) if len(sys.argv) > 2 else int(config.region_id)
    central_server_url = sys.argv[3] if len(sys.argv) > 3 else config.http['base_url']

    # 日志配置
    log_file = config.logging['file'].format(device_id=device_id)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        format=config.logging['format'],
        rotation=config.logging['max_size'],
        level=config.logging['level'],
        encoding="utf-8",
    )

    # MQTT 配置
    mqtt_config = config.mqtt
    
    # HTTP 配置（用于发送心跳到中央服务器）
    http_config = {
        **config.http,
        'base_url': central_server_url,
    }
    
    # 启动设备
    device = EdgeDevice(device_id, mqtt_config, http_config, heartbeat_interval=config.heartbeat_interval)
    device.region_id = region_id  # 设置区域节点ID
    device.start()

if __name__ == "__main__":
    main()
