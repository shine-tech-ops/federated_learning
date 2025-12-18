#!/usr/bin/env python3
"""
设备启动脚本
"""

import sys
import os
from utils import Utils
from loguru import logger
from main import EdgeDevice

def main():
    config = Utils.load_config()
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

    print("=" * 50)
    print("设备启动配置")
    print("=" * 50)
    print(f"设备 ID: {device_id}")
    print(f"区域 ID: {region_id}")
    print(f"MQTT 服务器: {mqtt_config['host']}:{mqtt_config['port']}")
    print(f"HTTP 服务器: {http_config['base_url']}")
    print("=" * 50)

    # 初始化工具类
    utils = Utils()
    success, msg = utils.update_device_info(
        device={"id": device_id},
        region={"id": region_id},
    )
    if not success:
        print("警告：更新失败，请检查工具类是否有误，设备启动已终止！")
    else:
        # 启动设备
        device = EdgeDevice(device_id, mqtt_config, http_config)
        device.region_id = region_id  # 设置区域节点ID
        device.start()

if __name__ == "__main__":
    main()
