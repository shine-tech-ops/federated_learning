#!/usr/bin/env python3
"""
设备启动脚本
"""

import sys
import os
from main import EdgeDevice

def main():
    if len(sys.argv) < 2:
        print("用法: python start_device.py <device_id> [region_id] [central_server_url]")
        print("示例: python start_device.py device_001 1 http://localhost:8085")
        sys.exit(1)
    
    device_id = sys.argv[1]
    region_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # 默认region_id为1
  
    # MQTT 配置
    mqtt_config = {
        'host': 'localhost',
        'port': 1883,
        'username': 'mqtt',
        'password': 'mqtt2024#',
        'keepalive': 60
    }
    
    # HTTP 配置（用于发送心跳到中央服务器）
    http_config = {
        'base_url':'http://localhost:8085'
    }
    
    # 启动设备
    device = EdgeDevice(device_id, mqtt_config, http_config)
    device.region_id = region_id  # 设置区域节点ID
    device.start()

if __name__ == "__main__":
    main()
