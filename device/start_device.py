#!/usr/bin/env python3
"""
设备启动脚本
"""

import sys
import os
from main import EdgeDevice

def main():
    if len(sys.argv) < 2:
        print("用法: python start_device.py <device_id>")
        print("示例: python start_device.py device_001")
        sys.exit(1)
    
    device_id = sys.argv[1]
    
    # MQTT 配置
    mqtt_config = {
        'host': 'localhost',
        'port': 1883,
        'username': 'mqtt',
        'password': 'mqtt2024#',
        'keepalive': 60
    }
    
    # 启动设备
    device = EdgeDevice(device_id, mqtt_config)
    device.start()

if __name__ == "__main__":
    main()
