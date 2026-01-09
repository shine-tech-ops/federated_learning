#!/usr/bin/env python3
"""
联邦学习测试脚本
"""

import subprocess
import time
import sys
import os

def start_services():
    """启动所有服务"""
    print("🚀 启动联邦学习测试...")
    
    # 1. 启动区域节点
    print("1. 启动区域节点...")
    regional_process = subprocess.Popen([
        sys.executable, "regional/regional_node.py"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    time.sleep(3)
    
    # 2. 启动设备1
    print("2. 启动设备1...")
    device1_process = subprocess.Popen([
        sys.executable, "device/start_device.py", "device_001"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    time.sleep(2)
    
    # 3. 启动设备2
    print("3. 启动设备2...")
    device2_process = subprocess.Popen([
        sys.executable, "device/start_device.py", "device_002"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    time.sleep(2)
    
    # 4. 启动设备3
    print("4. 启动设备3...")
    device3_process = subprocess.Popen([
        sys.executable, "device/start_device.py", "device_003"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("✅ 所有服务已启动")
    print("📝 现在可以通过中央服务器界面启动联邦学习任务")
    print("🛑 按 Ctrl+C 停止所有服务")
    
    try:
        # 等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止所有服务...")
        
        # 停止所有进程
        regional_process.terminate()
        device1_process.terminate()
        device2_process.terminate()
        device3_process.terminate()
        
        # 等待进程结束
        regional_process.wait()
        device1_process.wait()
        device2_process.wait()
        device3_process.wait()
        
        print("✅ 所有服务已停止")

if __name__ == "__main__":
    start_services()
