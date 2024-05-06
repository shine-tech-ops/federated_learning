#!/bin/bash

# 脚本：启动整个分层联邦学习模拟

echo "--- 正在清理旧进程 ---"
# 强制结束所有可能在运行的旧Python进程
pkill -f registration_service.py
pkill -f main_server.py
pkill -f regional_server.py
pkill -f client.py
sleep 2

echo "--- 1. 启动注册服务 ---"
python registration_service.py &
REG_PID=$!
sleep 2 # 等待服务启动

echo "--- 2. 启动中央服务器 ---"
python main_server.py &
MAIN_SERVER_PID=$!
sleep 2 # 等待服务器启动

echo "--- 3. 启动区域服务器 (Region-A) ---"
python regional_server.py --region-name="Region-A" --listen-address="[::]:8081" &
REGION_A_PID=$!
sleep 2

# 你可以启动更多区域服务器
# echo "--- 启动区域服务器 (Region-B) ---"
# python regional_server.py --region-name="Region-B" --listen-address="[::]:8082" &
# REGION_B_PID=$!
# sleep 2

echo "--- 4. 启动边缘设备客户端 ---"
# 启动2个客户端连接到 Region-A
python client.py --client-id=101 --region="Region-A" --regional-server-address="[::]:8081" &
CLIENT_1_PID=$!
python client.py --client-id=102 --region="Region-A" --regional-server-address="[::]:8081" &
CLIENT_2_PID=$!

# 启动1个客户端连接到 Region-B (如果启动了Region-B服务器)
# python client.py --client-id=201 --region="Region-B" --regional-server-address="[::]:8082" &
# CLIENT_3_PID=$!

echo "--- 模拟正在运行... ---"
echo "所有组件已启动。请观察各个终端的日志输出。"
echo "按 [CTRL+C] 结束模拟并清理所有进程。"

# 捕获CTRL+C信号并执行清理
trap "echo '--- 正在停止所有服务... ---'; kill $REG_PID $MAIN_SERVER_PID $REGION_A_PID $CLIENT_1_PID $CLIENT_2_PID; exit" INT

# 等待所有后台进程，否则脚本会立即退出
wait