#!/usr/bin/env python3
"""
测试脚本：发送任务数据到 RabbitMQ
"""

import json
import time
from datetime import datetime
from app.utils.rabbitmq_client import RabbitMQClient
from config import config

def send_test_task():
    """发送测试任务数据"""
    
    # 创建测试任务数据
    task_data = {
        "task_id": 999,
        "task_name": "测试任务",
        "description": "这是一个测试任务",
        "rounds": 5,
        "aggregation_method": "fedavg",
        "participation_rate": 50,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "region_node": {
            "id": 1,
            "name": "测试区域节点",
            "ip_address": "127.0.0.1",
            "description": "测试用区域节点",
        },
        "model_info": {
            "id": 1,
            "name": "测试模型",
            "description": "测试用模型",
        },
        "model_version": {
            "id": 1,
            "version": "v1.0",
            "model_file": "/path/to/model.pkl",
            "description": "测试模型版本",
            "accuracy": 0.95,
            "loss": 0.05,
            "metrics": {"precision": 0.94, "recall": 0.96},
        },
        "created_by": {
            "id": 1,
            "name": "测试用户",
        },
        "message_type": "federated_task_start",
        "timestamp": datetime.now().isoformat(),
    }
    
    # 创建 RabbitMQ 客户端
    rabbitmq_client = RabbitMQClient(config)
    
    try:
        # 连接到 RabbitMQ
        rabbitmq_client.connect()
        
        # 发送到 Regional Node 监听的 Exchange
        exchange_name = config.get_rabbitmq_exchange()  # federated_task_region-001
        queue_name = config.get_rabbitmq_queue()        # region_region-001_tasks
        
        print(f"🚀 发送测试任务数据...")
        print(f"📤 Exchange: {exchange_name}")
        print(f"📤 Queue: {queue_name}")
        print(f"📤 任务数据: {json.dumps(task_data, indent=2, ensure_ascii=False)}")
        
        # 发送消息
        rabbitmq_client.publish(exchange_name, task_data)
        
        print("✅ 任务数据发送成功！")
        print("🔍 请检查 Regional Node 是否收到并打印了数据")
        
    except Exception as e:
        print(f"❌ 发送失败: {e}")
    finally:
        rabbitmq_client.close()

if __name__ == "__main__":
    send_test_task()
