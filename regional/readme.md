# Regional Node - 简化的区域节点

这是一个**简化的区域节点服务**，专门用于联邦学习系统。

## 🎯 设计理念

- **无 HTTP 接口** - 纯后台服务，不提供 REST API
- **基于消息队列** - 所有通信都通过 RabbitMQ 和 MQTT
- **内网部署** - 不需要公网 IP，部署在内网环境
- **简单高效** - 代码简洁，易于维护

## 🏗️ 架构设计

```
中央服务器 (Backend)
    ↓ RabbitMQ
区域节点 (Regional Node)
    ↓ MQTT
边缘设备 (Edge Devices)
```

### 核心组件

1. **RabbitMQ 消费者** - 接收中央服务器发送的任务
2. **MQTT 客户端** - 与边缘设备通信
3. **任务管理器** - 管理联邦学习任务的生命周期（内存管理）

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
- 修改 federated_learning/regional/config.py
```bash
export REGION_ID=region-001
export RABBITMQ_HOST=192.168.3.174
export RABBITMQ_PORT=5672
export RABBITMQ_USER=rabbitmq
export RABBITMQ_PASSWORD=rabbitmq
export MQTT_BROKER_HOST=mqtt
export MQTT_BROKER_PORT=1883
export MQTT_USER=mqtt
export MQTT_PASSWORD=mqtt2024#
```

### 3. 运行服务
pip install -r requirements.txt
```bash
python run.py
```
