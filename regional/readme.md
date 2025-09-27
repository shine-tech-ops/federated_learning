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
```bash
python run.py
```

## 📋 功能特性

### ✅ 已实现
- RabbitMQ 消息消费
- MQTT 设备通信
- 任务生命周期管理（内存）
- 设备状态监控

### 🔄 工作流程
1. **接收任务** - 从 RabbitMQ 接收联邦学习任务
2. **分发任务** - 通过 MQTT 通知边缘设备
3. **监控状态** - 跟踪设备和任务状态（内存）
4. **收集结果** - 接收设备训练结果

## 🐳 Docker 部署

```bash
# 使用 Docker Compose
docker-compose -f docker-compose-regional.yml up -d

# 查看日志
docker-compose -f docker-compose-regional.yml logs -f regional
```

## 📁 项目结构

```
regional/
├── regional_node.py          # 主服务类
├── run.py                    # 启动脚本
├── requirements.txt          # 依赖列表
├── app/
│   ├── service/
│   │   └── task_manager.py   # 任务管理器
│   └── utils/
│       ├── rabbitmq_client.py # RabbitMQ 客户端
│       └── mqtt_client.py     # MQTT 客户端
└── logs/                     # 日志目录
```

## 🔧 配置说明

### 环境变量
- `REGION_ID`: 区域节点唯一标识
- `RABBITMQ_*`: RabbitMQ 连接配置
- `MQTT_*`: MQTT Broker 连接配置

### 消息格式
- **任务消息**: JSON 格式，包含任务详情
- **设备消息**: JSON 格式，包含设备状态和训练结果
- **状态管理**: 通过内存管理任务和设备状态

## 🎯 优势

1. **简化架构** - 无 HTTP 接口，减少复杂性
2. **安全可靠** - 内网部署，降低安全风险
3. **高效通信** - 基于消息队列，异步处理
4. **易于扩展** - 模块化设计，便于功能扩展
5. **资源节约** - 轻量级服务，资源占用少