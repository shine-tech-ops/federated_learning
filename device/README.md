# Device 客户端分析文档

## 📋 目录结构

```
device/
├── main.py              # 边缘设备主程序（EdgeDevice 类）
├── mqtt_handler.py      # MQTT 消息处理器
├── flower_client.py     # Flower 联邦学习客户端
├── mnist_trainer.py    # MNIST 模型训练器
├── mnist_model.py      # MNIST 模型定义
├── http_client.py      # HTTP 客户端（心跳和日志上传）
├── config.py           # 配置文件
├── start_device.py     # 设备启动脚本
├── requirements.txt    # Python 依赖
└── data/              # 训练数据目录
```

## 🏗️ 架构分析

### 核心组件

#### 1. **EdgeDevice** (`main.py`)
- **职责**: 设备主控制器，协调所有组件
- **功能**:
  - 管理 MQTT 连接和消息处理
  - 处理任务生命周期（启动/暂停/恢复/停止）
  - 管理 Flower 客户端连接
  - 定期发送心跳到中央服务器
  - 上传训练日志

#### 2. **MQTTHandler** (`mqtt_handler.py`)
- **职责**: MQTT 通信处理
- **功能**:
  - 连接到 MQTT Broker
  - 订阅设备专属命令主题: `federated_task_{device_id}/task_start`
  - 处理任务控制消息（task_start, task_pause, task_resume, task_stop）
  - 发布设备状态和训练结果

#### 3. **FlowerClient** (`flower_client.py`)
- **职责**: Flower 联邦学习客户端
- **功能**:
  - 实现 `NumPyClient` 接口
  - 连接到区域节点的 Flower 服务器
  - 执行本地训练（`fit` 方法）
  - 执行模型评估（`evaluate` 方法）
  - 上报训练日志

#### 4. **MNISTTrainer** (`mnist_trainer.py`)
- **职责**: MNIST 模型训练逻辑
- **功能**:
  - 准备训练数据（根据设备ID分配不同数据子集，模拟数据异构性）
  - 执行本地训练（3个epoch）
  - 模型评估
  - 保存模型和参数

#### 5. **HTTPClient** (`http_client.py`)
- **职责**: 与中央服务器通信
- **功能**:
  - 发送设备心跳: `POST /api/v1/learn_management/device/heartbeat/`
  - 上传训练日志: `POST /api/v1/learn_management/training_log/`

## 🔄 工作流程

```
1. 设备启动
   ├─> 连接 MQTT Broker
   ├─> 订阅设备专属命令主题
   └─> 进入主循环（发送心跳）

2. 接收任务命令（通过 MQTT）
   ├─> task_start: 启动联邦学习
   │   ├─> 创建 MNISTTrainer
   │   ├─> 创建 FlowerClient
   │   └─> 连接到区域节点的 Flower 服务器
   ├─> task_pause: 暂停训练
   ├─> task_resume: 恢复训练
   └─> task_stop: 停止训练

3. 联邦学习过程
   ├─> Flower 服务器发送全局模型参数
   ├─> 设备本地训练（fit）
   │   ├─> 加载全局参数
   │   ├─> 本地训练 3 个 epoch
   │   ├─> 计算训练指标
   │   └─> 保存模型
   ├─> 设备评估（evaluate）
   │   ├─> 加载全局参数
   │   └─> 在测试集上评估
   └─> 返回更新后的参数和指标

4. 日志和心跳
   ├─> 训练/评估日志 → HTTP 上传到中央服务器
   └─> 定期心跳 → HTTP 发送到中央服务器
```

## ⚙️ 配置说明

### 环境变量配置（`.env` 文件）

```env
# 设备标识
DEVICE_ID=device_001
REGION_ID=3

# MQTT 配置
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_USER=mqtt
MQTT_PASSWORD=mqtt2024#
MQTT_KEEPALIVE=60

# 中央服务器配置
CENTRAL_SERVER_URL=http://localhost:8085
HTTP_TIMEOUT=10

# 心跳间隔（秒）
HEARTBEAT_INTERVAL=30

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/device_{device_id}.log
LOG_FORMAT={time:YYYY-MM-DD HH:mm:ss} | {level} | {message}
LOG_MAX_SIZE=10 MB
```

### 命令行参数

```bash
python main.py [device_id] [region_id] [central_server_url]
```

示例:
```bash
python main.py device_001 3 http://localhost:8085
```

## 📡 MQTT 主题

### 订阅主题
- `federated_task_{device_id}/task_start` - 接收任务启动命令

### 发布主题
- `region/1/devices/{device_id}/status` - 设备状态
- `region/1/devices/{device_id}/result` - 训练结果

## 🔌 API 接口

### HTTP 心跳接口
```
POST /api/v1/learn_management/device/heartbeat/
Body: {
    "device_id": "device_001",
    "region_node": 3,
    "device_context": {
        "status": "online",
        "timestamp": 1234567890,
        "current_task": "task_123"
    },
    "status": "online"
}
```

### HTTP 训练日志接口
```
POST /api/v1/learn_management/training_log/
Body: {
    "task": "task_123",
    "region_node": 3,
    "device_id": "device_001",
    "round": 1,
    "phase": "train",
    "level": "INFO",
    "loss": 0.5,
    "accuracy": 0.95,
    "num_examples": 1000,
    "metrics": {...},
    "message": "...",
    "duration": 10.5
}
```

## 🎯 数据异构性模拟

设备根据 `device_id` 的哈希值分配不同的数据子集：
- **设备组 0**: 数字 0, 1, 2
- **设备组 1**: 数字 3, 4, 5
- **设备组 2**: 数字 6, 7, 8, 9

每个设备最多使用 1000 个样本，模拟真实场景中的数据限制。

## 🚀 启动方式

### 方式 1: 使用启动脚本
```bash
python start_device.py device_001 3 http://localhost:8085
```

### 方式 2: 直接运行主程序
```bash
python main.py device_001 3 http://localhost:8085
```

### 方式 3: 使用配置文件
创建 `.env` 文件，然后运行：
```bash
python main.py
```

## 📝 依赖项

- `torch==2.0.1` - PyTorch 深度学习框架
- `torchvision==0.15.2` - 计算机视觉工具
- `flwr==1.8.0` - Flower 联邦学习框架
- `paho-mqtt==2.1.0` - MQTT 客户端
- `numpy==1.24.3` - 数值计算
- `loguru==0.7.2` - 日志库
- `requests==2.31.0` - HTTP 客户端
- `python-dotenv==1.0.1` - 环境变量管理

## 🔍 关键特性

1. **异步任务处理**: 使用后台线程执行联邦学习，不阻塞主循环
2. **任务控制**: 支持任务的启动、暂停、恢复、停止
3. **日志上报**: 自动上传训练和评估日志到中央服务器
4. **心跳机制**: 定期发送心跳，保持设备在线状态
5. **数据异构性**: 模拟不同设备拥有不同数据分布的场景
6. **模型保存**: 训练后自动保存模型和参数

## 🐛 故障排查

1. **MQTT 连接失败**: 检查 MQTT Broker 地址和认证信息
2. **Flower 连接失败**: 检查区域节点 Flower 服务器地址
3. **HTTP 请求失败**: 检查中央服务器地址和网络连接
4. **数据加载失败**: 确保 `data/MNIST` 目录存在，或允许自动下载

