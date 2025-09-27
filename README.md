# Federated Learning 部署指南

## 系统要求
- ARM64 架构
- 安装 Docker 和 docker-compose
- 确保以下端口未被占用：
  - 5432 (PostgreSQL)
  - 8085 (Backend)
  - 8086 (Frontend)
  - 6379 (Redis)
  - 1883, 9001 (MQTT)
  - 5672, 15672 (RabbitMQ)

## 部署步骤

1. 加载 Docker 镜像：
```bash
docker load -i images.tar
```

2. 启动服务：
```bash
docker-compose up -d
```

## 服务访问
- 前端界面：http://localhost:8086
- RabbitMQ 管理界面：http://localhost:15672
  - 用户名：rabbitmq
  - 密码：rabbitmq

## 目录结构
```
federated_learning_deploy/
├── data/
│   └── db/                # PostgreSQL 数据
├── redis/
│   ├── data/             # Redis 数据
│   └── conf/             # Redis 配置
├── mosquitto/
│   ├── data/             # MQTT 数据
│   └── log/              # MQTT 日志
├── rabbitmq/
│   └── data/             # RabbitMQ 数据
├── docker-compose.yml    # Docker 编排文件
├── mosquitto.conf        # MQTT 配置
├── mosquitto.passwd      # MQTT 密码文件
└── images.tar           # Docker 镜像包
```

## 注意事项
1. 所有数据都会持久化存储在对应的目录中
2. 如需修改配置，请编辑对应的配置文件
3. 如需重启服务：
```bash
docker-compose restart
```
4. 如需停止服务：
```bash
docker-compose down
```