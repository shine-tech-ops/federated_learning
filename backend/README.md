# 联邦学习平台部署指南

本文档提供了联邦学习平台的完整部署指南。该平台包含前端、后端、数据库以及多个中间件服务。

## 系统要求

### 基础环境要求
- Docker Engine 24.0.0+
- Docker Compose v2.0.0+
- 至少 8GB RAM
- 至少 20GB 可用磁盘空间

### 端口要求
确保以下端口未被占用：
- 5432: PostgreSQL 数据库
- 8085: 后端服务
- 8086: 前端服务
- 6379: Redis
- 1883: MQTT Broker
- 9001: MQTT Websocket
- 5672: RabbitMQ
- 15672: RabbitMQ 管理界面

## 快速部署

1. 克隆代码仓库：
```bash
git clone [repository_url]
cd federated_learning
```

2. 配置环境变量（可选）：
如果需要自定义配置，可以修改以下文件：
- `backend/conf/env.py`：后端环境配置
- `frontend/.env`：前端环境配置
- `mosquitto.conf`：MQTT 配置
- `redis/redis.conf`：Redis 配置

3. 启动所有服务：
```bash
docker-compose up -d
```

4. 验证服务状态：
```bash
docker-compose ps
```

## 详细配置说明

### 1. 数据库配置
使用 TimescaleDB（PostgreSQL 扩展版）作为主数据库：
- 数据库名：backend
- 默认用户：postgres
- 默认密码：postgres
- 数据持久化目录：./data/db

### 2. 后端服务配置
- 框架：Django 5.0.6
- API 文档访问地址：http://localhost:8085/swagger/
- 默认 API Token：fNKZrhZlRGdqBCKZkgvpENkRIsorvRGBHUMoMEtobuU

### 3. 前端服务配置
- 框架：Vue 3 + TypeScript + Vite
- 访问地址：http://localhost:8086
- 构建命令：pnpm build

### 4. 中间件服务

#### Redis 配置
- 版本：7.2.5
- 持久化配置：./redis/redis.conf
- 数据目录：./redis/data

#### MQTT 配置
- 版本：Eclipse Mosquitto 2.0.18
- 配置文件：./mosquitto.conf
- 认证文件：./mosquitto.passwd
- 数据目录：./mosquitto/data
- 日志目录：./mosquitto/log

#### RabbitMQ 配置
- 版本：3-management
- 默认用户：rabbitmq
- 默认密码：rabbitmq
- 管理界面：http://localhost:15672
- 数据目录：./rabbitmq/data

## 维护指南

### 日志查看
```bash
# 查看所有容器日志
docker-compose logs

# 查看特定服务日志
docker-compose logs [service_name]
```

### 数据备份
1. 数据库备份：
```bash
docker-compose exec db pg_dump -U postgres backend > backup.sql
```

2. 配置文件备份：
建议定期备份以下目录：
- ./data
- ./redis/data
- ./mosquitto/data
- ./rabbitmq/data

### 更新部署
1. 拉取最新代码：
```bash
git pull origin main
```

2. 重新构建并启动服务：
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

## 故障排除

### 常见问题

1. 服务无法启动
- 检查端口占用情况
- 检查磁盘空间
- 检查 Docker 日志

2. 数据库连接失败
- 确认数据库容器运行状态
- 检查数据库配置参数
- 验证数据库权限

3. 消息队列连接问题
- 检查 RabbitMQ 管理界面状态
- 验证 MQTT Broker 连接参数
- 确认网络连接状态

### 联系支持
如遇到无法解决的问题，请联系技术支持团队：
- 提交 Issue：[repository_url]/issues
- 技术支持邮箱：[support_email]

## 安全建议

1. 生产环境部署前：
- 修改所有默认密码
- 配置 SSL/TLS
- 限制管理界面访问
- 配置防火墙规则

2. 定期维护：
- 更新依赖包
- 检查安全日志
- 备份重要数据
- 监控系统资源

## 许可证
[License 信息]