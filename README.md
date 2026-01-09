# Federated Learning Deployment Guide

## System Requirements
- ARM64 architecture
- Docker and docker-compose installed
- Ensure the following ports are not in use:
  - 5432 (PostgreSQL)
  - 8085 (Backend)
  - 8086 (Frontend)
  - 6379 (Redis)
  - 1883, 9001 (MQTT)
  - 5672, 15672 (RabbitMQ)

## Deployment Steps

1. Load Docker images:
```bash
docker load -i images.tar
```

2. Start services:
```bash
docker-compose up -d
```

## Service Access
- Frontend Interface: http://localhost:8086
- RabbitMQ Management Interface: http://localhost:15672
  - Username: rabbitmq
  - Password: rabbitmq

## Directory Structure
```
federated_learning_deploy/
├── data/
│   └── db/                # PostgreSQL data
├── redis/
│   ├── data/             # Redis data
│   └── conf/             # Redis configuration
├── mosquitto/
│   ├── data/             # MQTT data
│   └── log/              # MQTT logs
├── rabbitmq/
│   └── data/             # RabbitMQ data
├── docker-compose.yml    # Docker compose file
├── mosquitto.conf        # MQTT configuration
├── mosquitto.passwd      # MQTT password file
└── images.tar           # Docker image archive
```

## Notes
1. All data will be persisted in the corresponding directories
2. To modify configurations, edit the respective configuration files
3. To restart services:
```bash
docker-compose restart
```
4. To stop services:
```bash
docker-compose down
```


## Features

1. Network Connection
- Device online status
- Mobile client online status

2. Model Download/Upload
- Automatic upload
- Download via URL

3. Task Execution
   1. CNN tasks - automatically collect models after completion
   2. Proxy model tasks - different execution flow
   3. Large model tasks - execute large models and upload desensitized parameters to central server
