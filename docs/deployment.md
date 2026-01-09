# Deployment Guide

## System Requirements

- ARM64 architecture
- Docker and docker-compose installed
- Ensure the following ports are available:
  - 5432 (PostgreSQL)
  - 8085 (Backend)
  - 8086 (Frontend)
  - 6379 (Redis)
  - 1883, 9001 (MQTT)
  - 5672, 15672 (RabbitMQ)
  - 9000, 9001 (MinIO)

## Quick Deployment

### 1. Load Docker Images

```bash
docker load -i images.tar
```

### 2. Start Services

```bash
docker-compose up -d
```

## Service Access

- Frontend: http://localhost:8086
- RabbitMQ Management: http://localhost:15672
  - Username: `rabbitmq`
  - Password: `rabbitmq`
- MinIO Console: http://localhost:9001
  - Username: `admin`
  - Password: `admin123456`

## Directory Structure

```
federated_learning/
├── data/
│   └── db/                # PostgreSQL data
├── redis/
│   ├── data/             # Redis data
│   └── redis.conf        # Redis config
├── mosquitto/
│   ├── data/             # MQTT data
│   └── log/              # MQTT logs
├── rabbitmq/
│   └── data/             # RabbitMQ data
├── minio/
│   ├── data/             # MinIO data
│   └── config/           # MinIO config
├── docker-compose.yml    # Docker orchestration
├── mosquitto.conf        # MQTT config
└── mosquitto.passwd      # MQTT password file
```

## Component Deployment

### Central Server (Backend)

The backend is automatically started via docker-compose. It includes:
- Database migrations on startup
- API server on port 8085

### Frontend

The frontend is automatically started via docker-compose on port 8086.

### Regional Node

Deploy regional node separately:

```bash
cd regional
pip install -r requirements.txt

# Configure environment variables
export REGION_ID=region-001
export RABBITMQ_HOST=192.168.3.174
export RABBITMQ_PORT=5672
export RABBITMQ_USER=rabbitmq
export RABBITMQ_PASSWORD=rabbitmq
export MQTT_BROKER_HOST=mqtt
export MQTT_BROKER_PORT=1883
export MQTT_USER=mqtt
export MQTT_PASSWORD=mqtt2024#

# Start regional node
python run.py
```

### Edge Device

Deploy edge device:

```bash
cd device
pip install -r requirements.txt

# Configure device/config.py or use environment variables

# Start device
python start_device.py <device_id> <region_id> <central_server_url>
```

## Operations

### Restart Services

```bash
docker-compose restart
```

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Database Migrations

Backend automatically runs migrations on startup. To run manually:

```bash
docker-compose exec backend python manage.py migrate
```

## Configuration

### Environment Variables

Key environment variables for backend (in docker-compose.yml):
- `POSTGRES_HOST`, `POSTGRES_NAME`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASSWORD`
- `MINIO_ENDPOINT`, `MINIO_PORT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`

### MQTT Configuration

Edit `mosquitto.conf` and `mosquitto.passwd` for MQTT authentication.

### Redis Configuration

Edit `redis/redis.conf` for Redis settings.

## Troubleshooting

### Port Conflicts

If ports are already in use, modify `docker-compose.yml` port mappings.

### Database Connection Issues

Ensure PostgreSQL is healthy before backend starts:
```bash
docker-compose ps db
```

### Service Health Checks

Check service status:
```bash
docker-compose ps
```

## Data Persistence

All data is persisted in the following directories:
- `data/db/` - PostgreSQL data
- `redis/data/` - Redis data
- `rabbitmq/data/` - RabbitMQ data
- `minio/data/` - MinIO data

Backup these directories for data backup.
