# System Architecture

## Architecture Overview

```
Central Server (Backend)
    ↓ RabbitMQ
Regional Node
    ↓ MQTT
Edge Devices
```

## Core Components

### 1. Central Server (Backend)

**Tech Stack**: Django + PostgreSQL + Redis + RabbitMQ + MinIO

**Responsibilities**:
- Manage federated learning tasks
- Store models and versions
- Manage regional nodes
- Provide REST API

**Key Services**:
- PostgreSQL: Data storage
- Redis: Caching
- RabbitMQ: Communication with regional nodes
- MinIO: Model file storage

### 2. Frontend Interface

**Tech Stack**: Vue 3 + TypeScript + Vite

**Features**:
- Task management interface
- Model management
- Training log viewing
- Node status monitoring

### 3. Regional Node

**Tech Stack**: Python + fedevo + RabbitMQ + MQTT

**Responsibilities**:
- Receive task instructions from central server
- Manage fedevo federated learning server
- Communicate with edge devices
- Report task status

**Communication Methods**:
- RabbitMQ: Communication with central server
- MQTT: Communication with edge devices
- HTTP: Status reporting to central server

### 4. Edge Device

**Tech Stack**: Python + PyTorch + fedevo Client + MQTT

**Responsibilities**:
- Listen to MQTT task instructions
- Execute local model training
- Upload model updates
- Report training status

## Data Flow

### Task Startup Flow

1. Central server creates task → RabbitMQ
2. Regional node receives task → Starts fedevo server
3. Regional node sends task to devices via MQTT
4. Devices connect to fedevo server → Start training
5. Devices complete training → Upload model updates
6. fedevo server aggregates models
7. Regional node reports results to central server

### Model Flow

1. Central server → MinIO → Regional node downloads
2. Regional node → MQTT → Devices download
3. Devices train → fedevo server aggregates
4. Aggregated model → Regional node → Central server storage

## Network Architecture

### Port Allocation

- 5432: PostgreSQL
- 8085: Backend API
- 8086: Frontend
- 6379: Redis
- 5672: RabbitMQ
- 15672: RabbitMQ Management UI
- 1883, 9001: MQTT
- 9000, 9001: MinIO

### Communication Protocols

- **HTTP/REST**: Frontend to backend, regional node status reporting
- **RabbitMQ**: Central server to regional nodes
- **MQTT**: Regional nodes to edge devices
- **fedevo gRPC**: Regional nodes to edge devices (federated learning)

## Model Architecture

### Supported Frameworks

- **PyTorch**: Primary framework
- **TensorFlow**: Supported (to be improved)
- **ONNX**: Cross-platform inference

### Aggregation Algorithms

- **FedAvg**: Simple average aggregation
- **Weighted Aggregation**: Weighted by data size
- **Custom Aggregation**: Extensible

## Storage Architecture

- **PostgreSQL**: Tasks, model metadata, user data
- **MinIO**: Model file storage
- **Redis**: Cache and sessions
- **Local Files**: Device-side training data
