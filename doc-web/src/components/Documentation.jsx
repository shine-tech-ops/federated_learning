import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { marked } from 'marked'
import styles from './Documentation.module.css'
import ArchitectureDiagram from './ArchitectureDiagram'

// Documentation content mapping
const docs = {
  'overview': {
    title: 'System Overview',
    content: `# Federated Learning System Documentation

## Documentation Index

- [Architecture](docs/architecture) - System architecture and components
- [Deployment](docs/deployment) - System deployment and operations
- [Development](docs/development) - Development and extension guide
- [Adding Models](docs/adding-models) - How to add new models to federated learning
- [Federated Learning System](docs/federated-learning-system) - Core architecture and implementation details

## Quick Start

1. Check [Deployment Guide](deployment) for system deployment
2. Check [Architecture](architecture) for system architecture
3. Check [Development Guide](development) for development and extension

## System Overview

The federated learning system consists of the following components:

- **Central Server (Backend)** - Django backend for task and model management
- **Frontend Interface** - Vue 3 frontend
- **Regional Node** - Connects central server and edge devices
- **Edge Device** - Client that performs local training

## Contact

For questions, please check the relevant documentation or submit an Issue.
`
  },
  'architecture': {
    title: 'System Architecture',
    content: `# System Architecture

## Architecture Overview

The system uses a three-tier hierarchical architecture for efficient federated learning task management.

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

- **Fed-Evo**: Simple average aggregation
- **Weighted Aggregation**: Weighted by data size
- **Custom Aggregation**: Extensible

## Storage Architecture

- **PostgreSQL**: Tasks, model metadata, user data
- **MinIO**: Model file storage
- **Redis**: Cache and sessions
- **Local Files**: Device-side training data
`
  },
  'deployment': {
    title: 'Deployment Guide',
    content: `# Deployment Guide

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

\`\`\`bash
docker load -i images.tar
\`\`\`

### 2. Start Services

\`\`\`bash
docker-compose up -d
\`\`\`

## Service Access

- Frontend: http://localhost:8086
- RabbitMQ Management: http://localhost:15672
  - Username: \`rabbitmq\`
  - Password: \`rabbitmq\`
- MinIO Console: http://localhost:9001
  - Username: \`admin\`
  - Password: \`admin123456\`

## Directory Structure

\`\`\`
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
\`\`\`

## Component Deployment

### Central Server (Backend)

The backend is automatically started via docker-compose. It includes:
- Database migrations on startup
- API server on port 8085

### Frontend

The frontend is automatically started via docker-compose on port 8086.

### Regional Node

Deploy regional node separately:

\`\`\`bash
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
\`\`\`

### Edge Device

Deploy edge device:

\`\`\`bash
cd device
pip install -r requirements.txt

# Configure device/config.py or use environment variables

# Start device
python start_device.py <device_id> <region_id> <central_server_url>
\`\`\`

## Operations

### Restart Services

\`\`\`bash
docker-compose restart
\`\`\`

### Stop Services

\`\`\`bash
docker-compose down
\`\`\`

### View Logs

\`\`\`bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
\`\`\`

### Database Migrations

Backend automatically runs migrations on startup. To run manually:

\`\`\`bash
docker-compose exec backend python manage.py migrate
\`\`\`

## Configuration

### Environment Variables

Key environment variables for backend (in docker-compose.yml):
- \`POSTGRES_HOST\`, \`POSTGRES_NAME\`, \`POSTGRES_USER\`, \`POSTGRES_PASSWORD\`
- \`RABBITMQ_HOST\`, \`RABBITMQ_PORT\`, \`RABBITMQ_USER\`, \`RABBITMQ_PASSWORD\`
- \`MINIO_ENDPOINT\`, \`MINIO_PORT\`, \`MINIO_ACCESS_KEY\`, \`MINIO_SECRET_KEY\`

### MQTT Configuration

Edit \`mosquitto.conf\` and \`mosquitto.passwd\` for MQTT authentication.

### Redis Configuration

Edit \`redis/redis.conf\` for Redis settings.

## Troubleshooting

### Port Conflicts

If ports are already in use, modify \`docker-compose.yml\` port mappings.

### Database Connection Issues

Ensure PostgreSQL is healthy before backend starts:
\`\`\`bash
docker-compose ps db
\`\`\`

### Service Health Checks

Check service status:
\`\`\`bash
docker-compose ps
\`\`\`

## Data Persistence

All data is persisted in the following directories:
- \`data/db/\` - PostgreSQL data
- \`redis/data/\` - Redis data
- \`rabbitmq/data/\` - RabbitMQ data
- \`minio/data/\` - MinIO data

Backup these directories for data backup.
`
  },
  'development': {
    title: 'Development Guide',
    content: `# Development Guide

## Project Structure

\`\`\`
federated_learning/
├── backend/          # Django backend
├── frontend/         # Vue 3 frontend
├── regional/         # Regional node service
├── device/           # Edge device client
├── shared/           # Shared model definitions
└── docs/             # Documentation
\`\`\`

## Development Setup

### Backend Development

\`\`\`bash
cd backend
pip install -r requirements.txt

# Setup database
python manage.py migrate

# Run development server
python manage.py runserver
\`\`\`

### Frontend Development

\`\`\`bash
cd frontend
pnpm install

# Development server
pnpm dev

# Build for production
pnpm build
\`\`\`

### Regional Node Development

\`\`\`bash
cd regional
pip install -r requirements.txt

# Configure environment (see regional/readme.md)
python run.py
\`\`\`

### Device Development

\`\`\`bash
cd device
pip install -r requirements.txt

# Configure device/config.py
python start_device.py <device_id> <region_id> <central_server_url>
\`\`\`

## Core Concepts

### Federated Learning Flow

1. **Task Creation**: Central server creates a federated learning task
2. **Task Distribution**: Task sent to regional node via RabbitMQ
3. **Device Notification**: Regional node notifies devices via MQTT
4. **Training**: Devices connect to fedevo server and train locally
5. **Aggregation**: fedevo server aggregates model updates
6. **Result Upload**: Aggregated model uploaded to central server

### Model Management

Models are stored in MinIO and managed through the backend API:
- Upload models via frontend or API
- Models are versioned
- Models can be downloaded by regional nodes and devices

### Task Types

1. **CNN Tasks**: Standard federated learning, auto-collect models after completion
2. **Proxy Model Tasks**: Different execution flow
3. **Large Model Tasks**: Execute large models with parameter desensitization

## Code Structure

### Backend

- \`learn_management/\`: Task and model management
- \`user/\`: User management
- \`utils/\`: Common utilities and constants

### Regional Node

- \`app/fed/\`: fedevo server management
- \`app/service/\`: Task management
- \`app/utils/\`: Communication clients (RabbitMQ, MQTT, HTTP)

### Device

- \`flower_client.py\`: fedevo client implementation
- \`mqtt_handler.py\`: MQTT message handling
- \`main.py\`: Main device logic

## Adding New Features

### Adding a New Aggregation Algorithm

1. Implement aggregation function in regional node
2. Add algorithm to backend constants
3. Update frontend to show new option

### Adding a New Model Type

1. Define model in \`shared/\` or device-specific location
2. Update model loading logic
3. Update frontend model selection

### Adding New Communication Protocol

1. Implement client in \`app/utils/\` (regional) or device
2. Update configuration
3. Update documentation

## Testing

### Backend Tests

\`\`\`bash
cd backend
python manage.py test
\`\`\`

### Frontend Tests

\`\`\`bash
cd frontend
pnpm test:unit
\`\`\`

### Integration Testing

Use the test script:
\`\`\`bash
python test_federated_learning.py
\`\`\`

## Debugging

### Backend Logs

\`\`\`bash
docker-compose logs -f backend
\`\`\`

### Regional Node Logs

Check \`logs/regional.log\` or console output.

### Device Logs

Check device log files configured in \`device/config.py\`.

### MQTT Debugging

Use MQTT client tools to monitor topics:
- Task topics: \`federated_task_{device_id}/task_start\`
- Status topics: \`federated_task_{device_id}/status\`

## Best Practices

1. **Error Handling**: Always handle network errors and timeouts
2. **Logging**: Use structured logging (loguru)
3. **Configuration**: Use environment variables for sensitive data
4. **Code Style**: Follow Python PEP 8 and Vue style guide
5. **Documentation**: Update docs when adding features

## Common Issues

### Device Not Connecting

- Check MQTT broker connectivity
- Verify device ID and region ID
- Check network connectivity to regional node

### Training Not Starting

- Verify fedevo server is running on regional node
- Check device can reach regional node
- Verify model download succeeded

### Model Upload Failing

- Check MinIO connectivity
- Verify credentials
- Check disk space

## API Documentation

Backend provides REST API. Check API endpoints:
- Task management: \`/api/tasks/\`
- Model management: \`/api/models/\`
- Node management: \`/api/nodes/\`

Use frontend to explore API or check backend code for endpoint definitions.
`
  },
  'adding-models': {
    title: 'Adding Models Guide',
    content: `# Adding New Models to Federated Learning System

This guide explains how to add a new model to the federated learning system, covering the complete workflow from frontend operations to device execution.

## Overview

The model addition process involves multiple components:

1. **Frontend**: Upload model file and create model version
2. **Backend**: Store model file and metadata
3. **Task Creation**: Create federated learning task with model
4. **Regional Node**: Receive task, download model, start fedevo server
5. **Device**: Receive task via MQTT, download model, connect to fedevo server

## Step 1: Frontend Operations

### 1.1 Create Model Information

1. Navigate to **Model Management** page (\`/model/management\`)
2. Click **"新建模型"** (New Model) button
3. Fill in the form:
   - **Model Name**: e.g., "MNIST CNN"
   - **Description**: Optional description
4. Click **"确定"** (Confirm) to create

**API Call**: \`POST /v1/learn_management/model_info/\`

### 1.2 Upload Model File

1. Select the created model from the left sidebar
2. Click **"上传版本"** (Upload Version) button
3. In the dialog:
   - Enter **Version Number**: e.g., "v1.0.0"
   - Click **"点击上传"** (Click to Upload)
   - Select model file (supported formats: \`.pt\`, \`.pth\`, \`.zip\`, \`.pkl\`, \`.npz\`)
4. Click **"提交"** (Submit)

**File Upload Process**:
- Frontend calls \`POST /v1/learn_management/model_version/upload/\`
- Backend uploads file to MinIO (or local storage)
- Returns \`file_path\` and \`file_url\`

## Step 2: Backend Processing

### 2.1 Model File Upload Handler

**Location**: \`backend/learn_management/model_views.py\`

**Process**:
1. Receives file via \`ModelFileUploadView\`
2. Validates file type (\`.pt\`, \`.pth\`, \`.zip\`, \`.pkl\`, \`.npz\`)
3. Generates unique filename with timestamp
4. Uploads to MinIO (or local filesystem)
5. Returns file path and URL

## Step 3: Task Creation and Distribution

### 3.1 Create Federated Learning Task

1. Navigate to **Federated Task** page
2. Create new task with:
   - **Model**: Select deployed model version
   - **Regional Node**: Select target regional node
   - **Rounds**: Number of training rounds
   - **Aggregation Method**: e.g., "Fed-Evo"
   - **Participation Rate**: Percentage of devices to participate

### 3.2 Task Distribution to Regional Node

**Process**:
1. Backend sends task to RabbitMQ queue
2. Regional node receives task
3. Downloads model file
4. Starts fedevo server
5. Distributes task to devices via MQTT

## Step 4: Device Execution

### 4.1 Receive Task via MQTT

Device subscribes to topic: \`federated_task_{device_id}/task_start\`

### 4.2 Download Model File

Device downloads model file from provided URL

### 4.3 Connect to fedevo Server

Device connects to fedevo server and starts training

### 4.4 Execute Federated Learning

1. Device receives global model from fedevo server
2. Trains locally on device data
3. Sends model updates to fedevo server
4. Repeats for specified number of rounds
5. Reports training logs to central server

## Complete Workflow Summary

\`\`\`
1. Frontend: Upload model file → Create model version → Deploy
2. Frontend: Create federated task with model
3. Backend: Send task to RabbitMQ
4. Regional Node: Receive task → Download model → Start fedevo server
5. Regional Node: Publish task to MQTT
6. Device: Receive task → Download model → Create trainer
7. Device: Connect to fedevo server → Execute federated learning
8. Device: Send updates → fedevo server aggregates
9. Regional Node: Upload final model to central server
\`\`\`

## Troubleshooting

### Model File Not Found
- Check MinIO/local storage configuration
- Verify file path in model version
- Check file permissions

### fedevo Server Not Starting
- Check port availability (default: 8080)
- Verify model loading code
- Check regional node logs

### Device Not Connecting
- Verify MQTT connectivity
- Check fedevo server address
- Verify model download succeeded
- Check device logs

### Model Loading Errors
- Verify model file format matches code
- Check model architecture matches
- Verify PyTorch version compatibility

## Best Practices

1. **Model Versioning**: Always use semantic versioning (v1.0.0, v1.1.0, etc.)
2. **Model Testing**: Test model locally before uploading
3. **File Validation**: Validate model files before deployment
4. **Error Handling**: Implement proper error handling at each step
5. **Logging**: Add comprehensive logging for debugging
6. **Documentation**: Document model architecture and requirements
`
  },
  'federated-learning-system': {
    title: 'Federated Learning Core',
    content: `# Federated Learning Core Architecture

## Server-Side Core Functions

### Global Model Management

\`\`\`python
# Server maintains a global model
global_model = LeNet()  # Can be any deep learning model

# Model state management
current_round = 0
client_updates = []
MIN_CLIENTS = 2  # Minimum number of clients for aggregation
\`\`\`

### Client Count Control

\`\`\`python
# Control the number of clients participating in training
MIN_CLIENTS = 2  # Minimum 2 clients required for aggregation

# Dynamic participation strategy adjustment
def should_aggregate():
    return len(client_updates) >= MIN_CLIENTS

# Wait for enough clients before aggregation
if len(client_updates) >= MIN_CLIENTS:
    # Execute aggregation
    aggregated_model = custom_aggregation_algorithm(client_updates)
\`\`\`

### Custom Aggregation Algorithms

\`\`\`python
# 1. Simple Average Aggregation (Fed-Evo)
def Fed-Evo_aggregation(client_models):
    averaged_model = copy.deepcopy(client_models[0])
    for key in averaged_model.keys():
        for i in range(1, len(client_models)):
            averaged_model[key] += client_models[i][key]
        averaged_model[key] = torch.div(averaged_model[key], len(client_models))
    return averaged_model

# 2. Weighted Aggregation (by data size)
def weighted_aggregation(client_models, data_sizes):
    total_samples = sum(data_sizes)
    averaged_model = copy.deepcopy(client_models[0])
    for key in averaged_model.keys():
        weighted_sum = 0
        for model, size in zip(client_models, data_sizes):
            weighted_sum += model[key] * size
        averaged_model[key] = weighted_sum / total_samples
    return averaged_model

# 3. Custom Aggregation Strategy
def custom_aggregation_algorithm(client_models):
    # Can customize aggregation logic based on business needs
    # For example: outlier filtering, model quality assessment, etc.
    return your_custom_aggregation(client_models)
\`\`\`

## Client-Side Core Functions

### Local Training Implementation

\`\`\`python
class FederatedClient:
    def __init__(self, server_address, client_id):
        self.server_address = server_address
        self.client_id = client_id
        self.model = LeNet()  # Local model
        
        # Prepare local data
        self.setup_local_data()
    
    def setup_local_data(self):
        """Prepare local training data"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST('./data', train=True, download=True,
                                    transform=transform)
        # Simulate distributed data - each client uses different subset
        n_samples = len(self.dataset) // 4
        start_idx = self.client_id * n_samples
        end_idx = (self.client_id + 1) * n_samples
        self.dataset = torch.utils.data.Subset(
            self.dataset, range(start_idx, end_idx)
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )
    
    def local_training(self, epochs=1):
        """Execute local model training"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()
    
    def federated_training_loop(self, rounds=5):
        """Federated learning training loop"""
        for round in range(rounds):
            print(f"Round {round + 1}")
            
            # 1. Get global model from server
            self.get_global_model()
            
            # 2. Train on local data
            local_model = self.local_training(epochs=1)
            
            # 3. Send update to server
            self.send_model_update(local_model)
\`\`\`

## Network Setup

Flow: Client starts → Get model → Local training → Upload update → Server aggregation

### Execution Flow

Client → Server

### Startup Sequence

1. **Central Server**: Start fedevo server
2. **Regional Server**: Send task (central server IP, model type, etc.) to regional server, regional server forwards to MQTT
3. **Device**: Listen to MQTT, trigger task

## Supported Model Architectures

### 1. Deep Learning Framework Support

#### Server-Side Frameworks
- **PyTorch**: Dynamic graph, easy debugging
- **TensorFlow**: Static graph, production deployment
- **JAX**: High-performance computing, functional programming

#### Mobile Frameworks
- **TensorFlow Lite**: Lightweight deployment for Android, iOS
- **MLX**: Apple Silicon optimized, macOS/iOS specific
- **ONNX Runtime**: Cross-platform model inference

#### Advanced Frameworks
- **Transformers**: Hugging Face pre-trained model library
- **FastAI**: High-level API, rapid prototyping

#### Platform Support
- **Android**: TensorFlow Lite, ONNX Runtime
- **iOS**: Core ML, TensorFlow Lite, MLX

### 2. Model Optimization Strategies

#### Model Compression
- **Quantization**: INT8/FP16 precision reduction, reduce model size
- **Pruning**: Remove unimportant weights and neurons
- **Knowledge Distillation**: Knowledge transfer from large to small models

#### Dynamic Adjustment
- **Adaptive Architecture**: Adjust model complexity based on device performance
- **Layered Training**: Different layers use different training strategies
- **Progressive Training**: Gradually train from simple to complex
`
  }
}

export default function Documentation() {
  const { docId } = useParams()
  const [currentDoc, setCurrentDoc] = useState('overview')
  const [htmlContent, setHtmlContent] = useState('')

  useEffect(() => {
    const docKey = docId || 'overview'
    setCurrentDoc(docKey)
    
    if (docs[docKey]) {
      // Configure marked options to allow HTML
      marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false,
        sanitize: false  // Allow HTML/SVG rendering
      })

      // Use marked.parse() instead of marked() for better HTML support
      const html = marked.parse(docs[docKey].content, {
        sanitize: false,
        gfm: true
      })
      setHtmlContent(html)
    }
  }, [docId])

  const tocItems = [
    { id: 'overview', title: 'System Overview', icon: '📚' },
    { id: 'architecture', title: 'Architecture', icon: '🏗️' },
    { id: 'deployment', title: 'Deployment', icon: '🚀' },
    { id: 'development', title: 'Development', icon: '💻' },
    { id: 'adding-models', title: 'Adding Models', icon: '🤖' },
    { id: 'federated-learning-system', title: 'FL Core', icon: '🔬' },

  ]

  return (
    <div className={styles.documentation}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h2>📖 Documentation</h2>
        </div>
        <nav className={styles.toc}>
          {tocItems.map(item => (
            <Link
              key={item.id}
              to={`/docs/${item.id}`}
              className={`${styles.tocItem} ${currentDoc === item.id ? styles.active : ''}`}
            >
              <span className={styles.tocIcon}>{item.icon}</span>
              <span className={styles.tocTitle}>{item.title}</span>
            </Link>
          ))}
        </nav>
      </aside>
      
      <main className={styles.content}>
        <div className={styles.contentWrapper}>
          {currentDoc === 'architecture' ? (
            <article className={styles.markdown}>
              <h1>System Architecture</h1>
              <h2>Architecture Overview</h2>
              <ArchitectureDiagram />
              <div dangerouslySetInnerHTML={{
                __html: htmlContent.replace(/<h1>.*?<\/h1>/, '').replace(/<h2>Architecture Overview<\/h2>/, '').replace(/<div style=.*?<\/div>/, '')
              }} />
            </article>
          ) : (
            <article
              className={styles.markdown}
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          )}
        </div>
      </main>
    </div>
  )
}
