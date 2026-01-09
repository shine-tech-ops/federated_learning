# Adding New Models to Federated Learning System

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

1. Navigate to **Model Management** page (`/model/management`)
2. Click **"新建模型"** (New Model) button
3. Fill in the form:
   - **Model Name**: e.g., "MNIST CNN"
   - **Description**: Optional description
4. Click **"确定"** (Confirm) to create

**API Call**: `POST /v1/learn_management/model_info/`

### 1.2 Upload Model File

1. Select the created model from the left sidebar
2. Click **"上传版本"** (Upload Version) button
3. In the dialog:
   - Enter **Version Number**: e.g., "v1.0.0"
   - Click **"点击上传"** (Click to Upload)
   - Select model file (supported formats: `.pt`, `.pth`, `.zip`, `.pkl`, `.npz`)
4. Click **"提交"** (Submit)

**File Upload Process**:
- Frontend calls `POST /v1/learn_management/model_version/upload/`
- Backend uploads file to MinIO (or local storage)
- Returns `file_path` and `file_url`

**API Call**: `POST /v1/learn_management/model_version/upload/`

### 1.3 Create Model Version

After file upload, frontend automatically creates model version:

**API Call**: `POST /v1/learn_management/model_version/`

**Request Body**:
```json
{
  "model_info": 1,
  "version": "v1.0.0",
  "model_file": "models/1234567890_model.pt",
  "accuracy": null,
  "loss": null
}
```

### 1.4 Deploy Model Version


## Step 2: Backend Processing

### 2.1 Model File Upload Handler

**Location**: `backend/learn_management/model_views.py`

**Process**:
1. Receives file via `ModelFileUploadView`
2. Validates file type (`.pt`, `.pth`, `.zip`, `.pkl`, `.npz`)
3. Generates unique filename with timestamp
4. Uploads to MinIO (or local filesystem):
   - MinIO: `models/{timestamp}_{filename}`
   - Local: `MEDIA_ROOT/models/{timestamp}_{filename}`
5. Returns file path and URL

**Storage Configuration**:
- MinIO: Configured via `USE_MINIO_STORAGE` setting
- Local: Uses `MEDIA_ROOT` setting

### 2.2 Model Version Creation

**Location**: `backend/learn_management/models.py`

**Model Structure**:
- `ModelInfo`: Model metadata (name, description)
- `ModelVersion`: Version information (version number, file path, deployment status)

**Database Fields**:
- `model_file`: Path to model file in storage
- `is_deployed`: Whether version is active
- `model_info`: Foreign key to ModelInfo

### 2.3 Model Download

**API**: `GET /v1/learn_management/model_version/{id}/download/`

**Process**:
- If MinIO: Streams file from MinIO
- If Local: Serves file from filesystem
- Returns file with proper content-type headers

## Step 3: Task Creation and Distribution

### 3.1 Create Federated Learning Task

1. Navigate to **Federated Task** page
2. Create new task with:
   - **Model**: Select deployed model version
   - **Regional Node**: Select target regional node
   - **Rounds**: Number of training rounds
   - **Aggregation Method**: e.g., "fedavg"
   - **Participation Rate**: Percentage of devices to participate

**Backend Process**:
- Task created in database (`FederatedTask` model)
- Task includes:
  - `model_info`: Reference to model
  - `model_version`: Reference to model version
  - `region_node`: Target regional node
  - `rounds`: Training rounds
  - `aggregation_method`: Aggregation strategy

### 3.2 Task Distribution to Regional Node

**Location**: `backend/learn_management/task_views.py` (or similar)

**Process**:
1. Backend sends task to RabbitMQ queue
2. Message format:
```json
{
  "task_id": "task_123",
  "model_info": {
    "id": 1,
    "name": "MNIST CNN"
  },
  "model_version": {
    "id": 1,
    "version": "v1.0.0",
    "model_file": "models/1234567890_model.pt",
    "file_url": "http://minio:9000/models/1234567890_model.pt"
  },
  "rounds": 10,
  "aggregation_method": "fedavg",
  "participation_rate": 50,
  "edge_devices": ["device_1", "device_2"]
}
```

**RabbitMQ Configuration**:
- Exchange: Task distribution exchange
- Queue: Regional node specific queue
- Routing Key: `region.{region_id}.task`

## Step 4: Regional Node Reception

### 4.1 Receive Task from RabbitMQ

**Location**: `regional/regional_node.py`

**Process**:
1. Regional node listens to RabbitMQ queue
2. Receives task message
3. Calls `TaskManager.start_task(task_data)`
4. Extracts model information:
   - Model file URL
   - Model type/framework
   - Task parameters

### 4.2 Download Model File

**Location**: `regional/app/utils/http_client.py`

**Process**:
1. Regional node downloads model file from MinIO URL
2. Saves to local storage (e.g., `models/` directory)
3. Validates file integrity
4. Prepares model for fedevo server

**Code Example**:
```python
def download_model(self, model_url: str, save_path: str):
    """Download model file from URL"""
    response = requests.get(model_url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
```

### 4.3 Start fedevo Server

**Location**: `regional/app/fed/server_manager.py`

**Process**:
1. Regional node calls `FedServerManager.start_server(task_data)`
2. Creates model instance:
   ```python
   from shared.mnist_model import create_model
   self.model = create_model()
   ```
3. Configures fedevo server:
   - Host: `0.0.0.0`
   - Port: `8080`
   - Strategy: Based on `aggregation_method`
   - Parameters: `rounds`, `participation_rate`, etc.
4. Starts server in background thread:
   ```python
   fl.server.start_server(
       server_address=f"{host}:{port}",
       config=fl.server.ServerConfig(num_rounds=rounds),
       strategy=strategy
   )
   ```

**Server Configuration**:
- `fraction_fit`: Participation rate / 100
- `min_fit_clients`: Minimum number of clients
- `min_available_clients`: Minimum available clients
- `evaluate_fn`: Server-side evaluation function
- `on_fit_config_fn`: Training configuration function

### 4.4 Distribute Task to Devices via MQTT

**Location**: `regional/app/utils/mqtt_client.py`

**Process**:
1. Regional node publishes task to MQTT
2. Topic: `federated_task_{device_id}/task_start`
3. Message format:
```json
{
  "action": "task_start",
  "task_id": "task_123",
  "model_info": {
    "id": 1,
    "name": "MNIST CNN"
  },
  "model_version": {
    "id": 1,
    "version": "v1.0.0",
    "model_file": "models/1234567890_model.pt",
    "file_url": "http://minio:9000/models/1234567890_model.pt"
  },
  "flower_server": {
    "host": "192.168.1.100",
    "port": 8080
  },
  "rounds": 10,
  "local_epochs": 3,
  "learning_rate": 0.01
}
```

## Step 5: Device Reception and Execution

### 5.1 Receive Task via MQTT

**Location**: `device/mqtt_handler.py`

**Process**:
1. Device subscribes to topic: `federated_task_{device_id}/task_start`
2. Receives task message
3. Calls `EdgeDevice._handle_task_start(message)`

### 5.2 Download Model File

**Location**: `device/main.py` (in `_handle_task_start`)

**Process**:
1. Extracts model file URL from task message
2. Downloads model file to local storage
3. Validates file integrity
4. Prepares model for training

**Code Example**:
```python
def _download_model(self, model_url: str, save_path: str):
    """Download model file"""
    response = requests.get(model_url, stream=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
```

### 5.3 Create Trainer

**Location**: `device/main.py`

**Process**:
1. Creates trainer instance based on model type:
   ```python
   trainer_class = {
       "cnn": MNISTTrainer,
       # Add more trainer types here
   }
   self.trainer = trainer_class[model_type](
       device_id=self.device_id,
       task_data=message
   )
   ```
2. Trainer loads model:
   - Loads model architecture
   - Loads model weights from downloaded file
   - Prepares data loader

### 5.4 Connect to fedevo Server

**Location**: `device/flower_client.py`

**Process**:
1. Creates fedevo client instance:
   ```python
   self.flower_client = FlowerClient(
       device_id=self.device_id,
       trainer=self.trainer,
       server_address=server_address,
       task_id=message.get('task_id'),
       region_id=self.region_id
   )
   ```
2. Connects to fedevo server:
   ```python
   fl.client.start_numpy_client(
       server_address=server_address,
       client=FlowerClientAdapter(self.trainer)
   )
   ```

### 5.5 Execute Federated Learning

**Process**:
1. Device receives global model from fedevo server
2. Trains locally on device data
3. Sends model updates to fedevo server
4. Repeats for specified number of rounds
5. Reports training logs to central server

## Step 6: Adding New Model Code

To add support for a new model type, follow these steps:

### 6.1 Create Model Definition

**Location**: `shared/{model_name}_model.py`

**Example** (for a new CNN model):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(NewCNN, self).__init__()
        # Define your model architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # ... more layers
    
    def forward(self, x):
        # Define forward pass
        return x

def create_model(num_classes=10):
    """Create model instance"""
    return NewCNN(num_classes)

def get_model_parameters(model):
    """Get model parameters for federated learning"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Set model parameters from federated learning"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
```

### 6.2 Update Regional Node Model Loading

**Location**: `regional/app/fed/server_manager.py`

**Update**:
```python
# Import your new model
from new_model import create_model, get_model_parameters, set_model_parameters

# In start_server method:
self.model = create_model()  # Uses your new model
```

### 6.3 Create Device Trainer

**Location**: `device/{model_name}_trainer.py`

**Example**:
```python
from mnist_trainer import BaseTrainer  # Or create base class
from shared.new_model import create_model, get_model_parameters, set_model_parameters

class NewModelTrainer(BaseTrainer):
    def __init__(self, device_id: str, task_data: dict):
        super().__init__(device_id, task_data)
        # Load your model
        self.model = create_model()
        # Load weights if provided
        if 'model_file' in task_data:
            self.load_model(task_data['model_file'])
    
    def load_model(self, model_path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(model_path))
    
    def get_model_parameters(self):
        """Get model parameters for federated learning"""
        return get_model_parameters(self.model)
    
    def set_model_parameters(self, parameters):
        """Set model parameters from federated learning"""
        set_model_parameters(self.model, parameters)
    
    def train(self, epochs: int):
        """Local training logic"""
        # Implement your training logic
        pass
```

### 6.4 Register Trainer in Device

**Location**: `device/main.py`

**Update**:
```python
from new_model_trainer import NewModelTrainer

trainer_class = {
    "cnn": MNISTTrainer,
    "new_model": NewModelTrainer,  # Add your new trainer
}
```

### 6.5 Update Model Type in Frontend/Backend

**Frontend**: Add model type option in task creation form

**Backend**: Update model type choices if needed:
```python
MODEL_TYPE_CHOICES = [
    ('cnn', 'CNN'),
    ('new_model', 'New Model'),  # Add your model type
]
```

## Complete Workflow Summary

```
1. Frontend: Upload model file → Create model version → Deploy
2. Frontend: Create federated task with model
3. Backend: Send task to RabbitMQ
4. Regional Node: Receive task → Download model → Start fedevo server
5. Regional Node: Publish task to MQTT
6. Device: Receive task → Download model → Create trainer
7. Device: Connect to fedevo server → Execute federated learning
8. Device: Send updates → fedevo server aggregates
9. Regional Node: Upload final model to central server
```

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
