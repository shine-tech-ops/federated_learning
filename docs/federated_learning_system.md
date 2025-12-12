# Federated Learning Core Architecture

## Server-Side Core Functions

### Global Model Management

```python
# Server maintains a global model
global_model = LeNet()  # Can be any deep learning model

# Model state management
current_round = 0
client_updates = []
MIN_CLIENTS = 2  # Minimum number of clients for aggregation
```

### Client Count Control

```python
# Control the number of clients participating in training
MIN_CLIENTS = 2  # Minimum 2 clients required for aggregation

# Dynamic participation strategy adjustment
def should_aggregate():
    return len(client_updates) >= MIN_CLIENTS

# Wait for enough clients before aggregation
if len(client_updates) >= MIN_CLIENTS:
    # Execute aggregation
    aggregated_model = custom_aggregation_algorithm(client_updates)
```

### Custom Aggregation Algorithms

```python
# 1. Simple Average Aggregation (FedAvg)
def fedavg_aggregation(client_models):
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
```

## Client-Side Core Functions

### Local Training Implementation

```python
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
```

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

### 2. Model Optimization Strategies (Not Explored)

#### Model Compression
- **Quantization**: INT8/FP16 precision reduction, reduce model size
- **Pruning**: Remove unimportant weights and neurons
- **Knowledge Distillation**: Knowledge transfer from large to small models

#### Dynamic Adjustment
- **Adaptive Architecture**: Adjust model complexity based on device performance
- **Layered Training**: Different layers use different training strategies
- **Progressive Training**: Gradually train from simple to complex
