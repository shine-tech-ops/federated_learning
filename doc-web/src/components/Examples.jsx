import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { marked } from 'marked'
import styles from './Documentation.module.css'

// Examples content mapping
const examples = {
  'cnn-training': {
    title: 'Federated CNN Training',
    content: `# Feature 1: Federated CNN Training

## Overview

Federated CNN training enables multiple clients to collaboratively train a convolutional neural network without sharing raw data. Each client trains locally on their data, and the central server aggregates model updates until convergence.

## Key Components

- **Client-side**: Local data management, model training, parameter updates
- **Server-side**: Model aggregation, convergence monitoring, global model distribution
- **Validation**: MNIST dataset for proof of concept

## Architecture Flow

\`\`\`
┌─────────────────┐
│  Central Server │
│   (Aggregator)  │
└────────┬────────┘
         │ Global Model
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Client1│ │Client2│ │Client3│ │Client4│
│ CNN   │ │ CNN   │ │ CNN   │ │ CNN   │
└───┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
    │        │        │        │
    └────────┴────────┴────────┘
      Local Model Updates
\`\`\`

## Implementation Details

### 1. Client-Side Implementation

**Local Data Management**:
\`\`\`python
class FederatedCNNClient:
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.model = self.build_cnn_model()
        self.local_data = self.load_local_data(data_path)
        
    def load_local_data(self, data_path):
        """Load and preprocess local MNIST data"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load local partition of MNIST
        dataset = datasets.MNIST(
            data_path, 
            train=True, 
            download=True,
            transform=transform
        )
        
        # Create data loader
        return DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True
        )
    
    def build_cnn_model(self):
        """Build CNN architecture"""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def local_train(self, epochs=1):
        """Execute local training"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.local_data):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        return self.model.state_dict()
\`\`\`

### 2. Server-Side Aggregation

**FedAvg Algorithm**:
\`\`\`python
class FederatedServer:
    def __init__(self, num_clients):
        self.global_model = self.build_cnn_model()
        self.num_clients = num_clients
        self.convergence_threshold = 0.01
        
    def aggregate_models(self, client_models, client_weights):
        """
        Aggregate client models using weighted averaging
        
        Args:
            client_models: List of client model state dicts
            client_weights: List of weights (e.g., data sizes)
        """
        global_dict = self.global_model.state_dict()
        
        # Weighted average of all parameters
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            total_weight = sum(client_weights)
            
            for client_model, weight in zip(client_models, client_weights):
                global_dict[key] += client_model[key] * (weight / total_weight)
        
        self.global_model.load_state_dict(global_dict)
        return self.global_model.state_dict()
    
    def check_convergence(self, prev_loss, current_loss):
        """Check if model has converged"""
        return abs(prev_loss - current_loss) < self.convergence_threshold
\`\`\`

### 3. Training Loop

\`\`\`python
def federated_cnn_training(server, clients, num_rounds=10):
    """
    Main federated learning training loop
    
    Args:
        server: FederatedServer instance
        clients: List of FederatedCNNClient instances
        num_rounds: Number of federated learning rounds
    """
    for round_num in range(num_rounds):
        print(f"\\n=== Round {round_num + 1}/{num_rounds} ===")
        
        # 1. Distribute global model to clients
        global_weights = server.global_model.state_dict()
        for client in clients:
            client.model.load_state_dict(global_weights)
        
        # 2. Each client trains locally
        client_updates = []
        client_weights = []
        
        for client in clients:
            print(f"Client {client.client_id} training...")
            local_weights = client.local_train(epochs=1)
            client_updates.append(local_weights)
            client_weights.append(len(client.local_data.dataset))
        
        # 3. Server aggregates updates
        print("Aggregating models...")
        server.aggregate_models(client_updates, client_weights)
        
        # 4. Evaluate global model
        accuracy = evaluate_model(server.global_model, test_loader)
        print(f"Global Model Accuracy: {accuracy:.2%}")
        
        # 5. Check convergence
        if round_num > 0 and server.check_convergence(prev_acc, accuracy):
            print("Model converged!")
            break
        prev_acc = accuracy
\`\`\`

## MNIST Validation Example

\`\`\`python
# Setup
num_clients = 4
num_rounds = 10

# Initialize server and clients
server = FederatedServer(num_clients=num_clients)
clients = [
    FederatedCNNClient(client_id=i, data_path=f'./data/client_{i}')
    for i in range(num_clients)
]

# Run federated training
federated_cnn_training(server, clients, num_rounds=num_rounds)
\`\`\`

## Customizable Aggregation Methods

The system supports multiple aggregation algorithms:

1. **FedAvg**: Simple weighted average
2. **FedProx**: Proximal term for handling heterogeneous data
3. **FedOpt**: Server-side adaptive optimization
4. **Custom**: User-defined aggregation logic

## Key Features

- ✅ Local data management on each client
- ✅ Model management with version control
- ✅ Multiple aggregation algorithms
- ✅ MNIST dataset validation
- ✅ Convergence monitoring
- ✅ Privacy-preserving (no raw data sharing)
`
  },
  'optimization': {
    title: 'Federated Optimization',
    content: `# Feature 2: Federated Optimization

## Overview

Federated optimization trains surrogate models on each client to find optimal solutions. The server aggregates these models and uses a selection function to identify the best solution, which is then distributed back to clients for incorporation into their local training.

## Key Differences from CNN Training

- **Model Type**: Surrogate models instead of CNN
- **Optimization Goal**: Finding optimal solutions rather than model convergence
- **Selection Mechanism**: Best solution selection after aggregation

## Architecture Flow

\`\`\`
┌─────────────────────────────┐
│     Central Server          │
│  1. Aggregate Surrogate     │
│  2. Select Optimal Solution │
└──────────┬──────────────────┘
           │ Optimal Solution
    ┌──────┴──────┬──────────┬──────────┐
    │             │          │          │
┌───▼────────┐ ┌─▼─────────┐ ┌▼────────┐ ┌▼────────┐
│  Client 1  │ │ Client 2  │ │Client 3 │ │Client 4 │
│ Surrogate  │ │ Surrogate │ │Surrogate│ │Surrogate│
│   Model    │ │   Model   │ │  Model  │ │  Model  │
└────┬───────┘ └───┬───────┘ └─┬───────┘ └─┬───────┘
     │             │           │           │
     └─────────────┴───────────┴───────────┘
         Upload Surrogate Models
\`\`\`

## Implementation Details

### 1. Surrogate Model Definition

\`\`\`python
class SurrogateModel(nn.Module):
    """
    Surrogate model for optimization problems
    Can be a neural network approximating the objective function
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(SurrogateModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: objective value
        )
    
    def forward(self, x):
        return self.network(x)
\`\`\`

### 2. Client-Side Optimization

\`\`\`python
class FederatedOptimizationClient:
    def __init__(self, client_id, objective_function, search_space):
        self.client_id = client_id
        self.surrogate_model = SurrogateModel(input_dim=search_space.dim)
        self.objective_function = objective_function
        self.search_space = search_space
        self.local_solutions = []
        
    def train_surrogate(self, num_samples=100):
        """Train surrogate model on local objective evaluations"""
        # Sample points from search space
        X = self.search_space.sample(num_samples)
        y = torch.tensor([
            self.objective_function(x) for x in X
        ]).float()
        
        # Train surrogate model
        optimizer = optim.Adam(self.surrogate_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            optimizer.zero_grad()
            predictions = self.surrogate_model(X)
            loss = criterion(predictions.squeeze(), y)
            loss.backward()
            optimizer.step()
        
        return self.surrogate_model.state_dict()
    
    def find_local_optimum(self):
        """Use surrogate model to find local optimum"""
        best_solution = None
        best_value = float('inf')
        
        # Sample candidates and evaluate with surrogate
        candidates = self.search_space.sample(1000)
        with torch.no_grad():
            values = self.surrogate_model(candidates)
        
        # Find best candidate
        best_idx = values.argmin()
        best_solution = candidates[best_idx]
        best_value = self.objective_function(best_solution)
        
        return best_solution, best_value
\`\`\`

### 3. Server-Side Aggregation and Selection

\`\`\`python
class FederatedOptimizationServer:
    def __init__(self, search_space):
        self.global_surrogate = SurrogateModel(input_dim=search_space.dim)
        self.search_space = search_space
        self.best_solution = None
        self.best_value = float('inf')
        
    def aggregate_surrogates(self, client_surrogates, weights):
        """Aggregate client surrogate models"""
        global_dict = self.global_surrogate.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            total_weight = sum(weights)
            
            for surrogate, weight in zip(client_surrogates, weights):
                global_dict[key] += surrogate[key] * (weight / total_weight)
        
        self.global_surrogate.load_state_dict(global_dict)
    
    def select_optimal_solution(self, client_solutions):
        """
        Selection function to choose the best solution
        from all client submissions
        """
        best_solution = None
        best_value = float('inf')
        
        # Evaluate all client solutions with global surrogate
        for solution, value in client_solutions:
            if value < best_value:
                best_value = value
                best_solution = solution
        
        self.best_solution = best_solution
        self.best_value = best_value
        
        return best_solution, best_value
\`\`\`

### 4. Federated Optimization Loop

\`\`\`python
def federated_optimization(server, clients, num_rounds=20):
    """
    Main federated optimization loop
    """
    for round_num in range(num_rounds):
        print(f"\\n=== Optimization Round {round_num + 1}/{num_rounds} ===")
        
        # 1. Clients train surrogate models
        client_surrogates = []
        client_solutions = []
        
        for client in clients:
            print(f"Client {client.client_id} training surrogate...")
            surrogate_weights = client.train_surrogate()
            client_surrogates.append(surrogate_weights)
            
            # Find local optimum
            solution, value = client.find_local_optimum()
            client_solutions.append((solution, value))
            print(f"  Local optimum value: {value:.4f}")
        
        # 2. Server aggregates surrogates
        print("Aggregating surrogate models...")
        weights = [1.0] * len(clients)  # Equal weights
        server.aggregate_surrogates(client_surrogates, weights)
        
        # 3. Server selects optimal solution
        print("Selecting optimal solution...")
        best_solution, best_value = server.select_optimal_solution(
            client_solutions
        )
        print(f"Global best value: {best_value:.4f}")
        
        # 4. Distribute optimal solution to clients
        for client in clients:
            client.incorporate_optimal_solution(best_solution)
        
        # 5. Check termination criteria
        if best_value < convergence_threshold:
            print("Optimization converged!")
            break
\`\`\`

## Example Use Case

\`\`\`python
# Define optimization problem
def objective_function(x):
    """Example: Minimize Rosenbrock function"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))

# Setup
search_space = SearchSpace(dim=10, bounds=(-5, 5))
server = FederatedOptimizationServer(search_space)
clients = [
    FederatedOptimizationClient(i, objective_function, search_space)
    for i in range(4)
]

# Run optimization
federated_optimization(server, clients, num_rounds=20)
\`\`\`

## Key Features

- ✅ Surrogate model training on each client
- ✅ Server-side aggregation of surrogate models
- ✅ Optimal solution selection mechanism
- ✅ Solution distribution to clients
- ✅ Convergence monitoring
- ✅ Flexible objective function support
`
  },
  'large-small-collaboration': {
    title: 'Large-Small Model Collaboration',
    content: `# Feature 3: Federated Large-Small Model Collaboration

## Overview

This feature implements a hierarchical federated learning system where edge devices run small models for instant responses, while a cloud-based large model performs aggregation, validation, and retraining. The system uses federated distillation to update global knowledge.

## Scenario: Personal Health Management Assistant

A cross-modal (image, voice, text) health assistant that:
- Provides instant preliminary diagnosis on-device
- Uploads semantic summaries and parameter updates (not raw data)
- Performs multi-client aggregation and validation in the cloud
- Updates global knowledge through federated distillation

## System Architecture

\`\`\`
┌─────────────────────────────────────────┐
│         Cloud Server                    │
│  ┌───────────────────────────────────┐  │
│  │   Large Language Model (LLM)      │  │
│  │   - DeepSeek / ChatGPT / Llama    │  │
│  │   - Multi-client Aggregation      │  │
│  │   - Validation & Retraining       │  │
│  │   - Federated Distillation        │  │
│  └───────────────┬───────────────────┘  │
└──────────────────┼──────────────────────┘
                   │ Knowledge Updates
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
┌───────▼──────┐ ┌─▼─────────┐ ┌▼────────┐ ┌▼────────┐
│  Device 1    │ │ Device 2  │ │Device 3 │ │Device 4 │
│  (Android)   │ │  (iOS)    │ │(Android)│ │ (iOS)   │
│ ┌──────────┐ │ │┌─────────┐│ │┌────────┐│ │┌────────┐│
│ │Small LLM │ │ ││Small LLM││ ││Small LLM││ ││Small LLM││
│ │ TinyLlama│ │ ││  Phi-2  ││ ││MobileLLM││ ││ Gemma │││
│ └──────────┘ │ │└─────────┘│ │└────────┘│ │└────────┘│
│ Instant Reply│ │Instant   ││ │Instant  ││ │Instant  ││
└──────────────┘ │Reply     ││ │Reply    ││ │Reply    ││
                 └──────────┘│ └─────────┘│ └─────────┘│
                              └───────────┘ └───────────┘
\`\`\`

## Heterogeneous Device Setup

**20 Heterogeneous Devices**:
- 10 Android devices (various models, compute capabilities)
- 10 iOS devices (iPhone models with different chips)

**Model Heterogeneity**:
- Each device runs 2+ LLMs (open-source + closed-source)
- Examples: DeepSeek, ChatGPT API, Llama, Phi-2, Gemma, TinyLlama

**Server-Side Models**:
- Large models with API interfaces
- Switchable model backends for different tasks

## Implementation Details

### 1. Small Model (Edge Device)

\`\`\`python
class EdgeHealthAssistant:
    """
    Small model running on edge device for instant responses
    """
    def __init__(self, device_id, model_name, device_type):
        self.device_id = device_id
        self.device_type = device_type  # 'android' or 'ios'
        self.small_model = self.load_small_model(model_name)
        self.local_data_manager = LocalDataManager()
        
    def load_small_model(self, model_name):
        """Load appropriate small LLM for device"""
        if self.device_type == 'android':
            # Use TensorFlow Lite or ONNX Runtime
            return load_tflite_model(model_name)
        else:  # iOS
            # Use Core ML or MLX
            return load_coreml_model(model_name)
    
    def process_query(self, query, modality='text'):
        """
        Process user query and generate instant response
        
        Args:
            query: User input (text, image, or audio)
            modality: 'text', 'image', or 'audio'
        """
        # 1. Preprocess input based on modality
        processed_input = self.preprocess(query, modality)
        
        # 2. Generate instant response with small model
        response = self.small_model.generate(processed_input)
        
        # 3. Extract semantic summary (not raw data)
        semantic_summary = self.extract_semantics(
            query, response, modality
        )
        
        # 4. Compute parameter updates
        param_updates = self.compute_updates(processed_input, response)
        
        return response, semantic_summary, param_updates
    
    def extract_semantics(self, query, response, modality):
        """
        Extract semantic representation without exposing raw data
        """
        return {
            'modality': modality,
            'intent': self.classify_intent(query),
            'entities': self.extract_entities(query),
            'embedding': self.get_embedding(query),  # Encrypted
            'confidence': self.compute_confidence(response)
        }
    
    def compute_updates(self, input_data, output):
        """
        Compute model parameter updates for federated learning
        """
        # Use differential privacy for privacy preservation
        gradients = self.small_model.compute_gradients(input_data, output)
        noisy_gradients = self.add_differential_privacy(gradients)
        return noisy_gradients
    
    def upload_to_cloud(self, semantic_summary, param_updates):
        """
        Upload semantic summaries and updates (NOT raw data)
        """
        encrypted_data = {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'model_name': self.small_model.name,
            'semantic_summary': semantic_summary,
            'param_updates': param_updates,
            'timestamp': datetime.now()
        }
        
        # Encrypt before transmission
        encrypted_payload = self.encrypt(encrypted_data)
        return self.send_to_server(encrypted_payload)
\`\`\`

### 2. Cloud Large Model Server

\`\`\`python
class CloudLargeModelServer:
    """
    Cloud server managing large models and federated aggregation
    """
    def __init__(self, model_configs):
        self.large_models = self.initialize_models(model_configs)
        self.current_model = 'deepseek'  # Default
        self.aggregation_buffer = []
        
    def initialize_models(self, configs):
        """
        Initialize multiple large models with API interfaces
        """
        models = {}
        for config in configs:
            if config['type'] == 'deepseek':
                models['deepseek'] = DeepSeekAPI(config['api_key'])
            elif config['type'] == 'chatgpt':
                models['chatgpt'] = OpenAIAPI(config['api_key'])
            elif config['type'] == 'llama':
                models['llama'] = LlamaModel(config['model_path'])
            # Add more models as needed
        return models
    
    def switch_model(self, model_name):
        """Switch between different large models"""
        if model_name in self.large_models:
            self.current_model = model_name
            return True
        return False
    
    def aggregate_client_updates(self, client_updates):
        """
        Multi-client aggregation with validation
        
        Args:
            client_updates: List of (semantic_summary, param_updates) tuples
        """
        # 1. Validate client updates
        valid_updates = self.validate_updates(client_updates)
        
        # 2. Aggregate semantic summaries
        aggregated_semantics = self.aggregate_semantics(
            [u['semantic_summary'] for u in valid_updates]
        )
        
        # 3. Aggregate parameter updates
        aggregated_params = self.aggregate_parameters(
            [u['param_updates'] for u in valid_updates]
        )
        
        # 4. Update large model with aggregated knowledge
        self.update_large_model(aggregated_semantics, aggregated_params)
        
        return aggregated_params
    
    def validate_updates(self, client_updates):
        """
        Validate client updates for quality and security
        """
        valid_updates = []
        
        for update in client_updates:
            # Check data quality
            if self.check_quality(update):
                # Check for adversarial updates
                if not self.detect_adversarial(update):
                    valid_updates.append(update)
        
        return valid_updates
    
    def federated_distillation(self, small_model_updates):
        """
        Federated distillation: Transfer knowledge from large to small models
        """
        # 1. Generate soft labels from large model
        soft_labels = self.large_models[self.current_model].generate_soft_labels(
            small_model_updates
        )
        
        # 2. Create distilled knowledge package
        distilled_knowledge = {
            'soft_labels': soft_labels,
            'temperature': 3.0,  # Distillation temperature
            'global_patterns': self.extract_global_patterns(),
            'updated_weights': self.compute_distilled_weights()
        }
        
        # 3. Distribute to edge devices
        return distilled_knowledge
    
    def retrain_large_model(self, aggregated_data):
        """
        Retrain large model with aggregated client knowledge
        """
        # Use aggregated semantic summaries for retraining
        training_data = self.prepare_training_data(aggregated_data)
        
        # Fine-tune large model
        self.large_models[self.current_model].fine_tune(
            training_data,
            epochs=3,
            learning_rate=1e-5
        )
\`\`\`

### 3. Federated Learning Coordinator

\`\`\`python
class FederatedHealthCoordinator:
    """
    Coordinates federated learning between edge devices and cloud
    """
    def __init__(self, server, devices):
        self.server = server
        self.devices = devices
        self.round_num = 0
        
    def run_federated_round(self):
        """Execute one round of federated learning"""
        print(f"\\n=== Federated Round {self.round_num + 1} ===")
        
        # 1. Devices process local queries and generate updates
        device_updates = []
        for device in self.devices:
            # Simulate user interactions
            queries = device.get_pending_queries()
            
            for query in queries:
                response, semantics, params = device.process_query(query)
                
                # Upload to cloud (semantic + params, NO raw data)
                device.upload_to_cloud(semantics, params)
                device_updates.append({
                    'device_id': device.device_id,
                    'semantic_summary': semantics,
                    'param_updates': params
                })
        
        # 2. Server aggregates updates
        print(f"Aggregating updates from {len(device_updates)} devices...")
        aggregated_params = self.server.aggregate_client_updates(
            device_updates
        )
        
        # 3. Server performs validation
        print("Validating aggregated model...")
        validation_score = self.server.validate_aggregated_model()
        print(f"Validation score: {validation_score:.4f}")
        
        # 4. Federated distillation
        print("Performing federated distillation...")
        distilled_knowledge = self.server.federated_distillation(
            device_updates
        )
        
        # 5. Distribute updated knowledge to devices
        print("Distributing knowledge updates to devices...")
        for device in self.devices:
            device.update_from_distillation(distilled_knowledge)
        
        # 6. Retrain large model
        if self.round_num % 5 == 0:  # Every 5 rounds
            print("Retraining large model...")
            self.server.retrain_large_model(device_updates)
        
        self.round_num += 1
    
    def run_continuous_learning(self, num_rounds=100):
        """Run continuous federated learning"""
        for _ in range(num_rounds):
            self.run_federated_round()
            time.sleep(60)  # Wait between rounds
\`\`\`

### 4. Heterogeneous Device Management

\`\`\`python
class HeterogeneousDeviceManager:
    """
    Manage 20 heterogeneous devices with different capabilities
    """
    def __init__(self):
        self.devices = []
        self.setup_heterogeneous_devices()
        
    def setup_heterogeneous_devices(self):
        """
        Setup 20 devices with heterogeneous characteristics
        """
        # Android devices (10)
        android_models = [
            ('tinyllama', 'low'),      # Low compute
            ('mobilellm', 'medium'),   # Medium compute
            ('phi-2', 'high'),         # High compute
            ('gemma-2b', 'medium'),
            ('tinyllama', 'low'),
            ('mobilellm', 'medium'),
            ('phi-2', 'high'),
            ('gemma-2b', 'medium'),
            ('tinyllama', 'low'),
            ('mobilellm', 'high')
        ]
        
        for i, (model, compute) in enumerate(android_models):
            device = EdgeHealthAssistant(
                device_id=f'android_{i}',
                model_name=model,
                device_type='android'
            )
            device.compute_capability = compute
            self.devices.append(device)
        
        # iOS devices (10)
        ios_models = [
            ('phi-2', 'high'),         # iPhone 15 Pro
            ('gemma-2b', 'high'),      # iPhone 15 Pro Max
            ('mobilellm', 'medium'),   # iPhone 14
            ('tinyllama', 'medium'),   # iPhone 13
            ('phi-2', 'high'),         # iPhone 15
            ('gemma-2b', 'high'),      # iPhone 14 Pro
            ('mobilellm', 'medium'),   # iPhone 13 Pro
            ('tinyllama', 'low'),      # iPhone 12
            ('phi-2', 'high'),         # iPhone 15 Pro
            ('mobilellm', 'medium')    # iPhone 14
        ]
        
        for i, (model, compute) in enumerate(ios_models):
            device = EdgeHealthAssistant(
                device_id=f'ios_{i}',
                model_name=model,
                device_type='ios'
            )
            device.compute_capability = compute
            self.devices.append(device)
    
    def get_device_statistics(self):
        """Get statistics about device heterogeneity"""
        stats = {
            'total_devices': len(self.devices),
            'android': sum(1 for d in self.devices if d.device_type == 'android'),
            'ios': sum(1 for d in self.devices if d.device_type == 'ios'),
            'models': {},
            'compute_levels': {'low': 0, 'medium': 0, 'high': 0}
        }
        
        for device in self.devices:
            # Count models
            model_name = device.small_model.name
            stats['models'][model_name] = stats['models'].get(model_name, 0) + 1
            
            # Count compute levels
            stats['compute_levels'][device.compute_capability] += 1
        
        return stats
\`\`\`

## Complete Medical Scenario Example

\`\`\`python
# 1. Setup heterogeneous devices
device_manager = HeterogeneousDeviceManager()
devices = device_manager.devices

print("Device Statistics:")
print(device_manager.get_device_statistics())

# 2. Setup cloud server with multiple LLMs
model_configs = [
    {'type': 'deepseek', 'api_key': 'your_deepseek_key'},
    {'type': 'chatgpt', 'api_key': 'your_openai_key'},
    {'type': 'llama', 'model_path': './models/llama-70b'}
]
server = CloudLargeModelServer(model_configs)

# 3. Initialize federated coordinator
coordinator = FederatedHealthCoordinator(server, devices)

# 4. Run continuous federated learning
coordinator.run_continuous_learning(num_rounds=100)

# 5. Switch large model backend dynamically
server.switch_model('chatgpt')  # Switch to ChatGPT
coordinator.run_federated_round()

server.switch_model('llama')    # Switch to Llama
coordinator.run_federated_round()
\`\`\`

## Privacy and Security Features

### 1. Data Privacy
- Only semantic summaries uploaded (no raw data)
- Differential privacy on parameter updates
- End-to-end encryption

### 2. Model Security
- Adversarial update detection
- Secure aggregation protocols
- Model watermarking

### 3. Compliance
- HIPAA compliance for medical data
- GDPR compliance for EU users
- Local data storage only

## Performance Optimization

### 1. Adaptive Participation
- Select devices based on compute capability
- Battery-aware scheduling
- Network-aware communication

### 2. Model Compression
- Quantization for edge models
- Pruning for faster inference
- Knowledge distillation for size reduction

### 3. Communication Efficiency
- Gradient compression
- Sparse updates
- Federated dropout

## Key Features

- ✅ 20 heterogeneous devices (Android + iOS)
- ✅ Multiple LLM support (DeepSeek, ChatGPT, Llama, etc.)
- ✅ Small models for instant on-device responses
- ✅ Large models for aggregation and validation
- ✅ Federated distillation for knowledge transfer
- ✅ Privacy-preserving (semantic summaries only)
- ✅ Cross-modal support (text, image, audio)
- ✅ Medical scenario validation
`
  }
}

export default function Examples() {
  const { exampleId } = useParams()
  const [currentExample, setCurrentExample] = useState(null)
  const [htmlContent, setHtmlContent] = useState('')

  useEffect(() => {
    if (exampleId && examples[exampleId]) {
      setCurrentExample(exampleId)
      
      // Configure marked options
      marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
      })
      
      const html = marked(examples[exampleId].content)
      setHtmlContent(html)
    } else {
      // Show table of contents
      setCurrentExample(null)
      const tocHtml = marked(`# Application Examples

This section provides detailed implementation examples for the three main federated learning features.

## Table of Contents

### [1. Federated CNN Training](/examples/cnn-training)
Standard federated learning with convolutional neural networks. Each client trains a CNN locally on their data, and the central server aggregates model updates until convergence.

**Key Topics:**
- Client-side local data management
- Model training and parameter updates
- Server-side FedAvg aggregation
- MNIST dataset validation
- Multiple aggregation algorithms

---

### [2. Federated Optimization](/examples/optimization)
Surrogate model-based optimization for finding optimal solutions. Each client trains a surrogate model, and the server selects the best solution after aggregation.

**Key Topics:**
- Surrogate model training
- Server-side aggregation
- Optimal solution selection
- Solution distribution to clients
- Convergence monitoring

---

### [3. Large-Small Model Collaboration](/examples/large-small-collaboration)
Hierarchical federated learning with edge devices running small models and cloud running large models. Features 20 heterogeneous devices with multiple LLMs.

**Key Topics:**
- Personal health management assistant scenario
- 20 heterogeneous devices (Android + iOS)
- Small models for instant responses
- Large models for aggregation and validation
- Federated distillation
- Privacy-preserving semantic summaries
- Cross-modal support (text, image, audio)

---

## Getting Started

Click on any example above to view detailed implementation code, architecture diagrams, and usage examples.
`)
      setHtmlContent(tocHtml)
    }
  }, [exampleId])

  const tocItems = [
    { id: 'cnn-training', title: 'CNN Training', icon: '🧠' },
    { id: 'optimization', title: 'Optimization', icon: '🎯' },
    { id: 'large-small-collaboration', title: 'Large-Small Model', icon: '🤝' }
  ]

  return (
    <div className={styles.documentation}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h2>💡 Examples</h2>
        </div>
        <nav className={styles.toc}>
          <Link
            to="/examples"
            className={`${styles.tocItem} ${!currentExample ? styles.active : ''}`}
          >
            <span className={styles.tocIcon}>📋</span>
            <span className={styles.tocTitle}>Overview</span>
          </Link>
          {tocItems.map(item => (
            <Link
              key={item.id}
              to={`/examples/${item.id}`}
              className={`${styles.tocItem} ${currentExample === item.id ? styles.active : ''}`}
            >
              <span className={styles.tocIcon}>{item.icon}</span>
              <span className={styles.tocTitle}>{item.title}</span>
            </Link>
          ))}
        </nav>
      </aside>
      
      <main className={styles.content}>
        <div className={styles.contentWrapper}>
          <article 
            className={styles.markdown}
            dangerouslySetInnerHTML={{ __html: htmlContent }}
          />
        </div>
      </main>
    </div>
  )
}
