import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { marked } from 'marked'
import styles from './Documentation.module.css'
import CNNTrainingDiagram from './CNNTrainingDiagram'
import OptimizationDiagram from './OptimizationDiagram'
import CloudEdgeDiagram from './CloudEdgeDiagram'

// Examples content mapping
const examples = {
  'cnn-training': {
    title: 'Federated CNN Training',
    content: `# Feature 1: Federated CNN Training 

## Overview

Based on the **Fed-Evo** framework, this feature implements a hierarchical federated learning system for MNIST handwritten digit recognition. The architecture consists of a central management server, regional aggregation nodes, and edge devices.

## Architecture Flow

The system uses a three-tier architecture to manage federated learning tasks efficiently:

\`\`\`
      ┌─────────────────┐
      │  Center Server  │
      │(Task Management)│
      └────────┬────────┘
               │ rabbitMQ
               ▼
      ┌─────────────────┐
      │ Regional Node   │
      │ (fed Server) │◄──────┐
      └────────┬────────┘       │
               │MQTT     │ Aggregation
      ┌────────┴────────┐       │
      │                 │       │
┌─────▼─────┐     ┌─────▼─────┐ │
│  Device 1 │     │  Device 2 │ │
│(Flower Cl)│     │(Flower Cl)│ │
└───────────┘     └───────────┘
\`\`\`

## Key Components

- **Center Server**: Manages global tasks and coordinates regional nodes.
- **Regional Node (\`regional/\`)**: Runs the **Flower Server**, manages the \`Fed-Evo\` strategy, and aggregates updates from connected devices.
- **Edge Device (\`device/\`)**: Runs the **Flower Client**, performs local training using \`MNISTTrainer\`, and uploads model updates.
- **Shared Model (\`shared/\`)**: Standard CNN definition used across all nodes.

## Implementation Details

### 1. Shared Model Architecture
Defined in \`shared/mnist_model.py\`:

\`\`\`python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
\`\`\`

### 2. Client-Side Implementation

**Flower Client Wrapper (\`device/flower_client.py\`)**:
\`\`\`python
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, device_id, trainer, server_address, ...):
        self.trainer = trainer  # Instance of MNISTTrainer
        
    def fit(self, parameters, config):
        """Receive global parameters, train locally, return updates"""
        # Train using local trainer
        updated_params, num_examples, metrics = self.trainer.fit(parameters, config)
        return updated_params, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate global parameters on local data"""
        loss, num_examples, metrics = self.trainer.evaluate(parameters, config)
        return loss, num_examples, metrics
\`\`\`

**Local Trainer (\`device/mnist_trainer.py\`)**:
\`\`\`python
class MNISTTrainer:
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        # Local training loop (Standard PyTorch)
        for epoch in range(3):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), metrics
\`\`\`

### 3. Regional Server Aggregation

Managed by \`FedServerManager\` in \`regional/app/fed/server_manager.py\`. It uses Flower's built-in strategies.

\`\`\`python
def _run_server(self):
    # Configure Strategy (Fed-Evo)
    strategy = fl.server.strategy.Fed-Evo(
        fraction_fit=1.0,        # Sample 100% of available clients
        min_fit_clients=1,       # Minimum clients to train
        min_evaluate_clients=1,  # Minimum clients to evaluate
        evaluate_fn=self._evaluate_fn,
    )
    
    # Start Flower Server
    fl.server.start_server(
        server_address=f"0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
\`\`\`

## How to Run

1. **Start the Regional Node**:
   The regional server waits for task distribution from the center or manual start.
   \`\`\`bash
   python regional/main.py
   \`\`\`

2. **Start Edge Devices**:
   Start multiple devices to connect to the regional node.
   \`\`\`bash
   # Start a device with ID 'dev1'
   python device/main.py --device_id dev1 --server_address localhost:8080
   
   # Start another device
   python device/main.py --device_id dev2 --server_address localhost:8080
   \`\`\`

## Key Features

- ✅ **Hierarchical Architecture**: Center -> Regional -> Device.
- ✅ **Flower Framework**: Robust FL communication via gRPC.
- ✅ **Data Privacy**: Raw data stays on device; only model weights are transmitted.
- ✅ **Heterogeneity**: Supports custom data partitioning logic in \`MNISTTrainer\`.
`
  },
  'optimization': {
    title: 'Federated Optimization',
    content: `# Feature 2: Federated Surrogate Optimization

## Overview

This feature enables collaborative training of a **Global Surrogate Model** across multiple edge devices. Instead of static datasets, clients dynamically sample from their local search spaces, evaluate objective functions, and train local surrogate models. The regional server aggregates these models to build a powerful global estimator of the objective landscape.

## Architecture Flow

The system employs a three-tier architecture customized for optimization tasks:

\`\`\`
      ┌─────────────────┐
      │  Center Server  │
      │(Optimization Mgt)│
      └────────┬────────┘
               │ Task (Search Space)
               ▼
      ┌─────────────────┐
      │ Regional Node   │
      │(Surrogate Aggr) │◄──────┐
      └────────┬────────┘       │
               │ Aggregation    │ Weights
      ┌────────┴────────┐       │
      │                 │       │
┌─────▼─────┐     ┌─────▼─────┐ │
│  Device 1 │     │  Device 2 │ │
│(Sampler)  │     │(Sampler)  │ │
└───────────┘     └───────────┘
\`\`\`

## Key Differences from Standard FL

- **Model Role**: Approximator (Surrogate) vs. Classifier
- **Objective**: Minimize approximation error to find global optima

## Implementation Details

### 1. Surrogate Model Definition
A simple MLP to approximate the objective function.

\`\`\`python
class SurrogateNet(nn.Module):
    def __init__(self, input_dim=10):
        super(SurrogateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predicts objective value
        )
    
    def forward(self, x):
        return self.net(x)
\`\`\`

### 2. Client-Side: Dynamic Training

The client generates training data on-the-fly by sampling the search space and evaluating the "real" objective function.

\`\`\`python
class OptimizationTrainer:
    def __init__(self, objective_func, search_space):
        self.model = SurrogateNet()
        self.objective_func = objective_func
        
    def generate_data(self, n_samples=32):
        """Dynamic data generation"""
        X = self.search_space.sample(n_samples)
        y = [self.objective_func(x) for x in X]
        return torch.tensor(X), torch.tensor(y)

    def fit(self, parameters, config):
        """Client local training method"""
        self.set_parameters(parameters)
        
        # 1. Generate fresh training data
        X_train, y_train = self.generate_data()
        
        # 2. Train surrogate model locally
        self.train_step(X_train, y_train)
        
        return self.get_parameters(), len(X_train), {}
\`\`\`

### 3. Server-Side: Aggregation

The Regional Node aggregates the surrogate models from clients to form a global surrogate.

\`\`\`python
def start_surrogate_aggregation():
    # Use FedAvg to build a robust Global Surrogate
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2
    )
    
    server.start(
        strategy=strategy,
        rounds=20
    )
\`\`\`

## Key Features

- ✅ **Distributed Sampling**: Explore different regions of the search space in parallel.
- ✅ **Global Approximation**: Server builds a global view of the objective landscape.
- ✅ **Efficiency**: Reduces the number of expensive real-world evaluations by sharing knowledge.
`
  },
  'large-small-collaboration': {
    title: 'Large-Small Model Collaboration',
    content: `# Feature 3: Cloud-Edge Model Collaboration

## Overview

This feature implements a **cloud-edge collaborative training system** with a central server API. The central server provides endpoints for model upload/download and input-output data collection. Edge devices download compressed models, run inference, and upload input-output pairs back to the server for continuous model improvement.

## Workflow

The system operates in a continuous loop:

1. **Model Upload**: Upload trained models (large/compressed) to central server
2. **Model Download**: Edge devices download compressed models from server
3. **Edge Inference**: Devices run inference and collect input-output pairs
4. **Data Upload**: Upload input-output pairs to central server
5. **Model Retraining**: Server retrains models with collected data

## System Architecture

\`\`\`
      ┌─────────────────────────────────┐
      │    Central Server (API)         │
      │  ┌──────────────────────────┐   │
      │  │  POST /api/models/upload │   │
      │  │  GET  /api/models/download│  │
      │  │  POST /api/data/upload   │   │
      │  └──────────────────────────┘   │
      │  ┌──────────────────────────┐   │
      │  │  Model Storage (MinIO)   │   │
      │  │  Data Storage (Database) │   │
      │  └──────────────────────────┘   │
      └───────────┬─────────────────────┘
                  │ HTTP/REST API
         ┌────────┴────────┐
         │                 │
    ┌────▼────┐      ┌────▼────┐
    │ Device1 │      │ Device2 │
    │ (Edge)  │      │ (Edge)  │
    └─────────┘      └─────────┘
\`\`\`

## Implementation Details

### 1. Central Server API Endpoints

The central server provides RESTful APIs for model and data management.

\`\`\`python
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
MODEL_STORAGE_PATH = './models/'
DATA_STORAGE_PATH = './data/'
ALLOWED_EXTENSIONS = {'pth', 'pt', 'onnx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================
# Model Upload API
# ============================================
@app.route('/api/models/upload', methods=['POST'])
def upload_model():
    """
    Upload a model to the central server
    
    Request:
        - model_file: Model file (multipart/form-data)
        - model_type: 'large' or 'compressed'
        - model_name: Name of the model
        - version: Model version
    
    Response:
        - model_id: Unique identifier for the uploaded model
        - storage_path: Path where model is stored
    """
    if 'model_file' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model_file']
    model_type = request.form.get('model_type', 'compressed')
    model_name = request.form.get('model_name', 'model')
    version = request.form.get('version', '1.0')
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{model_name}_{model_type}_{version}_{timestamp}"
        storage_filename = f"{model_id}.pth"
        storage_path = os.path.join(MODEL_STORAGE_PATH, storage_filename)
        
        # Save model file
        os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
        file.save(storage_path)
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'version': version,
            'upload_time': timestamp,
            'storage_path': storage_path,
            'file_size': os.path.getsize(storage_path)
        }
        
        metadata_path = os.path.join(MODEL_STORAGE_PATH, f"{model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'storage_path': storage_path,
            'message': f'Model {model_id} uploaded successfully'
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400


# ============================================
# Model Download API
# ============================================
@app.route('/api/models/download', methods=['GET'])
def download_model():
    """
    Download a model from the central server
    
    Query Parameters:
        - model_id: Unique identifier of the model (optional)
        - model_name: Name of the model (optional)
        - model_type: 'large' or 'compressed' (default: 'compressed')
        - latest: If true, download the latest version (default: true)
    
    Response:
        - Binary model file
    """
    model_id = request.args.get('model_id')
    model_name = request.args.get('model_name')
    model_type = request.args.get('model_type', 'compressed')
    latest = request.args.get('latest', 'true').lower() == 'true'
    
    # Find model file
    if model_id:
        # Download specific model by ID
        model_path = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.pth")
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model {model_id} not found'}), 404
    
    elif model_name and latest:
        # Download latest version of a model
        model_files = [f for f in os.listdir(MODEL_STORAGE_PATH) 
                      if f.startswith(f"{model_name}_{model_type}") and f.endswith('.pth')]
        
        if not model_files:
            return jsonify({'error': f'No models found for {model_name}'}), 404
        
        # Sort by timestamp (latest first)
        model_files.sort(reverse=True)
        model_path = os.path.join(MODEL_STORAGE_PATH, model_files[0])
    
    else:
        return jsonify({'error': 'Must provide model_id or model_name'}), 400
    
    return send_file(model_path, as_attachment=True)


# ============================================
# Input-Output Data Upload API
# ============================================
@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """
    Upload input-output pairs from edge devices
    
    Request Body (JSON):
        {
            "device_id": "device_001",
            "model_id": "resnet_compressed_1.0_20231218",
            "data": [
                {
                    "input": [...],  # Input tensor as list
                    "output": [...], # Output tensor as list
                    "timestamp": 1702901234.56
                },
                ...
            ]
        }
    
    Response:
        - data_id: Unique identifier for uploaded data batch
        - count: Number of data points uploaded
    """
    if not request.json:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    device_id = request.json.get('device_id')
    model_id = request.json.get('model_id')
    data_points = request.json.get('data', [])
    
    if not device_id or not data_points:
        return jsonify({'error': 'Missing device_id or data'}), 400
    
    # Generate unique data batch ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_id = f"{device_id}_{timestamp}"
    
    # Save data to storage
    os.makedirs(DATA_STORAGE_PATH, exist_ok=True)
    data_file_path = os.path.join(DATA_STORAGE_PATH, f"{data_id}.json")
    
    data_batch = {
        'data_id': data_id,
        'device_id': device_id,
        'model_id': model_id,
        'upload_time': timestamp,
        'count': len(data_points),
        'data': data_points
    }
    
    with open(data_file_path, 'w') as f:
        json.dump(data_batch, f, indent=2)
    
    return jsonify({
        'success': True,
        'data_id': data_id,
        'count': len(data_points),
        'message': f'Uploaded {len(data_points)} data points from {device_id}'
    }), 200


# ============================================
# Model List API (Helper)
# ============================================
@app.route('/api/models/list', methods=['GET'])
def list_models():
    """
    List all available models
    
    Query Parameters:
        - model_type: Filter by model type (optional)
    
    Response:
        - models: List of model metadata
    """
    model_type = request.args.get('model_type')
    
    models = []
    for filename in os.listdir(MODEL_STORAGE_PATH):
        if filename.endswith('_metadata.json'):
            with open(os.path.join(MODEL_STORAGE_PATH, filename), 'r') as f:
                metadata = json.load(f)
                if not model_type or metadata['model_type'] == model_type:
                    models.append(metadata)
    
    # Sort by upload time (latest first)
    models.sort(key=lambda x: x['upload_time'], reverse=True)
    
    return jsonify({'models': models}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
\`\`\`

### 2. Edge Device Client

Edge devices interact with the central server API to download models and upload data.

\`\`\`python
import requests
import torch
import numpy as np
import time
import json

class EdgeDeviceClient:
    def __init__(self, device_id, server_url='http://localhost:5000'):
        self.device_id = device_id
        self.server_url = server_url
        self.model = None
        self.current_model_id = None
        self.data_buffer = []
        
    def download_model(self, model_name='resnet', model_type='compressed'):
        """
        Download the latest compressed model from central server
        """
        try:
            # Request latest model
            response = requests.get(
                f'{self.server_url}/api/models/download',
                params={
                    'model_name': model_name,
                    'model_type': model_type,
                    'latest': 'true'
                }
            )
            
            if response.status_code == 200:
                # Save model locally
                model_path = f'./local_models/{self.device_id}_model.pth'
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                
                # Load model
                self.model = torch.load(model_path)
                self.model.eval()
                self.current_model_id = f"{model_name}_{model_type}"
                
                print(f"[{self.device_id}] Model downloaded successfully")
                return True
            else:
                print(f"[{self.device_id}] Failed to download model: {response.text}")
                return False
                
        except Exception as e:
            print(f"[{self.device_id}] Error downloading model: {e}")
            return False
    
    def run_inference(self, input_data):
        """
        Run inference and collect input-output pairs
        """
        if self.model is None:
            raise ValueError("No model loaded. Call download_model() first.")
        
        with torch.no_grad():
            output = self.model(input_data)
        
        # Store input-output pair
        self.data_buffer.append({
            'input': input_data.cpu().numpy().tolist(),
            'output': output.cpu().numpy().tolist(),
            'timestamp': time.time()
        })
        
        return output
    
    def upload_data(self, batch_size=100):
        """
        Upload collected input-output pairs to central server
        """
        if len(self.data_buffer) < batch_size:
            print(f"[{self.device_id}] Buffer size {len(self.data_buffer)} < {batch_size}, skipping upload")
            return False
        
        try:
            # Prepare payload
            payload = {
                'device_id': self.device_id,
                'model_id': self.current_model_id,
                'data': self.data_buffer[:batch_size]
            }
            
            # Upload to server
            response = requests.post(
                f'{self.server_url}/api/data/upload',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[{self.device_id}] Uploaded {result['count']} data points")
                
                # Clear uploaded data from buffer
                self.data_buffer = self.data_buffer[batch_size:]
                return True
            else:
                print(f"[{self.device_id}] Failed to upload data: {response.text}")
                return False
                
        except Exception as e:
            print(f"[{self.device_id}] Error uploading data: {e}")
            return False
\`\`\`

### 3. Usage Example

\`\`\`python
# ============================================
# Server Side: Upload a compressed model
# ============================================
import requests

# Upload compressed model to server
with open('compressed_model.pth', 'rb') as f:
    files = {'model_file': f}
    data = {
        'model_type': 'compressed',
        'model_name': 'resnet',
        'version': '1.0'
    }
    response = requests.post('http://localhost:5000/api/models/upload', 
                            files=files, data=data)
    print(response.json())


# ============================================
# Edge Device: Download model and run inference
# ============================================

# Initialize edge device client
device = EdgeDeviceClient(device_id='device_001', server_url='http://localhost:5000')

# Download latest compressed model
device.download_model(model_name='resnet', model_type='compressed')

# Run inference and collect data
for i in range(500):
    # Get input data (e.g., from camera, sensor, etc.)
    input_data = torch.randn(1, 3, 224, 224)  # Example input
    
    # Run inference
    output = device.run_inference(input_data)
    
    # Upload data every 100 samples
    if (i + 1) % 100 == 0:
        device.upload_data(batch_size=100)

print(f"Inference complete. Buffer size: {len(device.data_buffer)}")
\`\`\`

## Key Features

- ✅ **RESTful API**: Standard HTTP endpoints for model and data management
- ✅ **Model Upload**: Upload trained models (large/compressed) to central server
- ✅ **Model Download**: Edge devices download latest compressed models
- ✅ **Data Upload**: Upload input-output pairs for model improvement
- ✅ **Model Versioning**: Track different versions of models
- ✅ **Metadata Management**: Store and retrieve model metadata
- ✅ **Privacy-Preserving**: Only anonymized input-output pairs are uploaded
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

      // Configure marked options to allow HTML
      marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false,
        sanitize: false  // Allow HTML/SVG rendering
      })

      // Use marked.parse() instead of marked() for better HTML support
      const html = marked.parse(examples[exampleId].content, {
        sanitize: false,
        gfm: true
      })
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
- Server-side Fed-Evo aggregation
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
Cloud-edge collaborative training system. Cloud trains large models and deploys compressed versions to edge devices. Edge devices upload input-output pairs for continuous model improvement.

**Key Topics:**
- Cloud training and model compression
- Edge deployment and inference
- Input-output data collection
- Cloud retraining with edge data
- Continuous improvement loop

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
          {currentExample === 'cnn-training' ? (
            <article className={styles.markdown}>
              <h1>Feature 1: Federated CNN Training</h1>
              <h2>Overview</h2>
              <div dangerouslySetInnerHTML={{
                __html: htmlContent
                  .replace(/<h1>.*?<\/h1>/, '')
                  .replace(/<h2>Overview<\/h2>/, '')
                  .split('<h2>Architecture Flow</h2>')[0]
              }} />
              <h2>Architecture Flow</h2>
              <CNNTrainingDiagram />
              <div dangerouslySetInnerHTML={{
                __html: '<div>' + htmlContent.split('<h2>Architecture Flow</h2>')[1].split('<h2>Key Components</h2>')[0] + '</div>'
              }} />
              <div dangerouslySetInnerHTML={{
                __html: '<div><h2>Key Components</h2>' + htmlContent.split('<h2>Key Components</h2>')[1] + '</div>'
              }} />
            </article>
          ) : currentExample === 'optimization' ? (
            <article className={styles.markdown}>
              <h1>Feature 2: Federated Surrogate Optimization</h1>
              <h2>Overview</h2>
              <div dangerouslySetInnerHTML={{
                __html: htmlContent
                  .replace(/<h1>.*?<\/h1>/, '')
                  .replace(/<h2>Overview<\/h2>/, '')
                  .split('<h2>Architecture Flow</h2>')[0]
              }} />
              <h2>Architecture Flow</h2>
              <OptimizationDiagram />
              <div dangerouslySetInnerHTML={{
                __html: '<div>' + htmlContent.split('<h2>Architecture Flow</h2>')[1].split('<h2>Key Differences from Standard FL</h2>')[0] + '</div>'
              }} />
              <div dangerouslySetInnerHTML={{
                __html: '<div><h2>Key Differences from Standard FL</h2>' + htmlContent.split('<h2>Key Differences from Standard FL</h2>')[1] + '</div>'
              }} />
            </article>
          ) : currentExample === 'large-small-collaboration' ? (
            <article className={styles.markdown}>
              <h1>Feature 3: Cloud-Edge Model Collaboration</h1>
              <h2>Overview</h2>
              <div dangerouslySetInnerHTML={{
                __html: htmlContent
                  .replace(/<h1>.*?<\/h1>/, '')
                  .replace(/<h2>Overview<\/h2>/, '')
                  .split('<h2>Workflow</h2>')[0]
              }} />
              <h2>Workflow</h2>
              <div dangerouslySetInnerHTML={{
                __html: '<div>' + htmlContent.split('<h2>Workflow</h2>')[1].split('<h2>System Architecture</h2>')[0] + '</div>'
              }} />
              <h2>System Architecture</h2>
              <CloudEdgeDiagram />
              <div dangerouslySetInnerHTML={{
                __html: '<div>' + htmlContent.split('<h2>System Architecture</h2>')[1].split('<h2>Implementation Details</h2>')[0] + '</div>'
              }} />
              <div dangerouslySetInnerHTML={{
                __html: '<div><h2>Implementation Details</h2>' + htmlContent.split('<h2>Implementation Details</h2>')[1] + '</div>'
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
          