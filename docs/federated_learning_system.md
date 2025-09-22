
## 联邦学习核心架构

### 1. 服务器端核心功能

#### 全局模型管理
```python
# 服务器维护一个全局模型
global_model = LeNet()  # 可以是任何深度学习模型

# 模型状态管理
current_round = 0
client_updates = []
MIN_CLIENTS = 2  # 控制参与聚合的最小客户端数量
```

#### 客户端数量控制
```python
# 控制参与训练的客户端数量
MIN_CLIENTS = 2  # 最少需要2个客户端才能聚合

# 动态调整参与策略
def should_aggregate():
    return len(client_updates) >= MIN_CLIENTS

# 等待足够客户端后再聚合
if len(client_updates) >= MIN_CLIENTS:
    # 执行聚合
    aggregated_model = custom_aggregation_algorithm(client_updates)
```

#### 自定义聚合算法
```python
# 1. 简单平均聚合 (FedAvg)
def fedavg_aggregation(client_models):
    averaged_model = copy.deepcopy(client_models[0])
    for key in averaged_model.keys():
        for i in range(1, len(client_models)):
            averaged_model[key] += client_models[i][key]
        averaged_model[key] = torch.div(averaged_model[key], len(client_models))
    return averaged_model

# 2. 加权聚合 (按数据量)
def weighted_aggregation(client_models, data_sizes):
    total_samples = sum(data_sizes)
    averaged_model = copy.deepcopy(client_models[0])
    for key in averaged_model.keys():
        weighted_sum = 0
        for model, size in zip(client_models, data_sizes):
            weighted_sum += model[key] * size
        averaged_model[key] = weighted_sum / total_samples
    return averaged_model

# 3. 自定义聚合策略
def custom_aggregation_algorithm(client_models):
    # 可以根据业务需求自定义聚合逻辑
    # 例如：异常值过滤、模型质量评估等
    return your_custom_aggregation(client_models)
```

### 2. 客户端核心功能

#### 本地训练实现
```python
class FederatedClient:
    def __init__(self, server_address, client_id):
        self.server_address = server_address
        self.client_id = client_id
        self.model = LeNet()  # 本地模型
        
        # 准备本地数据
        self.setup_local_data()
    
    def setup_local_data(self):
        """准备本地训练数据"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST('./data', train=True, download=True,
                                    transform=transform)
        # 模拟分布式数据 - 每个客户端使用不同子集
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
        """执行本地模型训练"""
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
        """联邦学习训练循环"""
        for round in range(rounds):
            print(f"Round {round + 1}")
            
            # 1. 从服务器获取全局模型
            self.get_global_model()
            
            # 2. 在本地数据上训练
            local_model = self.local_training(epochs=1)
            
            # 3. 发送更新到服务器
            self.send_model_update(local_model)
```

## 网络搭建
### 运行
client -> server 
### 启动
1. 中央服务器: 启动 flower server 
2. 区域服务器: 发送任务（中央服务器ip ，模型类型等）到区域服务器， 区域服务器转发到mqtt
3. 设备：监听mqtt，触发任务

## 支持的模型架构

### 1. 深度学习框架支持

#### 服务器端框架
- **PyTorch**: 动态图、易于调试
- **TensorFlow**: 静态图、生产部署
- **JAX**: 高性能计算、函数式编程

#### 移动端框架
- **TensorFlow Lite**: Android、iOS轻量级部署
- **MLX**: Apple Silicon优化，macOS/iOS专用
- **ONNX Runtime**: 跨平台模型推理

#### 高级框架
- **Transformers**: Hugging Face预训练模型库
- **FastAI**: 高级API、快速原型开发

#### 平台支持
- **Android**: TensorFlow Lite、ONNX Runtime
- **iOS**: Core ML、TensorFlow Lite、MLX


### 2. 模型优化策略（未探索）

#### 模型压缩
- **量化**: INT8/FP16精度降低，减少模型大小
- **剪枝**: 移除不重要的权重和神经元
- **知识蒸馏**: 大模型到小模型的知识转移

#### 动态调整
- **自适应架构**: 根据设备性能调整模型复杂度
- **分层训练**: 不同层使用不同训练策略
- **渐进式训练**: 从简单到复杂逐步训练
