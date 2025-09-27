"""
MNIST 训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from loguru import logger
from typing import List, Tuple, Dict, Any

from mnist_model import create_model, get_model_parameters, set_model_parameters


class MNISTTrainer:
    """MNIST 训练器"""
    
    def __init__(self, device_id: str, task_data: dict):
        self.device_id = device_id
        self.task_data = task_data
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"设备 {device_id} 使用计算设备: {self.device}")
        
        # 初始化数据
        self._prepare_data()
    
    def _prepare_data(self):
        """准备训练数据"""
        try:
            # 数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 加载 MNIST 数据集
            train_dataset = datasets.MNIST(
                root='./data', 
                train=True, 
                download=True, 
                transform=transform
            )
            
            test_dataset = datasets.MNIST(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
            
            # 根据设备ID分配不同的数据子集（模拟数据异构性）
            subset_indices = self._get_device_data_indices(train_dataset)
            
            # 创建数据子集
            train_subset = Subset(train_dataset, subset_indices)
            
            # 创建数据加载器
            self.train_loader = DataLoader(
                train_subset, 
                batch_size=32, 
                shuffle=True
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=32, 
                shuffle=False
            )
            
            logger.info(f"设备 {self.device_id} 数据准备完成，训练样本数: {len(subset_indices)}")
            
        except Exception as e:
            logger.error(f"准备数据失败: {e}")
            raise
    
    def _get_device_data_indices(self, dataset) -> List[int]:
        """根据设备ID获取数据索引（模拟数据异构性）"""
        # 简单的数据分配策略：根据设备ID的哈希值分配不同的数字类别
        device_hash = hash(self.device_id) % 3
        
        if device_hash == 0:
            # 设备1：数字 0, 1, 2
            target_classes = [0, 1, 2]
        elif device_hash == 1:
            # 设备2：数字 3, 4, 5
            target_classes = [3, 4, 5]
        else:
            # 设备3：数字 6, 7, 8, 9
            target_classes = [6, 7, 8, 9]
        
        # 获取目标类别的数据索引
        indices = []
        for idx, (_, target) in enumerate(dataset):
            if target.item() in target_classes:
                indices.append(idx)
        
        # 限制每个设备的数据量（模拟真实场景）
        max_samples = 1000
        if len(indices) > max_samples:
            indices = indices[:max_samples]
        
        return indices
    
    def get_parameters(self) -> List[np.ndarray]:
        """获取模型参数"""
        if self.model is None:
            self.model = create_model()
            self.model.to(self.device)
        
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """设置模型参数"""
        if self.model is None:
            self.model = create_model()
            self.model.to(self.device)
        
        set_model_parameters(self.model, parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """训练模型"""
        try:
            # 设置模型参数
            self.set_parameters(parameters)
            
            # 设置优化器
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            
            # 训练模型
            self.model.train()
            num_examples = 0
            
            for epoch in range(3):  # 本地训练3个epoch
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    num_examples += len(data)
            
            # 获取更新后的参数
            updated_parameters = self.get_parameters()
            
            # 计算指标
            metrics = self._evaluate()
            
            logger.info(f"设备 {self.device_id} 训练完成，样本数: {num_examples}, 准确率: {metrics['accuracy']:.4f}")
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            raise
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """评估模型"""
        try:
            # 设置模型参数
            self.set_parameters(parameters)
            
            # 评估模型
            metrics = self._evaluate()
            
            return metrics['loss'], len(self.test_loader.dataset), metrics
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            raise
    
    def _evaluate(self) -> Dict[str, Any]:
        """内部评估方法"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        accuracy = correct / total
        loss = test_loss / len(self.test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'correct': correct,
            'total': total
        }
