# flower_edge_client.py - 边缘设备客户端实现
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Dict
import logging
import os
import json
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """简单的CNN模型，适合边缘设备"""

    def __init__(self, num_classes: int = 10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


class EdgeDataset(Dataset):
    """边缘设备本地数据集"""

    def __init__(self, data_path: str = None, simulate: bool = True):
        self.simulate = simulate

        if simulate:
            # 模拟边缘设备的小规模数据集
            self.data = torch.randn(1000, 1, 28, 28)  # 1000个样本
            self.targets = torch.randint(0, 10, (1000,))
        else:
            # 从文件加载真实数据
            self._load_data(data_path)

    def _load_data(self, data_path: str):
        """从文件加载数据"""
        if os.path.exists(data_path):
            data = torch.load(data_path)
            self.data = data['images']
            self.targets = data['labels']
        else:
            logger.warning(f"Data path {data_path} not found, using simulated data")
            self.data = torch.randn(1000, 1, 28, 28)
            self.targets = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class EdgeClient(fl.client.NumPyClient):
    """边缘设备客户端"""

    def __init__(self, client_id: str, device_type: str = "mobile"):
        self.client_id = client_id
        self.device_type = device_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 根据设备类型调整模型和训练参数
        self.model = SimpleModel()
        self.model.to(self.device)

        # 边缘设备资源限制
        self.batch_size = 32 if device_type == "mobile" else 64
        self.local_epochs = 1 if device_type == "mobile" else 2
        self.learning_rate = 0.01

        # 加载本地数据
        self.dataset = EdgeDataset(simulate=True)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 评估数据集（使用部分训练数据）
        eval_size = min(200, len(self.dataset) // 5)
        eval_indices = torch.randperm(len(self.dataset))[:eval_size]
        eval_dataset = torch.utils.data.Subset(self.dataset, eval_indices)
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size)

        logger.info(f"Edge client {client_id} initialized on {device_type}")
        logger.info(f"Training samples: {len(self.dataset)}, Eval samples: {eval_size}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """获取模型参数"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """本地训练"""
        logger.info(f"Client {self.client_id} starting local training...")

        # 设置服务器参数
        self.set_parameters(parameters)

        # 本地训练
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.NLLLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += len(data)

                # 边缘设备可能需要限制训练时间
                if batch_idx > 50 and self.device_type == "mobile":
                    break

            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total

            logger.info(f"Client {self.client_id} - Epoch {epoch + 1}/{self.local_epochs}: "
                        f"Loss: {epoch_loss / len(self.train_loader):.4f}, "
                        f"Accuracy: {epoch_correct / epoch_total:.4f}")

        # 返回更新后的参数和训练指标
        avg_loss = total_loss / (self.local_epochs * len(self.train_loader))
        accuracy = correct / total

        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "client_id": self.client_id,
            "device_type": self.device_type,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size
        }

        logger.info(f"Client {self.client_id} training completed. "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return self.get_parameters(config={}), len(self.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """模型评估"""
        logger.info(f"Client {self.client_id} evaluating model...")

        # 设置参数
        self.set_parameters(parameters)

        # 评估
        self.model.eval()
        criterion = nn.NLLLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)

        avg_loss = total_loss / len(self.eval_loader)
        accuracy = correct / total

        metrics = {
            "accuracy": accuracy,
            "client_id": self.client_id,
            "device_type": self.device_type,
            "eval_samples": total
        }

        logger.info(f"Client {self.client_id} evaluation completed. "
                    f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return avg_loss, total, metrics


class EdgeDeviceManager:
    """边缘设备管理器"""

    def __init__(self):
        self.clients = {}
        self.device_configs = {}

    def register_client(self, client_id: str, device_type: str,
                        server_address: str = "localhost:8080"):
        """注册边缘客户端"""
        client = EdgeClient(client_id, device_type)
        self.clients[client_id] = {
            "client": client,
            "server_address": server_address,
            "device_type": device_type
        }

        self.device_configs[client_id] = {
            "client_id": client_id,
            "device_type": device_type,
            "server_address": server_address,
            "registered_at": np.datetime64('now')
        }

        logger.info(f"Registered client {client_id} ({device_type})")

    def start_client(self, client_id: str):
        """启动指定客户端"""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not registered")

        client_info = self.clients[client_id]
        client = client_info["client"]
        server_address = client_info["server_address"]

        logger.info(f"Starting client {client_id} connecting to {server_address}")

        try:
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client
            )
        except Exception as e:
            logger.error(f"Failed to start client {client_id}: {e}")

    def start_all_clients(self):
        """启动所有注册的客户端"""
        import threading

        threads = []
        for client_id in self.clients:
            thread = threading.Thread(target=self.start_client, args=(client_id,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        return threads

    def save_config(self, filename: str = "edge_devices_config.json"):
        """保存设备配置"""
        config_data = {}
        for client_id, config in self.device_configs.items():
            config_copy = config.copy()
            config_copy['registered_at'] = str(config_copy['registered_at'])
            config_data[client_id] = config_copy

        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Device configurations saved to {filename}")


def simulate_mobile_clients():
    """模拟移动设备客户端"""
    manager = EdgeDeviceManager()

    # 注册不同类型的边缘设备
    mobile_devices = [
        ("mobile_001", "mobile"),
        ("mobile_002", "mobile"),
        ("tablet_001", "tablet"),
        ("iot_001", "iot"),
        ("laptop_001", "laptop")
    ]

    for device_id, device_type in mobile_devices:
        manager.register_client(device_id, device_type)

    # 保存配置
    manager.save_config()

    return manager


def main():
    """主函数 - 启动边缘客户端"""
    import sys

    if len(sys.argv) < 2:
        # 如果没有指定客户端ID，则模拟多个客户端
        logger.info("Starting multiple edge clients simulation...")
        manager = simulate_mobile_clients()

        # 启动所有客户端
        threads = manager.start_all_clients()

        # 等待所有线程完成
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Shutting down edge clients...")
    else:
        # 启动单个客户端
        client_id = sys.argv[1]
        device_type = sys.argv[2] if len(sys.argv) > 2 else "mobile"
        server_address = sys.argv[3] if len(sys.argv) > 3 else "localhost:8080"

        client = EdgeClient(client_id, device_type)

        logger.info(f"Starting single client {client_id} ({device_type})")
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )


if __name__ == "__main__":
    main()