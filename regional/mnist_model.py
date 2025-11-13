"""
共享的 MNIST 模型定义
用于区域节点和设备端
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import requests


class SimpleCNN(nn.Module):
    """简单的 CNN 模型用于 MNIST 分类"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_model(num_classes=10):
    """创建模型实例"""
    return SimpleCNN(num_classes)


def get_model_parameters(model):
    """获取模型参数"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    """设置模型参数"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def download_model(url, save_path='model.pth'):
    """从 URL 下载模型"""
    if os.path.exists(save_path):
        print(f"模型已存在: {save_path}")
        return save_path
    
    print(f"正在下载模型: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"模型已保存: {save_path}")
    return save_path


def load_model(model_path):
    """加载模型"""
    model = create_model()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 兼容不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def predict_image(model, image_path):
    """预测图片中的数字"""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted = output.argmax(dim=1).item()
    
    return predicted


if __name__ == "__main__":
    import sys
    
    # 使用示例
    model_url = ""
    image_path = '/Users/vincent/code/federated_learning/device/data/MNIST/images/train/train_00002_label_4.png'
    
    if model_url:
        model_path = download_model(model_url)
        model = load_model(model_path)
    else:
        model = create_model()
    
    result = predict_image(model, image_path)
    print(result)