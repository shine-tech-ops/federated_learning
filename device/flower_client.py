"""
Flower 客户端
"""

import flwr as fl
from loguru import logger
from typing import Dict, Any, List, Tuple
import numpy as np

from mnist_trainer import MNISTTrainer


class FlowerClient(fl.client.NumPyClient):
    """Flower 客户端"""
    
    def __init__(self, device_id: str, trainer: MNISTTrainer, server_address: str):
        self.device_id = device_id
        self.trainer = trainer
        self.server_address = server_address
        self.client = None
        self.running = False
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """获取模型参数"""
        logger.info(f"设备 {self.device_id} 获取模型参数")
        return self.trainer.get_parameters()
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """训练模型"""
        logger.info(f"设备 {self.device_id} 开始训练")
        
        # 执行训练
        updated_parameters, num_examples, metrics = self.trainer.fit(parameters, config)
        
        # 记录训练结果
        logger.info(f"设备 {self.device_id} 训练完成: {metrics}")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """评估模型"""
        logger.info(f"设备 {self.device_id} 开始评估")
        
        # 执行评估
        loss, num_examples, metrics = self.trainer.evaluate(parameters, config)
        
        # 记录评估结果
        logger.info(f"设备 {self.device_id} 评估完成: {metrics}")
        
        return loss, num_examples, metrics
    
    def start(self):
        """启动 Flower 客户端"""
        try:
            logger.info(f"设备 {self.device_id} 连接到 Flower 服务器: {self.server_address}")
            
            # 启动 Flower 客户端
            fl.client.start_numpy_client(
                server_address=self.server_address,
                client=self
            )
            
            self.running = True
            logger.info(f"设备 {self.device_id} Flower 客户端已启动")
            
        except Exception as e:
            logger.error(f"启动 Flower 客户端失败: {e}")
            raise
    
    def pause(self):
        """暂停客户端"""
        logger.info(f"设备 {self.device_id} 暂停")
        # 这里可以实现暂停逻辑
        pass
    
    def resume(self):
        """恢复客户端"""
        logger.info(f"设备 {self.device_id} 恢复")
        # 这里可以实现恢复逻辑
        pass
    
    def stop(self):
        """停止客户端"""
        logger.info(f"设备 {self.device_id} 停止")
        self.running = False
        # 这里可以实现停止逻辑
        pass
