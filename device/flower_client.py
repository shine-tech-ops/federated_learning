"""
Flower 客户端
"""
import time
import os
import flwr as fl
from loguru import logger
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import numpy as np

from mnist_trainer import MNISTTrainer


class FlowerClient(fl.client.NumPyClient):
    """Flower 客户端"""
    
    def __init__(
        self, 
        device_id: str, 
        trainer: MNISTTrainer, 
        server_address: str,
        task_id: Optional[str] = None,
        region_id: Optional[Union[int, str]] = None,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.device_id = device_id
        self.trainer = trainer
        self.server_address = server_address
        self.client = None
        self.running = False
        self.task_id = task_id
        self.region_id = region_id
        self.log_callback = log_callback
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """获取模型参数"""
        logger.info(f"设备 {self.device_id} 获取模型参数")
        return self.trainer.get_parameters()
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """训练模型"""
        logger.info(f"设备 {self.device_id} 开始训练")
        start_ts = time.time()
        round_num = config.get("server_round")
        
        try:
            # 执行训练
            updated_parameters, num_examples, metrics = self.trainer.fit(parameters, config)

            duration = time.time() - start_ts

            # 训练完成后保存模型
            self._save_model_after_training(metrics)
            
            # 记录训练结果
            logger.info(f"设备 {self.device_id} 训练完成: {metrics}")

            # 上报训练日志
            self._emit_log({
                "phase": "train",
                "level": "INFO",
                "round": round_num,
                "num_examples": num_examples,
                "metrics": metrics,
                "loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
                "duration": duration,
                "message": f"local training finished, round={round_num}, examples={num_examples}"
            })
            
            return updated_parameters, num_examples, metrics
        except Exception as e:
            # 上报错误日志
            self._emit_log({
                "phase": "train",
                "level": "ERROR",
                "round": round_num,
                "error_message": str(e),
                "message": f"local training failed in round {round_num}"
            })
            raise
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """评估模型"""
        logger.info(f"设备 {self.device_id} 开始评估")
        start_ts = time.time()
        round_num = config.get("server_round")
        
        try:
            # 执行评估
            loss, num_examples, metrics = self.trainer.evaluate(parameters, config)
            duration = time.time() - start_ts
            
            # 记录评估结果
            logger.info(f"设备 {self.device_id} 评估完成: {metrics}")

            # 上报评估日志
            self._emit_log({
                "phase": "evaluate",
                "level": "INFO",
                "round": round_num,
                "num_examples": num_examples,
                "metrics": metrics,
                "loss": loss,
                "accuracy": metrics.get("accuracy"),
                "duration": duration,
                "message": f"local evaluation finished, round={round_num}, examples={num_examples}"
            })
            
            return loss, num_examples, metrics
        except Exception as e:
            self._emit_log({
                "phase": "evaluate",
                "level": "ERROR",
                "round": round_num,
                "error_message": str(e),
                "message": f"local evaluation failed in round {round_num}"
            })
            raise

    def _save_model_after_training(self, metrics: Dict[str, Any]):
        """训练后保存模型"""
        try:
            # 创建模型保存目录
            model_dir = f"models/{self.device_id}"
            os.makedirs(model_dir, exist_ok=True)

            # 生成文件名（包含时间戳和准确率）
            timestamp = int(time.time())
            accuracy = metrics.get('accuracy', 0)
            filename = f"model_{timestamp}_acc_{accuracy:.4f}.pth"
            file_path = os.path.join(model_dir, filename)

            # 保存模型
            self.trainer.save_model(file_path)

            # 同时保存一份参数文件
            param_filename = f"parameters_{timestamp}_acc_{accuracy:.4f}.pkl"
            param_path = os.path.join(model_dir, param_filename)
            self.trainer.save_model_parameters(param_path)

        except Exception as e:
            logger.error(f"训练后保存模型失败: {e}")
    
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

    def _emit_log(self, payload: Dict[str, Any]):
        """将本地训练/评估日志发送到上游回调"""
        if not self.log_callback:
            return

        log_data = {
            "task": self.task_id,
            "region_node": self.region_id,
            "device_id": self.device_id,
            "phase": payload.get("phase", "system"),
            "level": payload.get("level", "INFO"),
            "round": payload.get("round"),
            "loss": payload.get("loss"),
            "accuracy": payload.get("accuracy"),
            "num_examples": payload.get("num_examples"),
            "metrics": payload.get("metrics"),
            "duration": payload.get("duration"),
            "message": payload.get("message"),
            "error_message": payload.get("error_message"),
        }

        try:
            self.log_callback(log_data)
        except Exception as e:
            logger.warning(f"训练日志回调发送失败: {e}")
