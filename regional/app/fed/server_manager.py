"""
Flower Server Manager
"""

import threading
import time
import sys
import os
import glob
from loguru import logger
from typing import Dict, Any, Optional
import flwr as fl

# Configure Flower server specific logger
flower_logger = logger.bind(component="FlowerServer")

# Add shared module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

from mnist_model import create_model, get_model_parameters, set_model_parameters


class FedServerManager:
    """Flower Server Manager"""
    
    def __init__(self, region_id: str, completion_callback=None):
        self.region_id = region_id
        self.server_thread: Optional[threading.Thread] = None
        self.server_running = False
        self.current_task = None
        self.model = None
        self.server_config = None
        self.completion_callback = completion_callback  # 完成回调函数
        self.final_model_path = None  # 最终模型路径
    
    def start_server(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start Flower server"""
        try:
            if self.server_running:
                flower_logger.warning("Fed Server is already running")
                return self._get_server_info()
            
            # Save task data
            self.current_task = task_data
            
            # Log received task data
            flower_logger.info(f"Received task from central server: {task_data}")
            
            # Create model
            self.model = create_model()
            
            # Configure server
            self.server_config = {
                'host': '0.0.0.0',
                'port': 8080,
                'server_id': f"federated_server_{task_data['task_id']}"
            }
            
            # Start server in background thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )

            self.server_thread.start()
            
            # Wait for server to start
            time.sleep(2)
            
            flower_logger.info("Flower server started successfully")
            return self._get_server_info()
            
        except Exception as e:
            flower_logger.error(f"Failed to start Flower server: {e}")
            raise
    
    def _run_server(self):
        """Run Flower server - 基于任务参数配置联邦学习"""
        try:
            # 从任务参数获取配置
            participation_rate = self.current_task.get('participation_rate', 100) if self.current_task else 100
            fraction_fit = participation_rate / 100.0  # 转换为 0-1 之间的比例
            
            # 获取最小客户端数量（基于参与设备数量）
            edge_devices = self.current_task.get('edge_devices', []) if self.current_task else []
            min_clients = max(1, len(edge_devices))  # 至少需要1个客户端
            
            flower_logger.info(f"配置联邦学习策略:")
            flower_logger.info(f"  - 参与率: {participation_rate}% (fraction_fit: {fraction_fit})")
            flower_logger.info(f"  - 最小客户端数: {min_clients}")
            flower_logger.info(f"  - 聚合方法: {self.current_task.get('aggregation_method', 'fedavg') if self.current_task else 'fedavg'}")
            flower_logger.info(f"  - 服务器地址: {self.server_config['host']}:{self.server_config['port']}")
            
            # Create strategy - 目前使用 FedAvg，后续可以根据 aggregation_method 选择不同策略
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=fraction_fit,  # 基于参与率配置
                fraction_evaluate=fraction_fit,  # 评估也使用相同的参与率
                min_fit_clients=min_clients,  # 最小参与训练的客户端数
                min_evaluate_clients=min_clients,  # 最小参与评估的客户端数
                min_available_clients=min_clients,  # 最小可用客户端数
                evaluate_fn=self._evaluate_fn,
                on_fit_config_fn=self._fit_config_fn,
                on_evaluate_config_fn=self._evaluate_config_fn,
            )
            
            flower_logger.info(f"Starting federated learning server for task: {self.current_task['task_id']}")
            
            # Start server
            self.server_running = True

            # 获取训练轮次
            num_rounds = self.current_task.get('rounds', 10) if self.current_task else 10
            
            flower_logger.info(f"Starting federated learning with {num_rounds} rounds")
            
            fl.server.start_server(
                server_address=f"{self.server_config['host']}:{self.server_config['port']}",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
            
            flower_logger.info(f"Federated learning completed for task: {self.current_task['task_id']}")
            
            # 联邦学习完成，查找最终模型文件
            self._handle_training_completion()
            
        except Exception as e:
            flower_logger.error(f"Failed to run federated learning server: {e}")
        finally:
            self.server_running = False
    
    def _evaluate_fn(self, server_round: int, parameters, config):
        """Server-side evaluation function with parameter storage"""
        try:
            # Set model parameters
            set_model_parameters(self.model, parameters)
            
            # Store parameters based on strategy
            self._store_parameters(server_round, parameters)
            
            # Server-side evaluation logic
            # Currently returns default values
            loss = 0.0
            accuracy = 0.0
            metrics = {"loss": loss, "accuracy": accuracy}
            
            return loss, metrics
            
        except Exception as e:
            flower_logger.error(f"Server-side evaluation failed: {e}")
            return 0.0, {}
    
    def _fit_config_fn(self, server_round: int):
        """Training configuration function - 基于任务参数配置"""
        config = {
            "server_round": server_round,
            "local_epochs": self.current_task.get('local_epochs', 3) if self.current_task else 3,
            "learning_rate": self.current_task.get('learning_rate', 0.01) if self.current_task else 0.01
        }
        return config
    
    def _evaluate_config_fn(self, server_round: int):
        """Evaluation configuration function"""
        return {
            "server_round": server_round
        }
    
    def stop_server(self):
        """Stop Flower server"""
        try:
            if self.server_running:
                flower_logger.info("Stopping federated learning server...")
                self.server_running = False
                
                # Wait for server thread to end
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=5)
                
                flower_logger.info("Flower server stopped")
            else:
                flower_logger.info("Flower server is not running")
                
        except Exception as e:
            flower_logger.error(f"Failed to stop Flower server: {e}")
    
    def _get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        # 获取实际可访问的 IP 地址（用于客户端连接）
        # 如果 host 是 0.0.0.0，需要获取实际 IP
        host = self.server_config['host']
        if host == '0.0.0.0' or host == 'localhost':
            # 获取本机实际 IP 地址（用于外部连接）
            try:
                import socket
                # 创建一个 UDP socket 来获取本机 IP
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # 不需要实际连接，只是用来获取本机 IP
                s.connect(("8.8.8.8", 80))
                actual_ip = s.getsockname()[0]
                s.close()
                host = actual_ip
                flower_logger.info(f"Resolved 0.0.0.0 to actual IP: {host}")
            except Exception as e:
                flower_logger.warning(f"Failed to get actual IP address: {e}, using {host}")
        
        return {
            "host": host,
            "port": self.server_config['port'],
            "server_id": self.server_config['server_id'],
            "running": self.server_running
        }
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.server_running
    
    def _store_parameters(self, server_round: int, parameters):
        """Store parameters based on storage strategy"""
        try:
            # Create parameters directory if not exists
            import os
            params_dir = "parameters"
            if not os.path.exists(params_dir):
                os.makedirs(params_dir)
            
            # Storage strategy: checkpoint every 5 rounds + final round
            should_save = False
            filename = ""
            
            # Check if it's a checkpoint round
            if server_round % 5 == 0:
                should_save = True
                filename = f"{params_dir}/checkpoint_round_{server_round:03d}.npz"
                flower_logger.info(f"Saving checkpoint at round {server_round}")
            
            # Check if it's the final round
            if self.current_task and server_round == self.current_task.get('rounds', 0):
                should_save = True
                filename = f"{params_dir}/final_model_round_{server_round:03d}.npz"
                self.final_model_path = filename  # 保存最终模型路径
                flower_logger.info(f"Saving final model at round {server_round}")
            
            # Save parameters if needed
            if should_save:
                self._save_parameters_to_file(parameters, filename, server_round)
                
        except Exception as e:
            flower_logger.error(f"Failed to store parameters at round {server_round}: {e}")
    
    def _save_parameters_to_file(self, parameters, filepath: str, server_round: int):
        """Save parameters to file with metadata"""
        try:
            import numpy as np
            import json
            from datetime import datetime
            
            # Save parameters
            np.savez(filepath, *parameters)
            
            # Create metadata
            import os
            metadata = {
                "server_round": server_round,
                "task_id": self.current_task.get('task_id', 'unknown') if self.current_task else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "parameter_count": len(parameters),
                "parameter_shapes": [list(p.shape) for p in parameters],
                "total_parameters": sum(p.size for p in parameters),
                "file_size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0
            }
            
            # Save metadata
            metadata_file = filepath.replace('.npz', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            flower_logger.info(f"Parameters saved: {filepath}")
            flower_logger.info(f"Metadata saved: {metadata_file}")
            
        except Exception as e:
            flower_logger.error(f"Failed to save parameters to file {filepath}: {e}")
    
    def save_current_parameters(self, custom_filename: str = None) -> Dict[str, Any]:
        """Manually save current parameters"""
        try:
            if self.model is None:
                return {"error": "No model available"}
            
            # Create parameters directory
            import os
            params_dir = "parameters"
            if not os.path.exists(params_dir):
                os.makedirs(params_dir)
            
            # Generate filename
            if custom_filename:
                filename = f"{params_dir}/{custom_filename}"
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{params_dir}/manual_save_{timestamp}.npz"
            
            # Get current parameters
            current_parameters = get_model_parameters(self.model)
            
            # Save parameters
            self._save_parameters_to_file(current_parameters, filename, 0)
            
            return {
                "success": True,
                "filepath": filename,
                "message": f"Parameters saved to {filename}"
            }
            
        except Exception as e:
            flower_logger.error(f"Failed to save current parameters: {e}")
            return {"error": str(e)}
    
    def _handle_training_completion(self):
        """处理训练完成后的操作"""
        try:
            # 如果没有找到最终模型路径，尝试查找最新的模型文件
            if not self.final_model_path:
                params_dir = "parameters"
                if os.path.exists(params_dir):
                    # 查找最新的 final_model 文件
                    final_models = glob.glob(f"{params_dir}/final_model_*.npz")
                    if final_models:
                        # 按修改时间排序，取最新的
                        final_models.sort(key=os.path.getmtime, reverse=True)
                        self.final_model_path = final_models[0]
                        flower_logger.info(f"Found final model: {self.final_model_path}")
            
            # 如果找到了最终模型，调用完成回调
            if self.final_model_path and os.path.exists(self.final_model_path):
                flower_logger.info(f"Training completed. Final model: {self.final_model_path}")
                
                if self.completion_callback:
                    try:
                        self.completion_callback(
                            task_id=self.current_task.get('task_id') if self.current_task else None,
                            model_path=self.final_model_path,
                            task_data=self.current_task
                        )
                    except Exception as e:
                        flower_logger.error(f"Error in completion callback: {e}")
            else:
                flower_logger.warning("Final model file not found after training completion")
                
        except Exception as e:
            flower_logger.error(f"Error handling training completion: {e}")
