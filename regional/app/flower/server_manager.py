"""
Flower 服务器管理器
"""

import threading
import time
import sys
import os
from loguru import logger
from typing import Dict, Any, Optional
import flwr as fl

# 配置 Flower 服务器专用日志
flower_logger = logger.bind(component="FlowerServer")

# 添加共享模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

from mnist_model import create_model, get_model_parameters, set_model_parameters


class FlowerServerManager:
    """Flower 服务器管理器"""
    
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.server_thread: Optional[threading.Thread] = None
        self.server_running = False
        self.current_task = None
        self.model = None
        self.server_config = None
    
    def start_server(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """启动 Flower 服务器"""
        try:
            if self.server_running:
                flower_logger.warning("⚠️  Flower 服务器已在运行")
                return self._get_server_info()
            
            # 保存任务数据
            self.current_task = task_data
            
            # 创建模型
            flower_logger.info("🧠 正在创建联邦学习模型...")
            self.model = create_model()
            flower_logger.info("✅ 模型创建完成")
            
            # 配置服务器
            self.server_config = {
                'host': 'localhost',
                'port': 8080,
                'server_id': f"federated_server_{task_data['task_id']}"
            }
            
            # Flower 服务器启动区域
            flower_logger.info("┌" + "─" * 58 + "┐")
            flower_logger.info("│" + " " * 20 + "🌺 Flower 服务器启动" + " " * 20 + "│")
            flower_logger.info("├" + "─" * 58 + "┤")
            flower_logger.info(f"│ 服务器地址: {self.server_config['host']}:{self.server_config['port']:<35} │")
            flower_logger.info(f"│ 服务器ID: {self.server_config['server_id']:<40} │")
            flower_logger.info(f"│ 训练轮数: {task_data.get('rounds', 10):<40} │")
            flower_logger.info(f"│ 聚合策略: FedAvg{'':<35} │")
            flower_logger.info("└" + "─" * 58 + "┘")
            
            # 在后台线程启动服务器
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # 等待服务器启动
            time.sleep(2)
            
            flower_logger.info("✅ Flower 服务器已启动")
            return self._get_server_info()
            
        except Exception as e:
            flower_logger.error(f"❌ 启动 Flower 服务器失败: {e}")
            raise
    
    def _run_server(self):
        """运行 Flower 服务器"""
        try:
            flower_logger.info("🔧 正在配置联邦学习策略...")
            # 创建策略
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,  # 100% 的客户端参与训练
                fraction_evaluate=1.0,  # 100% 的客户端参与评估
                min_fit_clients=1,  # 最少1个客户端
                min_evaluate_clients=1,  # 最少1个客户端
                min_available_clients=1,  # 最少1个可用客户端
                evaluate_fn=self._evaluate_fn,
                on_fit_config_fn=self._fit_config_fn,
                on_evaluate_config_fn=self._evaluate_config_fn,
            )
            flower_logger.info("✅ 策略配置完成")
            
            flower_logger.info("🚀 正在启动联邦学习服务器...")
            flower_logger.info(f"   • 监听地址: {self.server_config['host']}:{self.server_config['port']}")
            flower_logger.info(f"   • 训练轮数: {self.current_task['rounds']}")
            flower_logger.info(f"   • 聚合策略: FedAvg")
            
            # 启动服务器
            fl.server.start_server(
                server_address=f"{self.server_config['host']}:{self.server_config['port']}",
                config=fl.server.ServerConfig(num_rounds=self.current_task['rounds']),
                strategy=strategy
            )
            
            self.server_running = True
            flower_logger.info("┌" + "─" * 58 + "┐")
            flower_logger.info("│" + " " * 20 + "🎉 Flower 服务器运行完成" + " " * 20 + "│")
            flower_logger.info("└" + "─" * 58 + "┘")
            
        except Exception as e:
            flower_logger.error(f"❌ 运行联邦学习服务器服务器失败: {e}")
        finally:
            self.server_running = False
    
    def _evaluate_fn(self, server_round: int, parameters, config):
        """服务器端评估函数"""
        try:
            flower_logger.info(f"📊 开始服务器端评估，轮次: {server_round}")
            
            # 设置模型参数
            set_model_parameters(self.model, parameters)
            
            # 这里可以实现服务器端评估逻辑
            # 目前返回默认值
            loss = 0.0
            accuracy = 0.0
            metrics = {"loss": loss, "accuracy": accuracy}
            
            flower_logger.info(f"✅ 服务器端评估完成，轮次: {server_round}")
            flower_logger.info(f"   • 损失值: {loss:.4f}")
            flower_logger.info(f"   • 准确率: {accuracy:.4f}")
            
            return loss, metrics
            
        except Exception as e:
            flower_logger.error(f"❌ 服务器端评估失败: {e}")
            return 0.0, {}
    
    def _fit_config_fn(self, server_round: int):
        """训练配置函数"""
        return {
            "server_round": server_round,
            "local_epochs": 3,
            "learning_rate": 0.01
        }
    
    def _evaluate_config_fn(self, server_round: int):
        """评估配置函数"""
        return {
            "server_round": server_round
        }
    
    def stop_server(self):
        """停止 Flower 服务器"""
        try:
            if self.server_running:
                flower_logger.info("🛑 正在停止联邦学习服务器...")
                self.server_running = False
                
                # 等待服务器线程结束
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=5)
                
                flower_logger.info("✅ Flower 服务器已停止")
            else:
                flower_logger.info("ℹ️  Flower 服务器未运行")
                
        except Exception as e:
            flower_logger.error(f"❌ 停止 Flower 服务器失败: {e}")
    
    def _get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "host": self.server_config['host'],
            "port": self.server_config['port'],
            "server_id": self.server_config['server_id'],
            "running": self.server_running
        }
    
    def is_running(self) -> bool:
        """检查服务器是否运行"""
        return self.server_running
