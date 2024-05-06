# flower_regional_server.py - 区域聚合服务器实现
import flwr as fl
import numpy as np
import asyncio
import logging
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, FitRes, EvaluateRes, Parameters
from flwr.server.strategy import FedAvg
from flwr.client import NumPyClient
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionalServer:
    """区域聚合服务器"""

    def __init__(self, region_id: str, central_server_address: str):
        self.region_id = region_id
        self.central_server_address = central_server_address
        self.local_clients = []
        self.aggregated_model = None
        self.is_running = False

    def start_regional_server(self, port: int):
        """启动区域服务器"""
        strategy = RegionalFedAvg(
            region_id=self.region_id,
            central_server_address=self.central_server_address,
            fraction_fit=1.0,  # 区域内所有客户端参与
            fraction_evaluate=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

        config = fl.server.ServerConfig(num_rounds=5)  # 区域内轮次较少

        logger.info(f"Starting regional server for region {self.region_id} on port {port}")

        # 在单独线程中启动服务器
        server_thread = threading.Thread(
            target=self._run_server,
            args=(f"0.0.0.0:{port}", config, strategy)
        )
        server_thread.daemon = True
        server_thread.start()

        return server_thread

    def _run_server(self, address: str, config, strategy):
        """在单独线程中运行服务器"""
        self.is_running = True
        fl.server.start_server(
            server_address=address,
            config=config,
            strategy=strategy,
        )


class RegionalFedAvg(FedAvg):
    """区域联邦平均策略"""

    def __init__(self, region_id: str, central_server_address: str, **kwargs):
        super().__init__(**kwargs)
        self.region_id = region_id
        self.central_server_address = central_server_address
        self.regional_history = []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """区域内聚合"""
        if not results:
            return None, {}

        # 区域内聚合
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # 记录区域统计信息
        train_losses = [r.metrics.get("train_loss", 0) for _, r in results]
        train_accuracies = [r.metrics.get("train_accuracy", 0) for _, r in results]

        regional_stats = {
            "region_id": self.region_id,
            "round": server_round,
            "num_clients": len(results),
            "avg_train_loss": np.mean(train_losses),
            "avg_train_accuracy": np.mean(train_accuracies),
        }

        self.regional_history.append(regional_stats)

        logger.info(f"Region {self.region_id} - Round {server_round}: "
                    f"Clients: {len(results)}, "
                    f"Avg Loss: {np.mean(train_losses):.4f}, "
                    f"Avg Acc: {np.mean(train_accuracies):.4f}")

        # 将区域聚合结果发送给中央服务器
        self._send_to_central_server(aggregated_parameters, regional_stats)

        return aggregated_parameters, aggregated_metrics

    def _send_to_central_server(self, parameters: Parameters, stats: Dict):
        """向中央服务器发送聚合结果"""
        try:
            # 这里应该实现与中央服务器的通信
            # 可以使用gRPC、HTTP API或者直接的网络连接
            logger.info(f"Sending aggregated results from region {self.region_id} to central server")

            # 示例：保存到文件供中央服务器读取
            import pickle
            import os

            os.makedirs("regional_results", exist_ok=True)

            result_data = {
                "region_id": self.region_id,
                "parameters": parameters,
                "stats": stats,
                "timestamp": time.time()
            }

            with open(f"regional_results/region_{self.region_id}_round_{stats['round']}.pkl", "wb") as f:
                pickle.dump(result_data, f)

        except Exception as e:
            logger.error(f"Failed to send results to central server: {e}")


class RegionalClient(NumPyClient):
    """区域服务器作为中央服务器的客户端"""

    def __init__(self, region_id: str, regional_server_port: int):
        self.region_id = region_id
        self.regional_server_port = regional_server_port
        self.regional_results = None

    def get_parameters(self, config):
        """获取区域聚合后的参数"""
        if self.regional_results is None:
            # 返回随机初始化参数
            return [np.random.randn(10, 10).astype(np.float32)]
        return self.regional_results

    def fit(self, parameters, config):
        """触发区域内训练"""
        logger.info(f"Region {self.region_id} starting local training...")

        # 等待区域内训练完成
        time.sleep(2)  # 模拟训练时间

        # 加载区域聚合结果
        self._load_regional_results()

        return self.regional_results, len(self.regional_results), {
            "region_id": self.region_id,
            "train_loss": np.random.uniform(0.1, 0.5),
            "train_accuracy": np.random.uniform(0.8, 0.95)
        }

    def evaluate(self, parameters, config):
        """评估区域模型"""
        logger.info(f"Region {self.region_id} evaluating model...")

        # 模拟评估
        loss = np.random.uniform(0.1, 0.3)
        accuracy = np.random.uniform(0.85, 0.98)

        return loss, 1000, {"accuracy": accuracy, "region_id": self.region_id}

    def _load_regional_results(self):
        """加载区域聚合结果"""
        try:
            import pickle
            import glob

            # 找到最新的区域结果文件
            pattern = f"regional_results/region_{self.region_id}_round_*.pkl"
            files = glob.glob(pattern)

            if files:
                latest_file = max(files, key=os.path.getctime)
                with open(latest_file, "rb") as f:
                    result_data = pickle.load(f)
                    self.regional_results = result_data["parameters"]

        except Exception as e:
            logger.error(f"Failed to load regional results: {e}")
            self.regional_results = [np.random.randn(10, 10).astype(np.float32)]


def start_hierarchical_fl_system():
    """启动分层联邦学习系统"""
    # 区域配置
    regions = [
        {"id": "asia", "port": 8081},
        {"id": "europe", "port": 8082},
        {"id": "america", "port": 8083}
    ]

    regional_servers = []

    # 启动区域服务器
    for region_config in regions:
        regional_server = RegionalServer(
            region_id=region_config["id"],
            central_server_address="localhost:8080"
        )

        server_thread = regional_server.start_regional_server(region_config["port"])
        regional_servers.append((regional_server, server_thread))

    logger.info("All regional servers started")

    # 启动区域客户端连接到中央服务器
    regional_clients = []
    for region_config in regions:
        client = RegionalClient(
            region_id=region_config["id"],
            regional_server_port=region_config["port"]
        )
        regional_clients.append(client)

    return regional_servers, regional_clients


def main():
    """主函数"""
    # 启动分层联邦学习系统
    regional_servers, regional_clients = start_hierarchical_fl_system()

    # 让区域客户端连接到中央服务器
    # 这里需要根据实际情况调整连接逻辑

    # 保持运行
    try:
        while True:
            time.sleep(10)
            logger.info("Regional servers running...")
    except KeyboardInterrupt:
        logger.info("Shutting down regional servers...")


if __name__ == "__main__":
    main()