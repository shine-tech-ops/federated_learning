# flower_server.py - 中央服务器实现
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, FitRes, EvaluateRes, Parameters
from flwr.server.strategy import FedAvg
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomFedAvg(FedAvg):
    """自定义联邦平均策略"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_history = []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """聚合客户端训练结果"""
        if not results:
            return None, {}

        # 记录训练指标
        train_losses = [r.metrics.get("train_loss", 0) for _, r in results]
        train_accuracies = [r.metrics.get("train_accuracy", 0) for _, r in results]

        logger.info(f"Round {server_round}: "
                    f"Avg train loss: {np.mean(train_losses):.4f}, "
                    f"Avg train accuracy: {np.mean(train_accuracies):.4f}")

        # 使用父类的聚合方法
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # 保存轮次历史
        self.round_history.append({
            "round": server_round,
            "num_clients": len(results),
            "avg_train_loss": np.mean(train_losses),
            "avg_train_accuracy": np.mean(train_accuracies),
        })

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """聚合客户端评估结果"""
        if not results:
            return None, {}

        # 计算加权平均准确率
        accuracies = [r.metrics.get("accuracy", 0) for _, r in results]
        losses = [r.loss for _, r in results]
        num_examples = [r.num_examples for _, r in results]

        # 加权平均
        weighted_accuracy = np.average(accuracies, weights=num_examples)
        weighted_loss = np.average(losses, weights=num_examples)

        logger.info(f"Round {server_round} - Evaluation: "
                    f"Loss: {weighted_loss:.4f}, "
                    f"Accuracy: {weighted_accuracy:.4f}")

        # 更新历史记录
        if self.round_history and self.round_history[-1]["round"] == server_round:
            self.round_history[-1].update({
                "eval_loss": weighted_loss,
                "eval_accuracy": weighted_accuracy,
            })

        return weighted_loss, {"accuracy": weighted_accuracy}


def get_model_parameters():
    """获取初始模型参数"""
    # 这里使用简单的CNN模型作为示例
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.25),
        torch.nn.Flatten(),
        torch.nn.Linear(9216, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 10),
    )

    # 转换为Flower参数格式
    parameters = [val.cpu().numpy() for val in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(parameters)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """计算加权平均指标"""
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # 计算加权平均准确率
    weighted_accuracies = [
        num_examples * m["accuracy"] for num_examples, m in metrics
    ]
    weighted_accuracy = sum(weighted_accuracies) / total_examples

    return {"accuracy": weighted_accuracy}


def main():
    """启动中央服务器"""
    # 配置策略
    strategy = CustomFedAvg(
        fraction_fit=0.8,  # 每轮选择80%的客户端进行训练
        fraction_evaluate=0.5,  # 每轮选择50%的客户端进行评估
        min_fit_clients=2,  # 最少2个客户端参与训练
        min_evaluate_clients=2,  # 最少2个客户端参与评估
        min_available_clients=2,  # 最少2个客户端可用
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=get_model_parameters(),
    )

    # 服务器配置
    config = fl.server.ServerConfig(num_rounds=10)

    logger.info("Starting Flower server...")

    # 启动服务器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

    # 保存训练历史
    if hasattr(strategy, 'round_history'):
        import json
        with open('training_history.json', 'w') as f:
            json.dump(strategy.round_history, f, indent=2)
        logger.info("Training history saved to training_history.json")


if __name__ == "__main__":
    main()