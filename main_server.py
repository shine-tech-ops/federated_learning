# file: main_server.py
import flwr as fl
from shared_logic import create_simple_model


def main():
    # 中央服务器的策略
    strategy = fl.server.strategy.FedAvg(
        # 中央服务器只与区域服务器（的桥接客户端）通信
        min_fit_clients=1,
        min_available_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(create_simple_model().get_weights())
    )

    print("中央服务器正在启动，监听地址 [::]:8080...")
    # 启动中央服务器
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # 运行3轮全局训练
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
