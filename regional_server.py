# file: regional_server.py
import flwr as fl
import threading
import argparse
from shared_logic import create_simple_model


# 这个客户端将代表整个区域连接到中央服务器
class RegionalBridgeClient(fl.client.NumPyClient):
    def __init__(self, region_name, regional_server_address):
        self.region_name = region_name
        self.regional_server_address = regional_server_address
        self.model = create_simple_model()

    def fit(self, parameters, config):
        print(f"[{self.region_name}] 从中央服务器接收到 FIT 指令，正在启动区域内训练...")

        # 启动一个临时的 Flower 轮次，只涉及本区域的客户端
        # 这完美模拟了「下发.html」中的模型下发流程
        strategy = fl.server.strategy.FedAvg(
            initial_parameters=fl.common.ndarrays_to_parameters(parameters),
            min_fit_clients=1,
            min_available_clients=1,
        )

        # 【修复】使用 fl.server.start_server 来启动一个临时的区域服务器
        # 它会监听指定地址，运行一轮，然后返回历史记录
        history = fl.server.start_server(
            server_address=self.regional_server_address,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=1),
        )

        # 从区域训练结果中提取聚合后的权重
        # 这完美模拟了「聚合.html」中的模型聚合流程
        aggregated_params = fl.common.parameters_to_ndarrays(history.parameters_centralized)
        num_examples = history.metrics_distributed_fit['fit_total'][0][1]

        print(f"[{self.region_name}] 区域内训练和聚合完成，正在将结果返回给中央服务器。")
        return aggregated_params, num_examples, {}


def run_regional_server(region_name, address):
    """为本区域的客户端运行一个永久的 Flower 服务器。"""
    print(f"[{region_name}] 区域服务器正在启动，监听地址 {address}...")
    # 这个服务器会一直运行，等待其下的客户端连接
    # 注意：我们在这里不启动它，因为上面的 start_server 会临时接管
    # 在真实生产环境中，这里会是一个独立的、长期运行的服务器进程
    pass  # 仅为逻辑占位


def main():
    parser = argparse.ArgumentParser(description="Regional Server")
    parser.add_argument("--region-name", type=str, required=True, help="区域的名称")
    parser.add_argument("--listen-address", type=str, default="[::]:8081", help="监听客户端连接的地址")
    parser.add_argument("--central-server-address", type=str, default="[::]:8080", help="中央服务器的地址")
    args = parser.parse_args()

    # 1. 启动一个线程来运行区域服务器（逻辑占位）
    # server_thread = threading.Thread(target=run_regional_server, args=(args.region_name, args.listen_address))
    # server_thread.start()

    # 2. 启动代表本区域的客户端，连接到中央服务器
    print(f"[{args.region_name}] 正在启动桥接客户端，连接到中央服务器 {args.central_server_address}...")
    fl.client.start_numpy_client(
        server_address=args.central_server_address,
        client=RegionalBridgeClient(
            region_name=args.region_name,
            regional_server_address=args.listen_address
        )
    )


if __name__ == "__main__":
    main()