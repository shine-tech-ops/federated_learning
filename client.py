# file: client.py
import flwr as fl
import requests
import argparse
from shared_logic import create_simple_model, load_mock_data


# 定义 Flower 客户端
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = create_simple_model()
        self.x_train, self.y_train = load_mock_data(self.client_id)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] 正在进行本地训练...")
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=int, required=True, help="客户端的唯一ID")
    parser.add_argument("--region", type=str, required=True, help="客户端所属的区域")
    parser.add_argument("--regional-server-address", type=str, default="[::]:8081", help="区域服务器的地址")
    args = parser.parse_args()

    # 1. 注册到注册服务
    registration_url = "http://localhost:5000/register"
    try:
        response = requests.post(registration_url, json={
            "client_id": args.client_id,
            "region": args.region,
            "address": f"client_{args.client_id}_placeholder_address"  # 实际地址信息
        })
        if response.status_code == 200:
            print(f"[Client {args.client_id}] 注册成功！")
        else:
            print(f"[Client {args.client_id}] 注册失败: {response.text}")
            return
    except requests.exceptions.ConnectionError as e:
        print(f"[Client {args.client_id}] 无法连接到注册服务: {e}")
        return

    # 2. 启动 Flower 客户端，连接到其区域服务器
    print(f"[Client {args.client_id}] 正在启动，并连接到区域服务器 {args.regional_server_address}...")
    fl.client.start_numpy_client(
        server_address=args.regional_server_address,
        client=FlowerClient(client_id=args.client_id)
    )


if __name__ == "__main__":
    main()
