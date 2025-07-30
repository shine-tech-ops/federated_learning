import flwr as fl
# 导入新的加载函数
from shared_logic import load_model_from_file, load_mock_data

# 将模型路径定义为配置项
MODEL_FILE_PATH = "initial_model.h5"


def main():
    # 中央服务器的策略
    # 从文件加载模型，而不是在代码中创建
    server_model = load_model_from_file(MODEL_FILE_PATH)
    x_test, y_test = load_mock_data(client_id=0)  # 假设 client_id=0 是测试集

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        initial_parameters=fl.common.ndarrays_to_parameters(server_model.get_weights()),
        evaluate_fn=get_evaluate_fn(server_model, x_test, y_test)
    )
    print("中央服务器正在启动，监听地址 [::]:8080...")
    # 启动中央服务器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # 运行3轮全局训练
        strategy=strategy,
    )


def get_evaluate_fn(server_model, x_test, y_test):
    """返回一个函数，在每轮聚合后评估全局模型质量"""
    def evaluate_fn(context, parameters, config):
        server_model.set_weights(parameters)  # 将聚合后的参数加载到模型中
        loss, accuracy = server_model.evaluate(x_test, y_test, verbose=0)
        return loss, {"accuracy": accuracy}
    return evaluate_fn


if __name__ == "__main__":
    main()
