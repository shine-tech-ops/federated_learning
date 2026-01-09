# federated_server.py
import flwr as fl
import argparse
from typing import Optional
from federated_model import ParameterConverter, create_evaluate_fn
from model_loader import ModelLoader


def initialize_global_model_parameters(federated_model, initialization_method: str = "default"):
    """
    初始化全局模型参数
    """
    if initialization_method == "default":
        parameters = federated_model.get_parameters()
        if isinstance(parameters, list):
            return fl.common.ndarrays_to_parameters(parameters)
        return parameters

    elif initialization_method == "zero":
        import numpy as np
        parameters = federated_model.get_parameters()
        zero_parameters = [np.zeros_like(param) for param in parameters]
        return fl.common.ndarrays_to_parameters(zero_parameters)

    elif initialization_method == "random":
        import numpy as np
        parameters = federated_model.get_parameters()
        random_parameters = [np.random.normal(0, 0.1, param.shape) for param in parameters]
        return fl.common.ndarrays_to_parameters(random_parameters)

    else:
        raise ValueError(f"Unsupported initialization method: {initialization_method}")


def validate_parameters(parameters):
    """
    验证参数格式是否正确
    """
    if not hasattr(parameters, 'tensors') and not isinstance(parameters, fl.common.Parameters):
        if isinstance(parameters, list):
            return fl.common.ndarrays_to_parameters(parameters)
        else:
            raise ValueError(f"Invalid parameter format: {type(parameters)}")
    return parameters


def create_federated_strategy(
        model_path: str,
        framework: str,
        data_loader_func,
        test_client_id: int = 0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        parameter_initialization: str = "default",
        strategy_name: str = "fedavg",
        strategy_params: dict = None,
        **kwargs
):
    """
    创建联邦学习策略
    """
    # 加载模型
    print(f"Loading model from {model_path}...")
    federated_model = ModelLoader.load_model(model_path, framework, **kwargs)

    print(f"Model type: {federated_model.get_model_type()}")
    print(f"Parameter initialization method: {parameter_initialization}")

    # 加载测试数据
    print(f"Loading test data from client {test_client_id}...")
    x_test, y_test = data_loader_func(client_id=test_client_id)

    # 创建评估函数
    evaluate_fn = create_evaluate_fn(federated_model, x_test, y_test)

    # 初始化全局模型参数
    initial_parameters = initialize_global_model_parameters(federated_model, parameter_initialization)
    initial_parameters = validate_parameters(initial_parameters)

    # 准备策略参数
    base_strategy_params = {
        'min_fit_clients': min_fit_clients,
        'min_available_clients': min_available_clients,
        'initial_parameters': initial_parameters,
        'evaluate_fn': evaluate_fn
    }

    # 合并自定义策略参数
    if strategy_params:
        base_strategy_params.update(strategy_params)

    # 创建策略（这里仍然使用旧方式创建策略对象，但不启动服务器）
    from main import create_strategy
    strategy = create_strategy(strategy_name, **base_strategy_params)

    return strategy


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Strategy Creator")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the initial model file")
    parser.add_argument("--framework", type=str, default="keras",
                        choices=["keras", "pytorch", "sklearn"],
                        help="Model framework")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of global training rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2,
                        help="Minimum number of clients used during fit")
    parser.add_argument("--min-available-clients", type=int, default=2,
                        help="Minimum number of total clients in the system")
    parser.add_argument("--test-client-id", type=int, default=0,
                        help="Client ID for test data")
    parser.add_argument("--strategy", type=str, default="fedavg",
                        help="Aggregation strategy")
    parser.add_argument("--strategy-params", type=str, default="{}",
                        help="Strategy parameters in JSON format")
    parser.add_argument("--param-init", type=str, default="default",
                        choices=["default", "zero", "random"],
                        help="Global parameter initialization method")

    args = parser.parse_args()

    # 解析策略参数
    import json
    try:
        strategy_params = json.loads(args.strategy_params)
    except json.JSONDecodeError:
        print("Invalid JSON format for strategy parameters")
        return 1

    # 数据加载函数
    def compatible_data_loader(client_id):
        import numpy as np
        batch_size = 100
        x = np.random.random((batch_size, 784))  # 默认MNIST形状
        y = np.random.randint(0, 10, batch_size)
        return x, y

    try:
        strategy = create_federated_strategy(
            model_path=args.model_path,
            framework=args.framework,
            data_loader_func=compatible_data_loader,
            test_client_id=args.test_client_id,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            parameter_initialization=args.param_init,
            strategy_name=args.strategy,
            strategy_params=strategy_params
        )

        print(f"Strategy created successfully: {args.strategy}")
        print("Now you can start the Flower SuperLink with:")
        print("  flower-superlink --insecure")
        print("And start the Flower Server with:")
        print("  flower-server --insecure")

        # 保存策略配置供后续使用
        import pickle
        strategy_config = {
            'model_path': args.model_path,
            'framework': args.framework,
            'strategy_name': args.strategy,
            'strategy_params': strategy_params,
            'min_fit_clients': args.min_fit_clients,
            'min_available_clients': args.min_available_clients,
            'param_init': args.param_init
        }

        with open('strategy_config.pkl', 'wb') as f:
            pickle.dump(strategy_config, f)
        print("Strategy configuration saved to strategy_config.pkl")

    except Exception as e:
        print(f"Error creating strategy: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
