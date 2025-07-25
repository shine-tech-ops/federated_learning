# main_server.py (完整修复版本)
import flwr as fl
import argparse
from typing import Optional
import inspect
from federated_model import ParameterConverter, create_evaluate_fn
from model_loader import ModelLoader


def get_all_strategies():
    """获取Flower中所有可用的策略"""
    strategies = {}

    # 获取fl.server.strategy模块中的所有类
    strategy_module = fl.server.strategy

    # 遍历模块中的所有属性
    for name in dir(strategy_module):
        attr = getattr(strategy_module, name)
        # 检查是否为类且是Strategy的子类
        if (isinstance(attr, type) and
                hasattr(attr, '__bases__') and
                name[0].isupper()):  # 类名通常以大写字母开头
            # 排除基类
            if name not in ['Strategy', 'FedOpt']:
                strategies[name.lower()] = attr

    return strategies


def create_strategy(strategy_name: str, **strategy_params):
    """
    动态创建聚合策略，支持Flower所有可用策略
    """
    # 获取所有可用策略
    strategies = get_all_strategies()

    if strategy_name.lower() not in strategies:
        available_strategies = list(strategies.keys())
        raise ValueError(f"Unsupported strategy: {strategy_name}. "
                         f"Available strategies: {available_strategies}")

    # 获取策略类
    strategy_class = strategies[strategy_name.lower()]

    # 检查策略类的构造函数参数
    sig = inspect.signature(strategy_class.__init__)
    valid_params = {}

    for param_name, param_value in strategy_params.items():
        if param_name in sig.parameters or param_name == 'kwargs':
            valid_params[param_name] = param_value

    # 创建策略实例
    return strategy_class(**valid_params)


def initialize_global_model_parameters(federated_model, initialization_method: str = "default"):
    """
    初始化全局模型参数

    Args:
        federated_model: 联邦学习模型适配器
        initialization_method: 初始化方法
            - "default": 使用模型默认参数
            - "zero": 初始化为零
            - "random": 随机初始化
    """
    if initialization_method == "default":
        # 使用模型的默认参数
        return ParameterConverter.model_to_parameters(federated_model)

    elif initialization_method == "zero":
        # 初始化为零参数
        import numpy as np
        parameters = federated_model.get_parameters()
        zero_parameters = [np.zeros_like(param) for param in parameters]
        return fl.common.ndarrays_to_parameters(zero_parameters)

    elif initialization_method == "random":
        # 随机初始化参数
        import numpy as np
        parameters = federated_model.get_parameters()
        random_parameters = [np.random.normal(0, 0.1, param.shape) for param in parameters]
        return fl.common.ndarrays_to_parameters(random_parameters)

    else:
        raise ValueError(f"Unsupported initialization method: {initialization_method}")


def start_federated_server(
        model_path: Optional[str],
        framework: str,
        data_loader_func,
        test_client_id: int = 0,
        server_address: str = "0.0.0.0:8080",
        num_rounds: int = 3,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        strategy_name: str = "fedavg",
        strategy_params: dict = None,
        parameter_initialization: str = "default",
        **kwargs
):
    """
    启动联邦学习中央服务器

    Args:
        model_path: 模型文件路径，如果为None则创建新模型
        framework: 模型框架 ('keras', 'pytorch', 'sklearn')
        data_loader_func: 数据加载函数
        test_client_id: 测试数据客户端ID
        server_address: 服务器地址
        num_rounds: 训练轮数
        min_fit_clients: 最少训练客户端数
        min_available_clients: 最少可用客户端数
        strategy_name: 聚合策略名称
        strategy_params: 聚合策略参数
        parameter_initialization: 全局参数初始化方法
    """

    # 加载或创建模型
    if model_path:
        print(f"Loading model from {model_path}...")
        federated_model = ModelLoader.load_model(model_path, framework, **kwargs)
    else:
        print("Creating new model...")
        federated_model = ModelLoader.create_model(framework, **kwargs)

    print(f"Model type: {federated_model.get_model_type()}")
    print(f"Parameter initialization method: {parameter_initialization}")

    # 加载测试数据
    print(f"Loading test data from client {test_client_id}...")
    x_test, y_test = data_loader_func(client_id=test_client_id)

    # 创建评估函数
    evaluate_fn = create_evaluate_fn(federated_model, x_test, y_test)

    # 初始化全局模型参数
    initial_parameters = initialize_global_model_parameters(federated_model, parameter_initialization)

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

    # 动态创建联邦学习策略
    strategy = create_strategy(strategy_name, **base_strategy_params)

    print(f"Starting central server, listening on {server_address}...")
    print(f"Training rounds: {num_rounds}")
    print(f"Min fit clients: {min_fit_clients}")
    print(f"Min available clients: {min_available_clients}")
    print(f"Aggregation strategy: {strategy_name}")

    # 启动中央服务器
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def list_available_strategies():
    """列出所有可用的策略"""
    strategies = get_all_strategies()
    print("Available strategies:")
    for name in sorted(strategies.keys()):
        print(f"  - {name}")
    return list(strategies.keys())


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Central Server")
    parser.add_argument("--model-path", type=str, help="Path to the initial model file")
    parser.add_argument("--framework", type=str, default="keras",
                        choices=["keras", "pytorch", "sklearn"],
                        help="Model framework")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080",
                        help="Server address")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of global training rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2,
                        help="Minimum number of clients used during fit")
    parser.add_argument("--min-available-clients", type=int, default=2,
                        help="Minimum number of total clients in the system")
    parser.add_argument("--test-client-id", type=int, default=0,
                        help="Client ID for test data")
    parser.add_argument("--strategy", type=str, default="fedavg",
                        help="Aggregation strategy (use --list-strategies to see all options)")
    parser.add_argument("--strategy-params", type=str, default="{}",
                        help="Strategy parameters in JSON format")
    parser.add_argument("--param-init", type=str, default="default",
                        choices=["default", "zero", "random"],
                        help="Global parameter initialization method")
    parser.add_argument("--list-strategies", action="store_true",
                        help="List all available strategies")

    args = parser.parse_args()

    # 如果请求列出所有策略
    if args.list_strategies:
        list_available_strategies()
        return 0

    # 解析策略参数
    import json
    try:
        strategy_params = json.loads(args.strategy_params)
    except json.JSONDecodeError:
        print("Invalid JSON format for strategy parameters")
        return 1

    # 这里需要您提供实际的数据加载函数
    # 示例：假设有一个load_data函数
    def mock_data_loader(client_id):
        # 这里应该替换为实际的数据加载逻辑
        # 例如：return load_mock_data(client_id)
        import numpy as np
        # 模拟数据
        x = np.random.random((100, 784))
        y = np.random.randint(0, 10, 100)
        return x, y

    try:
        start_federated_server(
            model_path=args.model_path,
            framework=args.framework,
            data_loader_func=mock_data_loader,  # 替换为实际的数据加载函数
            test_client_id=args.test_client_id,
            server_address=args.server_address,
            num_rounds=args.num_rounds,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.min_available_clients,
            strategy_name=args.strategy,
            strategy_params=strategy_params,
            parameter_initialization=args.param_init
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    print( fl.server.strategy.__all__)
    # main()
