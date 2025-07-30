# main_server.py (修复版本)
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
        parameters = federated_model.get_parameters()
        # 确保返回正确的Parameters对象
        if isinstance(parameters, list):
            return fl.common.ndarrays_to_parameters(parameters)
        return parameters

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


def validate_parameters(parameters):
    """
    验证参数格式是否正确
    """
    # 检查是否为Parameters对象
    if not hasattr(parameters, 'tensors') and not isinstance(parameters, fl.common.Parameters):
        # 如果是列表，转换为Parameters对象
        if isinstance(parameters, list):
            return fl.common.ndarrays_to_parameters(parameters)
        else:
            raise ValueError(f"Invalid parameter format: {type(parameters)}")
    return parameters


def validate_data_shapes(federated_model, x_test, y_test):
    """
    验证测试数据形状是否与模型兼容
    """
    try:
        # 获取模型信息
        model_type = federated_model.get_model_type()
        print(f"Model type: {model_type}")
        print(f"Test data shape: X={x_test.shape}, y={y_test.shape if y_test is not None else 'None'}")

        # 对于Keras模型，可以尝试获取输入形状信息
        if model_type == "keras":
            try:
                # 打印模型摘要
                print("Model Summary:")
                federated_model.model.summary()
                print(f"Model expected input shape: {federated_model.model.input_shape}")
                print(f"Model output shape: {federated_model.model.output_shape}")
            except Exception as e:
                print(f"Could not print model summary: {e}")

        return True
    except Exception as e:
        print(f"Warning: Could not validate data shapes: {e}")
        return True  # 继续执行，让模型自己报错


def create_compatible_data_loader(expected_input_shape, num_classes=10):
    """
    创建与模型输入形状兼容的数据加载函数
    """

    def compatible_data_loader(client_id):
        import numpy as np
        # 根据模型输入形状创建兼容的数据
        batch_size = 100
        # 移除batch维度(None)并创建对应形状的数据
        input_shape = expected_input_shape[1:] if expected_input_shape[0] is None else expected_input_shape
        x = np.random.random((batch_size,) + input_shape)
        y = np.random.randint(0, num_classes, batch_size)
        return x, y

    return compatible_data_loader


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

    # 验证数据形状
    print("Validating data shapes...")
    validate_data_shapes(federated_model, x_test, y_test)

    # 创建评估函数
    evaluate_fn = create_evaluate_fn(federated_model, x_test, y_test)

    # 初始化全局模型参数
    initial_parameters = initialize_global_model_parameters(federated_model, parameter_initialization)

    # 验证参数格式
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
    parser.add_argument("--input-shape", type=str, default="(784,)",
                        help="Input shape for data generation (e.g., '(784,)' for MNIST)")

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

    # 解析输入形状
    try:
        input_shape = eval(args.input_shape)  # 注意：在生产环境中应使用更安全的解析方法
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
        # 添加batch维度
        if len(input_shape) == 1 or input_shape[0] is not None:
            full_input_shape = (None,) + input_shape
        else:
            full_input_shape = input_shape
    except:
        print(f"Invalid input shape format: {args.input_shape}, using default (784,)")
        full_input_shape = (None, 784)

    # 创建与模型兼容的数据加载函数
    def compatible_data_loader(client_id):
        import numpy as np
        # 创建与模型输入形状兼容的数据
        batch_size = 100
        # 移除batch维度并创建对应形状的数据
        data_shape = full_input_shape[1:] if full_input_shape[0] is None else full_input_shape
        x = np.random.random((batch_size,) + data_shape)
        y = np.random.randint(0, 10, batch_size)  # 默认10分类
        return x, y

    try:
        start_federated_server(
            model_path=args.model_path,
            framework=args.framework,
            data_loader_func=compatible_data_loader,  # 使用兼容的数据加载函数
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
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
