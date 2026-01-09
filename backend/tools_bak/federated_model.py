# federated_model.py
from abc import ABC, abstractmethod
import flwr as fl
import numpy as np
from typing import Any, Tuple, Dict, List, Optional

class FederatedModel(ABC):
    """联邦学习模型抽象基类"""
    
    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """获取模型参数"""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """设置模型参数"""
        pass
    
    @abstractmethod
    def evaluate(self, x_test: Any, y_test: Any) -> Tuple[float, float]:
        """评估模型性能，返回(loss, accuracy)"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """获取模型类型"""
        pass

class KerasFederatedModel(FederatedModel):
    """Keras模型联邦学习适配器"""
    
    def __init__(self, model):
        self.model = model
    
    def get_parameters(self) -> List[np.ndarray]:
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.set_weights(parameters)
    
    def evaluate(self, x_test, y_test) -> Tuple[float, float]:
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, accuracy
    
    def get_model_type(self) -> str:
        return "keras"

class PyTorchFederatedModel(FederatedModel):
    """PyTorch模型联邦学习适配器"""
    
    def __init__(self, model):
        self.model = model
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        import torch
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def evaluate(self, x_test, y_test) -> Tuple[float, float]:
        import torch
        self.model.eval()
        with torch.no_grad():
            if isinstance(x_test, np.ndarray):
                x_tensor = torch.tensor(x_test, dtype=torch.float32)
                y_tensor = torch.tensor(y_test, dtype=torch.long)
            else:
                x_tensor, y_tensor = x_test, y_test
            
            outputs = self.model(x_tensor)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, y_tensor).item()
            predicted = outputs.argmax(1)
            accuracy = (predicted == y_tensor).float().mean().item()
        return loss, accuracy
    
    def get_model_type(self) -> str:
        return "pytorch"

class SklearnFederatedModel(FederatedModel):
    """Scikit-learn模型联邦学习适配器"""
    
    def __init__(self, model):
        self.model = model
    
    def get_parameters(self) -> List[np.ndarray]:
        params = []
        if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            params.append(np.array(self.model.coef_))
        if hasattr(self.model, 'intercept_') and self.model.intercept_ is not None:
            params.append(np.array(self.model.intercept_))
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        if len(parameters) > 0:
            if hasattr(self.model, 'coef_'):
                self.model.coef_ = parameters[0]
            if len(parameters) > 1 and hasattr(self.model, 'intercept_'):
                self.model.intercept_ = parameters[1]
    
    def evaluate(self, x_test, y_test) -> Tuple[float, float]:
        from sklearn.metrics import accuracy_score
        predictions = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        # 对于sklearn模型，简单返回准确率，损失设为0
        return 0.0, accuracy
    
    def get_model_type(self) -> str:
        return "sklearn"

class ModelAdapterFactory:
    """模型适配器工厂"""
    
    @staticmethod
    def create_adapter(model: Any, framework: str) -> FederatedModel:
        """创建适配器实例"""
        framework = framework.lower()
        if framework == 'keras':
            return KerasFederatedModel(model)
        elif framework == 'pytorch':
            return PyTorchFederatedModel(model)
        elif framework == 'sklearn':
            return SklearnFederatedModel(model)
        else:
            raise ValueError(f"Unsupported framework: {framework}")


class ParameterConverter:
    """参数转换工具类"""

    @staticmethod
    def model_to_parameters(federated_model: FederatedModel) -> fl.common.Parameters:
        """将模型参数转换为Flower Parameters"""
        parameters = federated_model.get_parameters()
        # 确保返回的是Parameters对象而不是列表
        if isinstance(parameters, list):
            return fl.common.ndarrays_to_parameters(parameters)
        return parameters

    @staticmethod
    def parameters_to_model(parameters: fl.common.Parameters,
                            federated_model: FederatedModel) -> None:
        """将Flower Parameters转换为模型参数"""
        # 确保parameters是正确的格式
        if hasattr(parameters, 'tensors'):
            param_arrays = fl.common.parameters_to_ndarrays(parameters)
        elif isinstance(parameters, list):
            param_arrays = parameters
        else:
            raise ValueError(f"Invalid parameters format: {type(parameters)}")
        federated_model.set_parameters(param_arrays)

def create_evaluate_fn(federated_model: FederatedModel, x_test, y_test):
    """创建评估函数"""
    def evaluate_fn(context, parameters, config):
        ParameterConverter.parameters_to_model(parameters, federated_model)
        loss, accuracy = federated_model.evaluate(x_test, y_test)
        return loss, {"accuracy": accuracy}
    return evaluate_fn
