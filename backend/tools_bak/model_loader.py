# model_loader.py (增强版本)
import os
from typing import Any, Optional
from federated_model import ModelAdapterFactory, FederatedModel


class ModelLoader:
    """统一模型加载器"""

    @staticmethod
    def load_model(model_path: str, framework: str, **kwargs) -> FederatedModel:
        """根据框架类型加载模型"""
        framework = framework.lower()

        if framework == 'keras':
            import tensorflow as tf
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded Keras model with input shape: {model.input_shape}")
            print(f"Loaded Keras model with output shape: {model.output_shape}")
            return ModelAdapterFactory.create_adapter(model, framework)

        elif framework == 'pytorch':
            import torch
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model_class = kwargs.get('model_class')
            if model_class is None:
                raise ValueError("model_class must be provided for PyTorch models")
            model = model_class()
            model.load_state_dict(torch.load(model_path))
            return ModelAdapterFactory.create_adapter(model, framework)

        elif framework == 'sklearn':
            import pickle
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return ModelAdapterFactory.create_adapter(model, framework)

        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def create_model(framework: str, **kwargs) -> FederatedModel:
        """创建新模型"""
        framework = framework.lower()

        if framework == 'keras':
            import tensorflow as tf
            model_config = kwargs.get('model_config', {})

            # 默认模型配置
            default_config = {
                'input_shape': (784,),  # 默认为MNIST数据集形状
                'hidden_units': [128],
                'num_classes': 10,
                'dropout_rate': 0.2
            }
            default_config.update(model_config)

            print(f"Creating Keras model with config: {default_config}")

            model = tf.keras.Sequential()
            # 输入层
            model.add(tf.keras.layers.Dense(
                default_config['hidden_units'][0],
                activation='relu',
                input_shape=default_config['input_shape']
            ))

            # 隐藏层
            for units in default_config['hidden_units'][1:]:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
                if default_config['dropout_rate'] > 0:
                    model.add(tf.keras.layers.Dropout(default_config['dropout_rate']))

            # 输出层
            model.add(tf.keras.layers.Dense(default_config['num_classes'], activation='softmax'))

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            print(f"Created Keras model with input shape: {model.input_shape}")
            print(f"Created Keras model with output shape: {model.output_shape}")
            return ModelAdapterFactory.create_adapter(model, framework)

        elif framework == 'pytorch':
            import torch.nn as nn
            model_class = kwargs.get('model_class')
            if model_class is None:
                raise ValueError("model_class must be provided for PyTorch models")
            model = model_class()
            return ModelAdapterFactory.create_adapter(model, framework)

        elif framework == 'sklearn':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            model_type = kwargs.get('model_type', 'logistic_regression')
            model_params = kwargs.get('model_params', {})

            if model_type == 'logistic_regression':
                model = LogisticRegression(**model_params)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(**model_params)
            else:
                raise ValueError(f"Unsupported sklearn model type: {model_type}")

            return ModelAdapterFactory.create_adapter(model, framework)

        else:
            raise ValueError(f"Model creation not implemented for framework: {framework}")
