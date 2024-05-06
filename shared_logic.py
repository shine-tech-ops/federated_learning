# file: shared_logic.py
import tensorflow as tf
import numpy as np


# 1. 定义一个简单的 Keras 模型
def create_simple_model():
    """创建一个用于二分类的简单逻辑回归模型。"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 2. 模拟加载本地数据
def load_mock_data(client_id: int):
    """为每个客户端生成一些唯一的模拟数据。"""
    # 确保每个客户端的数据略有不同
    np.random.seed(client_id)
    x_train = np.random.rand(100, 10).astype(np.float32)
    y_train = (np.random.rand(100) > 0.5).astype(np.int32)
    return x_train, y_train
