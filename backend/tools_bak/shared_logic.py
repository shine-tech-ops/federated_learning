import tensorflow as tf
import numpy as np


def load_model_from_file(model_path: str) -> tf.keras.Model:
    """
    从指定的 H5 文件加载 Keras 模型。
    这是一个更健壮和可维护的方式来管理模型。
    """
    print(f"正在从文件加载模型: {model_path}")
    try:
        # tf.keras.models.load_model 可以完整地重建模型
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise e


# 2. 模拟加载本地数据 (此函数保持不变)
def load_mock_data(client_id: int):
    """为每个客户端生成一些唯一的模拟数据。"""
    # 确保每个客户端的数据略有不同
    np.random.seed(client_id)
    x_train = np.random.rand(100, 10).astype(np.float32)
    y_train = (np.random.rand(100) > 0.5).astype(np.int32)
    return x_train, y_train


# 您可以保留或删除这个旧函数
def create_simple_model():
    """创建一个用于二分类的简单逻辑回归模型。"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
