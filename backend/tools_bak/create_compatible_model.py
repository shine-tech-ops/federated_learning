# create_compatible_model.py
import tensorflow as tf
import numpy as np


def create_and_save_model():
    """创建与数据兼容的模型"""
    # 创建一个兼容784输入维度的模型（如MNIST数据集）
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10个类别
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 创建一些虚拟数据来测试模型
    x_train = np.random.random((1000, 784))
    y_train = np.random.randint(0, 10, 1000)

    # 简单训练几步来初始化模型
    model.fit(x_train[:100], y_train[:100], epochs=1, verbose=0)

    # 保存模型
    model.save('initial_model.h5')
    print("Model saved as 'initial_model.h5'")

    # 打印模型信息
    print("\nModel Summary:")
    model.summary()

    return model


if __name__ == "__main__":
    create_and_save_model()