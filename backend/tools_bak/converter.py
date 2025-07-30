import os
import torch
import torch.onnx
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
from typing import Union, Tuple, Optional
import tempfile
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXConverter:
    """
    使用 ONNX 实现 PyTorch 和 TensorFlow 模型之间的相互转换
    """

    def __init__(self, opset_version: int = 13):
        """
        初始化转换器

        Args:
            opset_version: ONNX opset 版本
        """
        self.opset_version = opset_version
        self.temp_dir = tempfile.mkdtemp()

    def pytorch_to_onnx(self,
                        model: torch.nn.Module,
                        dummy_input: Union[torch.Tensor, Tuple],
                        onnx_path: str,
                        input_names: Optional[list] = None,
                        output_names: Optional[list] = None,
                        dynamic_axes: Optional[dict] = None) -> bool:
        """
        将 PyTorch 模型转换为 ONNX 格式

        Args:
            model: PyTorch 模型
            dummy_input: 示例输入数据
            onnx_path: 输出 ONNX 文件路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴定义

        Returns:
            bool: 转换是否成功
        """
        try:
            model.eval()

            # 默认参数
            if input_names is None:
                input_names = ['input']
            if output_names is None:
                output_names = ['output']

            # 导出 ONNX 模型
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

            # 验证模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            logger.info(f"PyTorch 模型成功转换为 ONNX: {onnx_path}")
            return True

        except Exception as e:
            logger.error(f"PyTorch 转 ONNX 失败: {e}")
            return False

    def tensorflow_to_onnx(self,
                           tf_model_path: str,
                           onnx_path: str,
                           opset: Optional[int] = None) -> bool:
        """
        将 TensorFlow 模型转换为 ONNX 格式

        Args:
            tf_model_path: TensorFlow 模型路径（SavedModel 格式）
            onnx_path: 输出 ONNX 文件路径
            opset: ONNX opset 版本

        Returns:
            bool: 转换是否成功
        """
        try:
            # 需要安装 tf2onnx
            import tf2onnx

            if opset is None:
                opset = self.opset_version

            # 转换命令
            import subprocess
            import sys

            cmd = [
                sys.executable, '-m', 'tf2onnx.convert',
                '--saved-model', tf_model_path,
                '--output', onnx_path,
                '--opset', str(opset)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # 验证模型
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                logger.info(f"TensorFlow 模型成功转换为 ONNX: {onnx_path}")
                return True
            else:
                logger.error(f"TensorFlow 转 ONNX 失败: {result.stderr}")
                return False

        except ImportError:
            logger.error("请安装 tf2onnx: pip install tf2onnx")
            return False
        except Exception as e:
            logger.error(f"TensorFlow 转 ONNX 失败: {e}")
            return False

    def onnx_to_pytorch(self, onnx_path: str, custom_class=None) -> Optional[torch.nn.Module]:
        """
        将 ONNX 模型转换为 PyTorch 模型

        Args:
            onnx_path: ONNX 模型路径
            custom_class: 自定义 PyTorch 模型类（可选）

        Returns:
            torch.nn.Module: 转换后的 PyTorch 模型，失败时返回 None
        """
        try:
            # 需要安装 onnx2pytorch
            from onnx2pytorch import ConvertModel

            # 加载并验证 ONNX 模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 转换为 PyTorch 模型
            pytorch_model = ConvertModel(onnx_model)
            pytorch_model.eval()

            logger.info("ONNX 成功转换为 PyTorch 模型")
            return pytorch_model

        except ImportError:
            logger.error("请安装 onnx2pytorch: pip install onnx2pytorch")
            return None
        except Exception as e:
            logger.error(f"ONNX 转 PyTorch 失败: {e}")
            return None

    def onnx_to_tensorflow(self, onnx_path: str, tf_model_path: str) -> bool:
        """
        将 ONNX 模型转换为 TensorFlow 模型

        Args:
            onnx_path: ONNX 模型路径
            tf_model_path: 输出 TensorFlow 模型路径

        Returns:
            bool: 转换是否成功
        """
        try:
            # 需要安装 onnx_tf
            from onnx_tf.backend import prepare

            # 加载并验证 ONNX 模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 转换为 TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(tf_model_path)

            logger.info(f"ONNX 成功转换为 TensorFlow 模型: {tf_model_path}")
            return True

        except ImportError:
            logger.error("请安装 onnx-tf: pip install onnx-tf")
            return False
        except Exception as e:
            logger.error(f"ONNX 转 TensorFlow 失败: {e}")
            return False

    def pytorch_to_tensorflow(self,
                              pytorch_model: torch.nn.Module,
                              dummy_input: Union[torch.Tensor, Tuple],
                              tf_model_path: str,
                              input_names: Optional[list] = None,
                              output_names: Optional[list] = None,
                              dynamic_axes: Optional[dict] = None) -> bool:
        """
        直接将 PyTorch 模型转换为 TensorFlow 模型

        Args:
            pytorch_model: PyTorch 模型
            dummy_input: 示例输入数据
            tf_model_path: 输出 TensorFlow 模型路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴定义

        Returns:
            bool: 转换是否成功
        """
        try:
            # 创建临时 ONNX 文件
            temp_onnx_path = os.path.join(self.temp_dir, "temp_model.onnx")

            # PyTorch -> ONNX
            if not self.pytorch_to_onnx(
                    pytorch_model, dummy_input, temp_onnx_path,
                    input_names, output_names, dynamic_axes
            ):
                return False

            # ONNX -> TensorFlow
            success = self.onnx_to_tensorflow(temp_onnx_path, tf_model_path)

            # 清理临时文件
            if os.path.exists(temp_onnx_path):
                os.remove(temp_onnx_path)

            return success

        except Exception as e:
            logger.error(f"PyTorch 转 TensorFlow 失败: {e}")
            return False

    def tensorflow_to_pytorch(self,
                              tf_model_path: str,
                              pytorch_model_class=None) -> Optional[torch.nn.Module]:
        """
        直接将 TensorFlow 模型转换为 PyTorch 模型

        Args:
            tf_model_path: TensorFlow 模型路径
            pytorch_model_class: 自定义 PyTorch 模型类（可选）

        Returns:
            torch.nn.Module: 转换后的 PyTorch 模型，失败时返回 None
        """
        try:
            # 创建临时 ONNX 文件
            temp_onnx_path = os.path.join(self.temp_dir, "temp_model.onnx")

            # TensorFlow -> ONNX
            if not self.tensorflow_to_onnx(tf_model_path, temp_onnx_path):
                return None

            # ONNX -> PyTorch
            pytorch_model = self.onnx_to_pytorch(temp_onnx_path, pytorch_model_class)

            # 清理临时文件
            if os.path.exists(temp_onnx_path):
                os.remove(temp_onnx_path)

            return pytorch_model

        except Exception as e:
            logger.error(f"TensorFlow 转 PyTorch 失败: {e}")
            return None

    def verify_conversion(self,
                          original_model,
                          converted_model,
                          test_input: Union[np.ndarray, torch.Tensor],
                          framework: str = "pytorch") -> dict:
        """
        验证转换结果的正确性

        Args:
            original_model: 原始模型
            converted_model: 转换后的模型
            test_input: 测试输入数据
            framework: 原始模型框架 ("pytorch" 或 "tensorflow")

        Returns:
            dict: 包含验证结果的字典
        """
        try:
            if framework.lower() == "pytorch":
                return self._verify_pytorch_conversion(original_model, converted_model, test_input)
            elif framework.lower() == "tensorflow":
                return self._verify_tensorflow_conversion(original_model, converted_model, test_input)
            else:
                raise ValueError("不支持的框架类型")

        except Exception as e:
            logger.error(f"验证转换失败: {e}")
            return {"success": False, "error": str(e)}

    def _verify_pytorch_conversion(self,
                                   original_model: torch.nn.Module,
                                   converted_model,
                                   test_input: Union[np.ndarray, torch.Tensor]) -> dict:
        """验证 PyTorch 转换"""
        try:
            original_model.eval()

            # 确保输入是 PyTorch Tensor
            if isinstance(test_input, np.ndarray):
                test_input = torch.from_numpy(test_input).float()

            # 原始模型推理
            with torch.no_grad():
                original_output = original_model(test_input)
                if isinstance(original_output, tuple):
                    original_output = original_output[0]

            # 转换后模型推理（假设是 ONNX Runtime）
            if hasattr(converted_model, 'run'):  # ONNX Runtime session
                input_name = converted_model.get_inputs()[0].name
                converted_output = converted_model.run(None, {input_name: test_input.numpy()})
                converted_output = converted_output[0]
            else:
                # 假设是转换回的 PyTorch 模型
                with torch.no_grad():
                    converted_output = converted_model(test_input)
                    if isinstance(converted_output, tuple):
                        converted_output = converted_output[0]
                    converted_output = converted_output.numpy()

            # 计算差异
            diff = np.abs(original_output.numpy() - converted_output).max()

            return {
                "success": True,
                "max_difference": float(diff),
                "is_close": diff < 1e-5
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _verify_tensorflow_conversion(self,
                                      original_model,
                                      converted_model,
                                      test_input: Union[np.ndarray, torch.Tensor]) -> dict:
        """验证 TensorFlow 转换"""
        try:
            # 确保输入是 numpy array
            if isinstance(test_input, torch.Tensor):
                test_input = test_input.numpy()

            # 原始模型推理
            original_output = original_model(test_input)
            if isinstance(original_output, list):
                original_output = original_output[0]

            # 转换后模型推理
            if isinstance(converted_model, torch.nn.Module):
                # 转换为 PyTorch 模型
                test_input_torch = torch.from_numpy(test_input).float()
                with torch.no_grad():
                    converted_output = converted_model(test_input_torch)
                    if isinstance(converted_output, tuple):
                        converted_output = converted_output[0]
                    converted_output = converted_output.numpy()
            else:
                # 其他类型模型
                converted_output = converted_model(test_input)

            # 计算差异
            diff = np.abs(original_output.numpy() - converted_output).max()

            return {
                "success": True,
                "max_difference": float(diff),
                "is_close": diff < 1e-5
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def __del__(self):
        """清理临时目录"""
        import shutil
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 创建转换器实例
    converter = ONNXConverter(opset_version=13)

    # 示例：PyTorch 转 TensorFlow
    """
    # 假设你有一个 PyTorch 模型
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # 创建模型和示例输入
    pytorch_model = SimpleModel()
    dummy_input = torch.randn(1, 3, 32, 32)

    # 转换
    success = converter.pytorch_to_tensorflow(
        pytorch_model, 
        dummy_input, 
        "tf_model"
    )

    if success:
        print("转换成功！")
    """



# # 基础依赖
# pip install torch tensorflow onnx onnxruntime numpy
#
# # 转换工具（根据需要安装）
# pip install tf2onnx          # TensorFlow 转 ONNX
# pip install onnx-tf          # ONNX 转 TensorFlow
# pip install onnx2pytorch     # ONNX 转 PyTorch

# # 1. 创建转换器
# converter = ONNXConverter(opset_version=13)
#
# # 2. PyTorch 转 TensorFlow
# success = converter.pytorch_to_tensorflow(
#     pytorch_model,
#     dummy_input,
#     "tensorflow_model"
# )
#
# # 3. TensorFlow 转 PyTorch
# pytorch_model = converter.tensorflow_to_pytorch("tensorflow_model")
#
# # 4. 验证转换结果
# result = converter.verify_conversion(
#     original_model,
#     converted_model,
#     test_input,
#     "pytorch"
# )