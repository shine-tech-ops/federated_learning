import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Dict, Any, List, Union, Tuple
from config import Config, config


class Utils:
    def __init__(self, device_info_path: str = None):
        """
        初始化工具类
        Args:
            device_info_path: 设备信息JSON文件路径
        """
        self.device = None
        if device_info_path is None:
            _config = self.load_config()
            termux_home = _config.directory['termux_home']
            self.device_info_path = f"{termux_home}/transmit/device_info.json"
        else:
            self.device_info_path = device_info_path
        self.device_data = self._load_and_initialize_device_info(self.device_info_path)
        logger.info("Initializing Utils")

    def _load_and_initialize_device_info(self, file_path: str) -> Dict[str, Any]:
        """
        加载并初始化设备信息，确保包含所有必需的字段
        Args:
            file_path: JSON文件路径
        Returns:
            完整的设备信息字典
        """
        # 默认的设备信息结构
        default_device_info = {
            "region": {"id": 0},
            "task": {
                "id": "",
                "name": ""
            },
            "model": {
                "id": "",
                "name": "",
                "description": "",
                "file": ""
            },
            "train": {
                "index": 0,
                "round": 0,
                "aggregation_method": None,
                "progress": 0,
                "accuracy": 0,
                "loss": 0,
                "correct": 0,
                "total": 0
            },
            "device": {
                "id": "",
                "status": "offline",
                "timestamp": 0,
                "description": ""
            },
            "flower_server": {
                "host": "127.0.0.1",
                "port": 8080,
                "server_id": "",
                "running": ""
            }
        }

        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                # 深度合并：使用加载的数据覆盖默认值
                merged_data = self._deep_merge_dicts(default_device_info, loaded_data)
                logger.info(f"设备信息已从 {file_path} 加载")
                return merged_data
            else:
                logger.warning(f"文件 {file_path} 不存在，使用默认设备信息")
                return default_device_info.copy()
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON文件 {file_path} 时出错: {e}")
            return default_device_info.copy()
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
            return default_device_info.copy()

    def _deep_merge_dicts(self, default_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典，update_dict会覆盖default_dict中的值
        Args:
            default_dict: 默认字典
            update_dict: 更新字典
        Returns:
            合并后的字典
        """
        result = default_dict.copy()

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                # 直接覆盖值
                result[key] = value

        return result

    def _load_json_file(self, file_path: str) -> Union[Dict[str, Any], List[Any]]:
        """
        加载JSON文件
        Args:
            file_path: JSON文件路径
        Returns:
            解析后的数据，如果文件不存在或格式错误返回空字典或空列表
        """
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"File {file_path} does not exist")
                # 根据文件名判断返回类型
                if "device_info" in file_path:
                    return {}
                elif "tran_task" in file_path or "chat_history" in file_path:
                    return []
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            # 根据文件名判断返回类型
            if "device_info" in file_path:
                return {}
            elif "tran_task" in file_path or "chat_history" in file_path:
                return []
            return {}
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            if "device_info" in file_path:
                return {}
            elif "tran_task" in file_path or "chat_history" in file_path:
                return []
            return {}

    def _save_json_file(self, file_path: str, data: Union[Dict[str, Any], List[Any]]) -> bool:
        """
        保存数据到JSON文件
        Args:
            file_path: JSON文件路径
            data: 要保存的数据
        Returns:
            是否成功保存
        """
        try:
            # 创建目录（如果不存在）
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {e}")
            return False

    def get_device_info(self) -> Dict[str, Any]:
        return self._load_json_file(self.device_info_path)

    def update_device_info(self, data: Dict[str, Any] = None, **kwargs) -> Tuple[bool, str]:
        """
        更新设备信息，支持部分更新
        Args:
            data: 要更新的字段字典，可以包含任意字段
            **kwargs: 也可以使用关键字参数传递字段
        Returns:
            (是否成功更新, 描述信息)
        """
        try:
            # 合并所有更新数据
            updates = {}
            if data is not None:
                updates.update(data)
            if kwargs:
                updates.update(kwargs)

            if not updates:
                return False, "没有提供更新数据"

            # 记录更新了哪些字段
            updated_fields = []

            # 遍历更新数据
            for key, value in updates.items():
                # 特殊处理：如果值是字典且不是空字典，进行深度合并
                if (isinstance(value, dict) and value and
                        key in self.device_data and
                        isinstance(self.device_data[key], dict)):
                    # 深度合并字典
                    self._deep_merge(self.device_data[key], value)
                    updated_fields.append(f"{key}.* (深度合并)")
                else:
                    # 直接更新值
                    self.device_data[key] = value
                    updated_fields.append(key)

            # 如果更新了device字段，更新self.device
            if 'device' in updates:
                self.device = self.device_data.get('device', {})

            # 自动更新时间戳
            if 'device' in self.device_data and isinstance(self.device_data['device'], dict):
                self.device_data['device']['timestamp'] = int(datetime.now().timestamp())

            # 保存到文件
            if self._save_json_file(self.device_info_path, self.device_data):
                fields_str = ", ".join(updated_fields)
                logger.info(f"设备信息更新成功，更新的字段: {fields_str}")
                return True, f"成功更新了以下字段: {fields_str}"
            return False, "保存文件失败"
        except Exception as e:
            error_msg = f"更新设备信息时出错: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def _deep_merge(self, original: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        深度合并字典，将updates的内容合并到original中
        Args:
            original: 原始字典
            updates: 更新字典
        """
        for key, value in updates.items():
            if (key in original and
                    isinstance(original[key], dict) and
                    isinstance(value, dict)):
                # 递归合并嵌套字典
                self._deep_merge(original[key], value)
            else:
                # 直接设置值（覆盖或新增）
                original[key] = value

    def record_tran_task(self, task_file: str = "tran_task.json",
                         additional_data: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        记录训练任务，基于device_info.json文件读取并添加到tran_task.json的开头位置
        Args:
            task_file: 训练任务JSON文件路径
            additional_data: 额外的任务数据
        Returns:
            (是否成功记录, 描述信息)
        """
        try:
            # 读取现有任务数据
            tasks = self._load_json_file(task_file)

            # 确保tasks是列表
            if not isinstance(tasks, list):
                tasks = []

            # 从device_info.json中获取当前设备信息作为任务数据
            task_data = {
                "timestamp": datetime.now().isoformat(),
                "record_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device_info": self.device_data.copy()
            }

            # 添加额外数据
            if additional_data:
                task_data.update(additional_data)

            # 添加到列表开头
            tasks.insert(0, task_data)

            # 限制任务数量（可选，保留最近的100条记录）
            if len(tasks) > 100:
                tasks = tasks[:100]

            # 保存到文件
            if self._save_json_file(task_file, tasks):
                logger.info(f"训练任务已记录到 {task_file}")
                return True, "训练任务记录成功"
            return False, "保存训练任务文件失败"
        except Exception as e:
            error_msg = f"记录训练任务时出错: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def save_chat_history(self, user_message: str, assistant_message: str,
                          history_file: str = "chat_history.json",
                          additional_data: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        保存对话历史
        Args:
            user_message: 用户消息
            assistant_message: 助手回复
            history_file: 对话历史JSON文件路径
            additional_data: 额外的数据字段
        Returns:
            (是否成功保存, 描述信息)
        """
        try:
            # 读取现有对话历史
            history = self._load_json_file(history_file)

            # 确保history是列表
            if not isinstance(history, list):
                history = []

            # 创建对话记录
            chat_record = {
                "timestamp": datetime.now().isoformat(),
                "record_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_message,
                "assistant": assistant_message
            }

            # 添加额外数据
            if additional_data:
                chat_record.update(additional_data)

            # 添加到历史记录
            history.append(chat_record)

            # 限制历史记录数量（可选，保留最近的1000条记录）
            if len(history) > 1000:
                history = history[-1000:]

            # 保存到文件
            if self._save_json_file(history_file, history):
                logger.info(f"对话历史已保存到 {history_file}")
                return True, "对话历史保存成功"
            return False, "保存对话历史文件失败"
        except Exception as e:
            error_msg = f"保存对话历史时出错: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def get_chat_history(self, history_file: str = "chat_history.json",
                         limit: int = None, offset: int = 0,
                         reverse: bool = False) -> List[Dict[str, Any]]:
        """
        获取对话历史
        Args:
            history_file: 对话历史JSON文件路径
            limit: 限制返回的记录数量（None表示返回所有）
            offset: 起始偏移量
            reverse: 是否反转顺序（True表示最新在前）
        Returns:
            对话历史列表
        """
        try:
            # 读取对话历史
            history = self._load_json_file(history_file)

            # 确保history是列表
            if not isinstance(history, list):
                return []

            # 如果需要反转顺序
            if reverse:
                history = list(reversed(history))

            # 应用偏移量和限制
            if offset > 0 or limit is not None:
                start = offset
                end = offset + limit if limit is not None else None
                return history[start:end]

            return history
        except Exception as e:
            logger.error(f"获取对话历史时出错: {e}")
            return []

    def get_device_info(self) -> Dict[str, Any]:
        """获取完整的设备信息"""
        return self.device_data

    def get_region_info(self) -> Dict[str, Any]:
        """获取区域信息"""
        return self.device_data.get('region', {})

    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return self.device_data.get('task', {})

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.device_data.get('model', {})

    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        return self.device_data.get('train', {})

    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态信息"""
        return self.device_data.get('device', {})

    def get_flower_server_info(self) -> Dict[str, Any]:
        """获取Flower服务器信息"""
        return self.device_data.get('flower_server', {})

    def update_training_progress(self, progress: float = None,
                                 accuracy: float = None,
                                 loss: float = None,
                                 correct: int = None,
                                 total: int = None,
                                 round: int = None,
                                 index: int = None,
                                 aggregation_method: str = None) -> Tuple[bool, str]:
        """
        更新训练进度和指标
        Args:
            progress: 训练进度 (0-100)
            accuracy: 准确率
            loss: 损失值
            correct: 正确样本数
            total: 总样本数
            round: 训练轮次
            index: 当前轮次
            aggregation_method: 聚合方法
        Returns:
            (是否成功更新, 描述信息)
        """
        try:
            # 获取训练信息，如果不存在则创建
            if 'train' not in self.device_data:
                self.device_data['train'] = {
                    "index": 0,
                    "round": 0,
                    "aggregation_method": None,
                    "progress": 0,
                    "accuracy": 0,
                    "loss": 0,
                    "correct": 0,
                    "total": 0
                }

            train_info = self.device_data['train']
            updated_fields = []

            # 更新提供的字段
            if progress is not None:
                train_info['progress'] = progress
                updated_fields.append('progress')
            if accuracy is not None:
                train_info['accuracy'] = accuracy
                updated_fields.append('accuracy')
            if loss is not None:
                train_info['loss'] = loss
                updated_fields.append('loss')
            if correct is not None:
                train_info['correct'] = correct
                updated_fields.append('correct')
            if total is not None:
                train_info['total'] = total
                updated_fields.append('total')
            if round is not None:
                train_info['round'] = round
                updated_fields.append('round')
            if index is not None:
                train_info['index'] = index
                updated_fields.append('index')
            if aggregation_method is not None:
                train_info['aggregation_method'] = aggregation_method
                updated_fields.append('aggregation_method')

            # 计算准确率（如果提供了correct和total）
            if correct is not None and total is not None and total > 0:
                calculated_accuracy = correct / total
                # 只有当用户没有提供accuracy时才自动计算
                if accuracy is None:
                    train_info['accuracy'] = calculated_accuracy
                    if 'accuracy' not in updated_fields:
                        updated_fields.append('accuracy')

            # 保存到文件
            if self._save_json_file(self.device_info_path, self.device_data):
                fields_str = ", ".join(updated_fields)
                logger.info(f"训练进度更新成功，更新的字段: {fields_str}")
                return True, f"训练进度更新成功: {fields_str}"
            return False, "保存训练进度失败"
        except Exception as e:
            error_msg = f"更新训练进度时出错: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def reset_device_info(self) -> Tuple[bool, str]:
        """
        重置设备信息为默认值
        Returns:
            (是否成功重置, 描述信息)
        """
        try:
            # 使用默认的设备信息结构
            default_device_info = {
                "region": {"id": 0},
                "task": {
                    "id": "",
                    "name": ""
                },
                "model": {
                    "id": "",
                    "name": "",
                    "description": "",
                    "file": ""
                },
                "train": {
                    "index": 0,
                    "round": 0,
                    "aggregation_method": None,
                    "progress": 0,
                    "accuracy": 0,
                    "loss": 0,
                    "correct": 0,
                    "total": 0
                },
                "device": {
                    "id": "",
                    "status": "offline",
                    "timestamp": int(datetime.now().timestamp()),
                    "description": ""
                },
                "flower_server": {
                    "host": "127.0.0.1",
                    "port": 8080,
                    "server_id": "",
                    "running": ""
                }
            }

            self.device_data = default_device_info.copy()
            self.device = self.device_data.get('device', {})

            # 保存到文件
            if self._save_json_file(self.device_info_path, self.device_data):
                logger.info("设备信息已重置为默认值")
                return True, "设备信息重置成功"
            return False, "保存重置后的设备信息失败"
        except Exception as e:
            error_msg = f"重置设备信息时出错: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def load_config() -> Config:
        return config


# 使用示例
if __name__ == "__main__":
    print("=== 测试更新后的Utils类 ===")

    # 初始化工具类
    utils = Utils("transmit/device_info.json")

    # 1. 显示当前设备信息
    print("\n1. 当前设备信息:")
    device_info = utils.get_device_info()
    print(json.dumps(device_info, indent=2, ensure_ascii=False))

    # 2. 获取区域信息
    print("\n2. 区域信息:")
    region_info = utils.get_region_info()
    print(f"Region ID: {region_info.get('id')}")

    # 3. 只更新region字段
    success, msg = utils.update_device_info(
        region={"id": 1, "name": "北京区域"}
    )
    print(f"\n3. 更新region字段: {success}, {msg}")
    print("更新后的区域信息:", json.dumps(utils.get_region_info(), indent=2))

    # 4. 更新多个字段（包括新增的name字段）
    success, msg = utils.update_device_info({
        "region": {"id": 2, "description": "华东区域"},
        "task": {"priority": "high"},  # 新增字段
        "device": {"location": "机房A"}  # 新增字段
    })
    print(f"\n4. 更新多个字段（包括新增字段）: {success}, {msg}")
    print("更新后的设备信息:", json.dumps(utils.get_device_info(), indent=2, ensure_ascii=False))

    # 5. 测试训练进度更新（不提供accuracy，自动计算）
    success, msg = utils.update_training_progress(
        progress=30.0,
        correct=450,
        total=600,
        round=4,
        index=1
    )
    print(f"\n5. 更新训练进度（自动计算准确率）: {success}, {msg}")
    train_info = utils.get_training_info()
    print(f"训练进度: {train_info.get('progress')}%")
    print(f"准确率: {train_info.get('accuracy')}")
    print(f"正确数/总数: {train_info.get('correct')}/{train_info.get('total')}")
    print(f"轮次: {train_info.get('index')}/{train_info.get('round')}")

    # 6. 测试记录训练任务
    success, msg = utils.record_tran_task(
        additional_data={"operation": "测试记录", "user": "测试员"}
    )
    print(f"\n6. 记录训练任务: {success}, {msg}")

    # 7. 测试重置功能
    print("\n7. 重置设备信息测试:")
    success, msg = utils.reset_device_info()
    print(f"重置结果: {success}, {msg}")
    print("重置后的设备状态:", json.dumps(utils.get_device_status(), indent=2))

    # 8. 测试从文件重新加载
    print("\n8. 重新加载设备信息:")
    utils = Utils("transmit/device_info.json")  # 重新初始化
    print("重新加载后的设备信息:")
    print(json.dumps(utils.get_device_info(), indent=2, ensure_ascii=False))