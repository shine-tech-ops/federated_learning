#!/usr/bin/env python3
"""
边缘设备主程序
模拟联邦学习客户端
"""

import time
import threading
import sys
import os
from loguru import logger
from utils import Utils

# 添加共享模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from mqtt_handler import MQTTHandler
from flower_client import FlowerClient
from mnist_trainer import MNISTTrainer
from http_client import HTTPClient

trainer_class = {
    "cnn": MNISTTrainer,
    # "proxy":
}

class EdgeDevice:
    """边缘设备主类"""
    
    def __init__(
        self,
        device_id: str,
        mqtt_config: dict,
        http_config: dict = None,
        heartbeat_interval: int = 30,
    ):
        self.device_id = device_id
        self.mqtt_handler = MQTTHandler(device_id, mqtt_config)
        self.flower_client = None
        self.trainer = None
        self.current_task = None
        self.running = False
        self.region_id = None  # 区域节点ID，从注册信息或配置中获取
        self.heartbeat_interval = heartbeat_interval
        
        # HTTP客户端配置（用于发送心跳到中央服务器）
        http_base_url = http_config.get('base_url') if http_config else 'http://localhost:8085'
        http_timeout = http_config.get('timeout', 10) if http_config else 10
        self.http_client = HTTPClient(base_url=http_base_url, timeout=http_timeout)
        
        # 设置 MQTT 消息回调
        self.mqtt_handler.set_message_callback(self._handle_mqtt_message)
    
    def start(self):
        """启动设备"""
        logger.info(f"启动边缘设备: {self.device_id}")
        
        try:
            # 连接 MQTT
            self.mqtt_handler.connect()
            
            # 订阅设备专属主题
            self.mqtt_handler.subscribe_device_commands()
            
            # 设置运行状态
            self.running = True
            
            # 主循环
            self._main_loop()
            
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭设备...")
        except Exception as e:
            logger.error(f"设备运行错误: {e}")
        finally:
            self.stop()
    
    def _handle_mqtt_message(self, topic: str, message: dict):
        """处理 MQTT 消息"""
        try:
            action = message.get('action')
            logger.info(f"收到 MQTT 消息: {action}")
            
            if action == 'task_start':
                self._handle_task_start(message)
            elif action == 'task_pause':
                self._handle_task_pause(message)
            elif action == 'task_resume':
                self._handle_task_resume(message)
            elif action == 'task_stop':
                self._handle_task_stop(message)
            else:
                logger.warning(f"未知动作: {action}")
                
        except Exception as e:
            logger.error(f"处理 MQTT 消息错误: {e}")
    
    def _handle_task_start(self, message: dict):
        """处理任务开始"""
        logger.info(f"开始处理任务: {message['task_id']}")
        
        try:
            # 保存任务信息
            self.current_task = message
            
            # 获取 Flower 服务器信息
            print("message", message)
            flower_server = message.get('flower_server', {})
            if not flower_server:
                logger.error("缺少 Flower 服务器信息")
                return

            # 更新传递数据
            model_info = message.get('model_info', {})
            model_version = message.get('model_version', {})
            device_info = message.get('device_info', {})
            utils = Utils()
            utils.update_device_info(
                task={'id': message.get('task_id', None), 'name': message.get('task_name', None)},
                model={'id': model_info['id'], 'name': model_info['name'], 'description': model_info['description'], 'file': model_version['model_file']},
                train={
                    "index": 0,
                    "round": message.get('rounds', 0),
                    "aggregation_method": message.get('aggregation_method', None),
                    "progress": 0,
                },
                device={'description': device_info['description']},
                flower_server=flower_server,
            )

            # 创建训练器
            self.trainer = MNISTTrainer(
                device_id=self.device_id,
                task_data=message
            )
            
            # 获取 Flower 服务器地址
            server_host = flower_server.get('host', 'localhost')
            server_port = flower_server.get('port', 8080)
            
            # 如果 host 是 0.0.0.0 或 localhost，尝试从 MQTT broker 获取区域节点 IP
            if server_host in ['0.0.0.0', 'localhost', '127.0.0.1']:
                # 从 MQTT 连接信息中获取 broker 的 IP（区域节点和 MQTT broker 通常在同一台机器）
                try:
                    mqtt_host = self.mqtt_handler.client._host
                    if mqtt_host and mqtt_host not in ['localhost', '127.0.0.1']:
                        server_host = mqtt_host
                        logger.info(f"使用 MQTT broker IP 作为 Flower 服务器地址: {server_host}")
                    else:
                        logger.warning(f"无法从 MQTT broker 获取 IP，使用默认地址: {server_host}")
                except Exception as e:
                    logger.warning(f"获取 MQTT broker IP 失败: {e}，使用原始地址: {server_host}")
            
            server_address = f"{server_host}:{server_port}"
            logger.info(f"fed 服务器地址: {server_address}")
            
            # 创建 Flower 客户端
            self.flower_client = FlowerClient(
                device_id=self.device_id,
                trainer=self.trainer,
                server_address=server_address,
                task_id=message.get('task_id'),
                region_id=self.region_id,
                log_callback=self._report_training_log
            )
            
            # 在后台线程启动联邦学习
            fl_thread = threading.Thread(
                target=self._start_federated_learning,
                daemon=True
            )
            fl_thread.start()
            
            logger.info(f"设备 {self.device_id} 开始联邦学习")
            
        except Exception as e:
            logger.error(f"处理任务开始错误: {e}")
    
    def _handle_task_pause(self, message: dict):
        """处理任务暂停"""
        logger.info(f"暂停任务: {message['task_id']}")
        if self.flower_client:
            self.flower_client.pause()
    
    def _handle_task_resume(self, message: dict):
        """处理任务恢复"""
        logger.info(f"恢复任务: {message['task_id']}")
        if self.flower_client:
            self.flower_client.resume()
    
    def _handle_task_stop(self, message: dict):
        """处理任务停止"""
        logger.info(f"停止任务: {message['task_id']}")
        if self.flower_client:
            self.flower_client.stop()
        self.current_task = None
    
    def _start_federated_learning(self):
        """启动联邦学习"""
        try:
            if self.flower_client:
                self.flower_client.start()
        except Exception as e:
            logger.error(f"联邦学习错误: {e}")
    
    def _main_loop(self):
        """主循环"""
        logger.info("进入主循环...")
        
        while self.running:
            try:
                # 发送心跳
                self._send_heartbeat()
                
                # 休眠
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(5)
    
    def _send_heartbeat(self):
        """发送心跳（同时发送MQTT心跳和HTTP心跳到中央服务器）"""
        try:
            heartbeat_data = {
                "device_id": self.device_id,
                "status": "online",
                "timestamp": time.time(),
                "current_task": self.current_task['task_id'] if self.current_task else None
            }
            
            # 发送MQTT心跳（给区域节点）
            
            # 发送HTTP心跳到中央服务器
            if self.region_id:
                device_context = {
                    "status": "online",
                    "timestamp": time.time(),
                    "current_task": self.current_task['task_id'] if self.current_task else None
                }
                self.http_client.send_heartbeat(
                    device_id=self.device_id,
                    region_node=self.region_id,
                    device_context=device_context
                )
            else:
                logger.warning(f"设备 {self.device_id} 未设置region_id，跳过HTTP心跳")
            
        except Exception as e:
            logger.error(f"发送心跳失败: {e}")

    def _report_training_log(self, log_data: dict):
        """向中央服务器上传训练/评估日志"""
        try:
            task_id = log_data.get("task") or (self.current_task.get('task_id') if self.current_task else None)
            if not task_id:
                logger.warning("当前任务ID缺失，跳过上传训练日志")
                return

            if not self.region_id:
                logger.warning("未设置 region_id，跳过上传训练日志")
                return

            payload = {
                "task": task_id,
                "region_node": log_data.get("region_node") or self.region_id,
                "device_id": log_data.get("device_id") or self.device_id,
                "round": log_data.get("round"),
                "phase": log_data.get("phase", "system"),
                "level": log_data.get("level", "INFO"),
                "loss": log_data.get("loss"),
                "accuracy": log_data.get("accuracy"),
                "num_examples": log_data.get("num_examples"),
                "metrics": log_data.get("metrics"),
                "message": log_data.get("message"),
                "error_message": log_data.get("error_message"),
                "duration": log_data.get("duration"),
                "extra_data": log_data.get("extra_data"),
            }

            success = self.http_client.upload_training_logs(payload)
            if not success:
                logger.warning(f"训练日志上传失败: {payload.get('message')}")
        except Exception as e:
            logger.error(f"上传训练日志异常: {e}")
    
    def stop(self):
        """停止设备"""
        logger.info("正在停止边缘设备...")
        self.running = False
        
        if self.flower_client:
            self.flower_client.stop()
        
        if self.mqtt_handler:
            self.mqtt_handler.close()
        
        if self.http_client:
            self.http_client.close()
        
        logger.info("边缘设备已停止")


if __name__ == "__main__":
    # 设备ID、区域ID、中央服务器地址可通过命令行覆盖配置文件
    config = Utils.load_config()
    device_id = sys.argv[1] if len(sys.argv) > 1 else config.device_id
    region_id = int(sys.argv[2]) if len(sys.argv) > 2 else int(config.region_id)
    central_server_url = sys.argv[3] if len(sys.argv) > 3 else config.http['base_url']
    
    # 配置日志
    log_file = config.logging['file'].format(device_id=device_id)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        format=config.logging['format'],
        rotation=config.logging['max_size'],
        level=config.logging['level'],
        encoding="utf-8",
    )
    
    # MQTT 配置
    mqtt_config = config.mqtt
    
    # HTTP 配置（用于发送心跳到中央服务器）
    http_config = {
        **config.http,
        'base_url': central_server_url,
    }

    # 初始化工具类
    initialize_utils = Utils()
    success, msg = initialize_utils.update_device_info(
        device={"id": device_id},
        region={"id": region_id},
    )
    if not success:
        print("警告：更新失败，请检查工具类是否有误，设备启动已终止！")
    else:
        # 启动设备
        device = EdgeDevice(device_id, mqtt_config, http_config, heartbeat_interval=config.heartbeat_interval)
        device.region_id = region_id  # 设置区域节点ID
        device.start()
