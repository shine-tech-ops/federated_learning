import json

from utils.mqtt_client import MqttClient
from utils.rabbitmq_client import RabbitMQClient
import utils.common_constant as consts
from loguru import logger


class DeviceMQTTTrainingConsumer:
    def __init__(self):
        self.mqtt_client = MqttClient()
        self.topic = "device/training/#"  # 可根据配置文件读取
        self.mqtt_client.handle_message = self.handle_message

    def handle_message(self, topic, payload):
        """重写消息处理逻辑"""
        # 示例：解析 JSON 并触发联邦学习任务或状态更新
        try:
            data = json.loads(payload)
            logger.info(f"Parsed message: {data}")
            # 在这里添加你的设备状态处理逻辑
            region_id = data.get("region_id")
            device_id = data.get("device_id")
            device_context = data.get("device_context")
            training_context = data.get("training_context")
            task_id = data.get("task_id")
            model_id = data.get("model_id")
            model_version_id = data.get("model_version_id")
            training_status = data.get("training_status")

            if not region_id:
                logger.error("DeviceMQTTTrainingConsumer Error: missing region_id")
                return
            if not device_id:
                logger.error("DeviceMQTTTrainingConsumer Error: missing device_id")
                return
            if not task_id:
                logger.error("DeviceMQTTTrainingConsumer Error: missing task_id")
                return
            if not model_id:
                logger.error("DeviceMQTTTrainingConsumer Error: missing model_id")
                return
            if not model_version_id:
                logger.error("DeviceMQTTTrainingConsumer Error: missing model_version_id")
                return
            if not training_status:
                logger.error("DeviceMQTTTrainingConsumer Error: missing training_status")
                return
            msg_data = {
                "device_id": device_id,
                "region_id": region_id,
                "device_context": device_context,
                "training_context": training_context,
                "task_id": task_id,
                "model_id": model_id,
                "model_version_id": model_version_id,
                "training_status": training_status,
            }
            RabbitMQClient().publisher(
                consts.MQ_DEVICE_TRAINING_EXCHANGE,
                msg_data,
            )
        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")

    def consumer(self):
        try:
            self.mqtt_client.subscribe(self.topic)
            self.mqtt_client.start_listening()
            logger.info(f"Subscribed to MQTT topic: {self.topic}, listening...")
        except Exception as e:
            logger.error(f"Failed to start MQTT consumer: {e}")