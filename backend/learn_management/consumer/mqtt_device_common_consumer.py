import json

from utils.mqtt_client import MqttClient
from utils.rabbitmq_client import RabbitMQClient
import utils.common_constant as consts
from loguru import logger


class DeviceMQTTCommonConsumer:
    def __init__(self):
        self.mqtt_client = MqttClient()
        self.topic = "device/common/#"  # 可根据配置文件读取
        self.mqtt_client.handle_message = self.handle_message

    def handle_message(self, topic, payload):
        """重写消息处理逻辑"""
        # 示例：解析 JSON 并触发联邦学习任务或状态更新
        try:
            data = json.loads(payload)
            # 在这里添加你的设备状态处理逻辑
            region_id = data.get("region_id")
            device_id = data.get("device_id")
            device_context = data.get("device_context")
            event_name = data.get("event_name")
            evnet_context = data.get("evnet_context")
            if not event_name:
                logger.error("DeviceMQTTCommonConsumer Error: missing event_name")
                return
            if not evnet_context:
                logger.error("DeviceMQTTCommonConsumer Error: missing evnet_context")
                return
            if not region_id:
                logger.error("DeviceMQTTCommonConsumer Error: missing region_id")
                return
            if not device_id:
                logger.error("DeviceMQTTCommonConsumer Error: missing device_id")
                return
            # 调用注册接口
            self.device_common_event(region_id, device_id, device_context, event_name, evnet_context)
        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")

    def device_common_event(self, region_id, device_id,device_context, event_name, evnet_context):
        """
        设备通用事件
        """
        msg_data =  {
                "device_id": device_id,
                "region_id": region_id,
                "device_context": device_context,
                "event_name": event_name,
                "evnet_context": evnet_context
            }
        RabbitMQClient().publisher(
            consts.MQTT_DEVICE_COMMON_EXCHANGE,
            msg_data,
        )

    def consumer(self):
        try:
            self.mqtt_client.subscribe(self.topic)
            self.mqtt_client.start_listening()
            logger.info(f"Subscribed to MQTT topic: {self.topic}, listening...")
        except Exception as e:
            logger.error(f"Failed to start MQTT consumer: {e}")