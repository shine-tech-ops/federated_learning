from app.utils.mqtt_client import MqttClient
from loguru import logger


class DeviceMQTTConsumer:
    def __init__(self):
        self.mqtt_client = MqttClient()
        self.topic = "device/status/#"  # 可根据配置文件读取

    def handle_message(self, topic, payload):
        """重写消息处理逻辑"""
        logger.info(f"[MQTT Consumer] Received on {topic}: {payload}")
        # 示例：解析 JSON 并触发联邦学习任务或状态更新
        try:
            data = payload if isinstance(payload, dict) else eval(payload)
            logger.info(f"Parsed message: {data}")
            # 在这里添加你的设备状态处理逻辑
        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")

    def consumer(self):
        try:
            self.mqtt_client.subscribe(self.topic)
            self.mqtt_client.start_listening()
            logger.info(f"Subscribed to MQTT topic: {self.topic}, listening...")
        except Exception as e:
            logger.error(f"Failed to start MQTT consumer: {e}")