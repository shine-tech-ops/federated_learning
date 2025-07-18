from utils.rabbitmq_client import RabbitMQClient
from loguru import logger
import json
import traceback

class DeviceRegisterMqConsumer(RabbitMQClient):
    def callback(self, ch, method, properties, body):
        try:
            message = json.loads(body.decode())
            logger.info(f"DeviceRegisterConsumer Received: {message}")
            device_id = message.get("device_id")
            region_id = message.get("region_id")
            device_context = message.get("device_context")
            address = device_context.get("ip")

            if not all([device_id, region_id, address, device_context]):
                logger.error("DeviceRegisterConsumer Error: missing parameters")
                return

        except Exception as e:
            traceback.print_exc()
            logger.error(f"DeviceRegisterConsumer Error: {e}")

