from utils.rabbitmq_client import RabbitMQClient
from loguru import logger
import json
import traceback

class DeviceRegisterConsumer(RabbitMQClient):
    def callback(self, ch, method, properties, body):
        try:
            message = json.loads(body.decode())
            logger.info(f"DeviceRegisterConsumer Received: {message}")
            device_id = message.get("device_id")
            region = message.get("region")
            address = message.get("address")
            device_context = message.get("device_context")

            if not all([device_id, region, address, device_context]):
                logger.error("DeviceRegisterConsumer Error: missing parameters")
                return

            # 调用注册接口
            self.device_register(device_id, region, address, device_context)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"DeviceRegisterConsumer Error: {e}")

    def device_register(self, device_id, region, address, device_context):
        """
        设备注册
        """
        from learn_management.models import EdgeNode
        node, created = EdgeNode.objects.update_or_create(
            device_id=device_id,
            region_id=region,
            defaults={
                "ip_address": address,
                "device_context": device_context,
            }
        )
        if created:
            logger.info(f"DeviceRegisterConsumer Success: device_id={device_id}")
        else:
            logger.info(f"DeviceRegisterConsumer Update: device_id={device_id}")

