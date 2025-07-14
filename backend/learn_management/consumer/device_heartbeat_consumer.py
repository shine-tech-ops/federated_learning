from django.utils import timezone
from utils.rabbitmq_client import RabbitMQClient
from loguru import logger
import json
import traceback

class DeviceHeartbeatConsumer(RabbitMQClient):
    def callback(self, ch, method, properties, body):
        try:
            message = json.loads(body.decode())
            logger.info(f"DeviceHeartbeatConsumer Received: {message}")
            device_id = message.get("device_id")
            region = message.get("region")


            if not all([device_id, region]):
                logger.error("DeviceHeartbeatConsumer Error: missing parameters")
                return

            # 调用注册接口
            self.device_heartbeat(device_id, region)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"DeviceHeartbeatConsumer Error: {e}")

    def device_heartbeat(self, device_id, region):
        """
        设备注册
        """
        from learn_management.models import EdgeNode
        node, created = EdgeNode.objects.update_or_create(
            device_id=device_id,
            region_id=region,
            defaults={
                "last_heartbeat": timezone.now(),
                "status": "online",
            }
        )
        if created:
            logger.info(f"DeviceHeartbeatConsumer Success: device_id={device_id}")
        else:
            logger.info(f"DeviceHeartbeatConsumer Update: device_id={device_id}")

