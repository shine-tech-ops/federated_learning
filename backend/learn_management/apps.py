import threading

from django.apps import AppConfig
from loguru import logger
from threading import Thread
import utils.common_constant as const
from learn_management.consumer.device_register_consumer import (
    DeviceRegisterConsumer,
)
from learn_management.consumer.device_heartbeat_consumer import (
    DeviceHeartbeatConsumer,
)
from learn_management.consumer.device_status_check import check_offline_devices


class LearnManagementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "learn_management"

    def ready(self):
        thread = Thread(name="RabbitMQ_thread_device_register", target=self.device_register_consumer)
        thread.daemon = True
        thread.start()
        logger.info("RabbitMQ device register consumer daemon start.")

        thread = Thread(name="RabbitMQ_thread_device_heartbeat", target=self.device_heartbeat_consumer)
        thread.daemon = True
        thread.start()
        logger.info("RabbitMQ device heartbeat consumer daemon start.")

        # 启动后台定时任务（延迟 5 秒后执行，避免数据库未就绪）
        thread = threading.Thread(target=check_offline_devices, daemon=True)
        thread.start()
        logger.info("Starting background task for offline device check.")

    def device_register_consumer(self):
        client = DeviceRegisterConsumer()
        queue = const.MQ_DEVICE_REG_QUEUE
        exchange = const.MQ_DEVICE_REG_EXCHANGE
        client.consumer(exchange, queue)

    def device_heartbeat_consumer(self):
        client = DeviceHeartbeatConsumer()
        queue = const.MQ_DEVICE_HEARTBEAT_QUEUE
        exchange = const.MQ_DEVICE_HEARTBEAT_EXCHANGE
        client.consumer(exchange, queue)
