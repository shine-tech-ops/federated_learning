import threading

from django.apps import AppConfig
from loguru import logger
from threading import Thread
import utils.common_constant as const

from learn_management.consumer.device_status_check import check_offline_devices
from learn_management.consumer.device_training_consumer import DeviceTrainingMqConsumer
from learn_management.consumer.device_register_consumer import DeviceRegisterMqConsumer

from learn_management.consumer.mqtt_device_heartbeat_consumer import DeviceMQTTHeartbeatConsumer
from learn_management.consumer.mqtt_device_training_consumer import DeviceMQTTTrainingConsumer
from learn_management.consumer.mqtt_device_common_consumer import DeviceMQTTCommonConsumer


class LearnManagementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "learn_management"

    def ready(self):
        # thread = Thread(name="RabbitMQ_thread_device_register", target=self.device_register_consumer)
        # thread.daemon = True
        # thread.start()
        # logger.info("RabbitMQ device register consumer daemon start.")

        # thread = Thread(name="RabbitMQ_thread_device_training", target=self.device_training_consumer)
        # thread.daemon = True
        # thread.start()
        # logger.info("RabbitMQ device training consumer daemon start.")

        # # 启动后台定时任务（延迟 5 秒后执行，避免数据库未就绪）
        # thread = threading.Thread(target=check_offline_devices, daemon=True)
        # thread.start()
        # logger.info("Starting background task for offline device check.")

        # # 启动 MQTT 消费线程
        # mqtt_thread = Thread(name="MQTT_Device_Heartbeat_Consumer", target=self.start_mqtt_heartbeat_consumer, daemon=True)
        # mqtt_thread.start()
        # logger.info("Starting MQTT Device Heartbeat Consumer thread.")

        # mqtt_thread = Thread(name="MQTT_Device_Training_Consumer", target=self.start_mqtt_training_consumer, daemon=True)
        # mqtt_thread.start()
        # logger.info("Starting MQTT Device Training Consumer thread.")

        # mqtt_thread  = Thread(name="MQTT_Device_Common_Consumer", target=self.start_mqtt_common_consumer, daemon=True)
        # mqtt_thread.start()
        # logger.info("Starting MQTT Device Common Consumer thread.")
        pass

    def device_register_consumer(self):
        client = DeviceRegisterMqConsumer()
        queue = const.MQ_DEVICE_REG_QUEUE
        exchange = const.MQ_DEVICE_REG_EXCHANGE
        client.consumer(exchange, queue)

    def device_training_consumer(self):
        client = DeviceTrainingMqConsumer()
        queue = const.MQ_DEVICE_TRAINING_QUEUE
        exchange = const.MQ_DEVICE_TRAINING_EXCHANGE
        client.consumer(exchange, queue)

    def start_mqtt_heartbeat_consumer(self):
        consumer = DeviceMQTTHeartbeatConsumer()
        consumer.consumer()

    def start_mqtt_training_consumer(self):
        consumer = DeviceMQTTTrainingConsumer()
        consumer.consumer()

    def start_mqtt_common_consumer(self):
        consumer = DeviceMQTTCommonConsumer()
        consumer.consumer()