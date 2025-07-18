import json
import uuid
import paho.mqtt.client as mqtt
import paho.mqtt.publish as mqtt_publish
import conf.env as env
from loguru import logger
from paho.mqtt.enums import CallbackAPIVersion


class MqttClient:
    """Mqtt tools"""

    def __init__(self):
        self._host = env.MQTT_BROKER_HOST
        self._port = env.MQTT_BROKER_PORT
        self._username = env.MQTT_USER
        self._password = env.MQTT_PASSWORD
        self._client_id = str(uuid.uuid4())
        self._client = None
        self._subscribed_topics = set()

    def mqtt_publisher(self, topic, data):
        """发布单条 MQTT 消息"""
        logger.info(f"Publishing mqtt data: {data} to {topic}")
        mqtt_publish.single(
            topic,
            payload=json.dumps(data),
            hostname=self._host,
            port=self._port,
            client_id=self._client_id,
            auth={
                "username": self._username,
                "password": self._password,
            },
        )

    def mqtt_will_publisher(self, topic, data, will_qos=1, will_retain=True):
        """设置遗嘱消息并发布"""
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2, client_id=self._client_id
        )
        client.username_pw_set(self._username, self._password)
        try:
            client.connect(host=self._host, port=int(self._port))
            client.will_set(
                topic, payload=json.dumps(data), qos=will_qos, retain=will_retain
            )
            client.publish(
                topic, payload=json.dumps(data), qos=will_qos, retain=will_retain
            )
        finally:
            client.disconnect()

    def connect(self):
        """连接到 MQTT Broker（用于订阅）"""
        if not self._client:
            self._client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self._client_id,
            )
            self._client.username_pw_set(self._username, self._password)
            self._client.on_message = self._on_message
            self._client.connect(self._host, int(self._port))
            logger.info("MQTT client connected for subscription.")

    def subscribe(self, topic, qos=1):
        if not hasattr(self, '_subscribed_topics'):
            self._subscribed_topics = set()
        if topic in self._subscribed_topics:
            logger.warning(f"Topic {topic} already subscribed, skipping.")
            return
        self.connect()
        self._client.subscribe(topic, qos=qos)
        self._subscribed_topics.add(topic)
        logger.info(f"Subscribed to topic: {topic}")

    def start_listening(self):
        """开始监听消息"""
        self.connect()
        self._client.loop_start()
        logger.info("Started MQTT message loop.")

    def stop_listening(self):
        """停止监听消息"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            logger.info("Stopped MQTT message loop.")

    def _on_message(self, client, userdata, msg):
        """消息回调函数"""
        try:
            payload = msg.payload.decode("utf-8")
            logger.info(f"Received MQTT message on topic '{msg.topic}': {payload}")
            # 可以在这里添加自定义的消息处理逻辑
            self.handle_message(msg.topic, payload)
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def handle_message(self, topic, payload):
        """
        自定义消息处理方法（可重写）
        示例：解析 JSON 并触发事件
        """
        try:
            pass
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message.")
