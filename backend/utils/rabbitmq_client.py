import time
import pika
import json
import conf.env as env
from loguru import logger


class RabbitMQClient:
    """RabbitMQ tools"""

    def __init__(self):
        self._host = env.RABBITMQ_HOST
        self._port = env.RABBITMQ_PORT
        self._username = env.RABBITMQ_USER
        self._password = env.RABBITMQ_PASSWORD
        self._credentials = pika.PlainCredentials(self._username, self._password)
        self._connection = None

    def publisher(self, exchange, data, routing_key="task"):
        """
        发布消息到 RabbitMQ
        
        Args:
            exchange: Exchange 名称
            data: 要发送的数据（字典）
            routing_key: 路由键，默认为 "task"
        """
        try:
            self.connect()
            channel = self._connection.channel()
            channel.exchange_declare(exchange=exchange, exchange_type="direct", durable=True)
            message = json.dumps(data)
            channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=message,
                properties=pika.BasicProperties(delivery_mode=2),
            )
            logger.debug(f"消息已发布到 Exchange: {exchange}, Routing Key: {routing_key}")
        finally:
            self._connection.close()

    def consumer(self, exchange, queue, routing_key="task", auto_ack=True):
        """
        消费 RabbitMQ 消息
        
        Args:
            exchange: Exchange 名称
            queue: 队列名称
            routing_key: 路由键，默认为 "task"
            auto_ack: 是否自动确认消息
        """
        try:
            self.connect()
            channel = self._connection.channel()
            channel.exchange_declare(exchange=exchange, exchange_type="direct", durable=True)
            result = channel.queue_declare(queue=queue, durable=True)
            queue_name = result.method.queue
            # 使用 routing_key 绑定队列到 exchange
            channel.queue_bind(exchange=exchange, queue=queue_name, routing_key=routing_key)
            channel.basic_consume(
                queue=queue_name, on_message_callback=self.callback, auto_ack=auto_ack
            )
            channel.start_consuming()
        except pika.exceptions.ConnectionClosed:
            logger.info("RabbitMQ connection closed, reconnecting after 5 seconds.")
            time.sleep(5)
            self._connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self._host,
                    port=self._port,
                    credentials=self._credentials,
                    virtual_host="/",
                )
            )
            logger.info("RabbitMQ reconnected.")
            # 重连后重新消费，传递所有必要参数
            self.consumer(exchange, queue, routing_key, auto_ack)
        finally:
            # 确保通道和连接被正确关闭
            channel.stop_consuming()
            channel.close()
            self._connection.close()

    def callback(ch, method, properties, body):
        logger.info(f"RabbitMQ Received {body}")

    def connect(self):
        while not self._connection:
            try:
                self._connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=self._host,
                        port=self._port,
                        credentials=self._credentials,
                        virtual_host="/",
                    )
                )
            except Exception as e:
                logger.info("RabbitMQ connection error, retrying in 5 seconds...")
                time.sleep(5)
