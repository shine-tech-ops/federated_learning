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

    def publisher(self, exchange, data):
        try:
            self.connect()
            channel = self._connection.channel()
            channel.exchange_declare(exchange=exchange, exchange_type="fanout")
            message = json.dumps(data)
            channel.basic_publish(
                exchange=exchange,
                routing_key="",
                body=message,
                properties=pika.BasicProperties(delivery_mode=2),
            )
        finally:
            self._connection.close()

    def consumer(self, exchange, queue, auto_ack=True):
        try:
            self.connect()
            channel = self._connection.channel()
            channel.exchange_declare(exchange=exchange, exchange_type="fanout")
            result = channel.queue_declare(queue=queue, durable=True)
            queue_name = result.method.queue
            channel.queue_bind(exchange=exchange, queue=queue_name)
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
            self.consumer(queue)
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
