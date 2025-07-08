from app import create_app
from threading import Thread
from loguru import logger
from app.consumer.mqtt_device_consumer import DeviceMQTTConsumer
from flasgger import Swagger

app = create_app()

# 启用 Swagger，默认模板会自动生成
swagger = Swagger(app)

def start_mqtt_consumer():
    consumer = DeviceMQTTConsumer()
    consumer.consumer()

if __name__ == '__main__':
    logger.add(
        "logs/backend.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        filter="",
        level="INFO",
        encoding="utf-8",
    )

    # 启动 MQTT 消费线程
    mqtt_thread = Thread(name="MQTT_Device_Consumer", target=start_mqtt_consumer, daemon=True)
    mqtt_thread.start()
    logger.info("Starting MQTT consumer thread.")

    app.run(debug=True, host='0.0.0.0', port=8000)