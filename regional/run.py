from app import create_app
from threading import Thread
from loguru import logger

from flasgger import Swagger
from flask_cors import CORS
import os

app = create_app()
CORS(app, methods=["*"], origins="*", headers=["*"], supports_credentials=True, expose_headers=["*"], allow_headers=["*"], automatic_options=True)

# 启用 Swagger，默认模板会自动生成
swagger = Swagger(app)

if __name__ == '__main__':
    logger.add(
        "logs/backend.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        filter="",
        level="INFO",
        encoding="utf-8",
    )

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # 启动 MQTT 消费线程
        pass

    app.run(debug=True, host='0.0.0.0', port=8000)