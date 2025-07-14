import app.conf.env as env

from flask import Blueprint, jsonify, request
from loguru import logger
from flasgger import swag_from
import app.utils.common_constants as consts

from app.utils.rabbitmq_client import RabbitMQClient
from app.utils.common import generate_auth_token
main = Blueprint("main", __name__)


@swag_from({
    'tags': ['设备管理'],
    'description': '客户端调用此接口进行注册',
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'properties': {
                'device_id': {'type': 'string', 'example': 'device_001'},
                'region': {'type': 'string', 'example': 'asia-east'},
                'address': {'type': 'string', 'example': '192.168.1.100:5000'},
                'device_context': {'type': 'object', 'example': {
                    'cpu': '16',
                    'memory': '16',
                    'disk': '100',
                    'gpu': '1'
                }}
            },
            'required': ['device_id', 'region', 'address']
        }
    }],
    'responses': {
        '200': {
            'description': '注册成功',
            'schema': {
                'type': 'object',
                'properties': {
                    'code': {'type': 'integer'},
                    'msg': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'token': {'type': 'string'},
                            'region': {'type': 'string'},
                            'address': {'type': 'string'},
                            'mqtt': {
                                'type': 'object',
                                'properties': {
                                    'host': {'type': 'string'},
                                    'port': {'type': 'integer'},
                                    'username': {'type': 'string'},
                                    'password': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            }
        },
        '400': {
            'description': '缺少必要字段',
            'schema': {
                'type': 'object',
                'properties': {
                    'code': {'type': 'integer'},
                    'msg': {'type': 'string'},
                    'data': {'type': 'object'}
                }
            }
        }
    }
})
@main.route("/register", methods=["POST"])
def register_device():
    """
    对应「边缘设备注册.html」中的注册流程。
    客户端调用此接口进行注册。
    """
    data = request.get_json()
    device_id = data.get("device_id")
    device_context = data.get("device_context")
    region = data.get("region")
    address = data.get("address")  # 客户端连接到区域服务器的地址

    if not all([device_id, region, address]):
        return jsonify(
            {"code": "error", "msg": "Missing required fields", "data": data}
        )
    logger.info(
        f"[Registry] 设备注册成功: Device ID={device_id}, Region={region}, Address={address}"
    )
    
    auth_token = generate_auth_token(device_id, region)

    # 发送注册消息到mq
    RabbitMQClient().publisher(
        consts.MQ_DEVICE_REG_EXCHANGE,
        {
            "device_id": device_id,
            "region": region,
            "address": address,
            "device_context": device_context, 
            "token": auth_token
        },
    )
    # 真实场景中，这里会返回一个认证 Token
    return jsonify(
        {
            "code": "success",
            "msg": "Device registered successfully",
            "data": {
                "token": auth_token,
                "region": region,
                "address": address,
                "mqtt_config": {
                    "host": env.MQTT_BROKER_HOST,
                    "port": env.MQTT_BROKER_PORT,
                    "username": env.MQTT_USER,
                    "password": env.MQTT_PASSWORD,
                },
            },
        }
    )
