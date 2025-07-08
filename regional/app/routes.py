from flask import Blueprint, jsonify, request
from loguru import logger
from app.utils.redis_client import RedisClient
from flasgger import swag_from

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
                'client_id': {'type': 'string', 'example': 'device_001'},
                'region': {'type': 'string', 'example': 'asia-east'},
                'address': {'type': 'string', 'example': '192.168.1.100:5000'}
            },
            'required': ['client_id', 'region', 'address']
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

    RedisClient().set(device_id, device_context, expire=86400)

    # 真实场景中，这里会返回一个认证 Token
    return jsonify(
        {
            "code": "success",
            "msg": "Device registered successfully",
            "data": {
                "token": "your_authentication_token",
                "region": region,
                "address": address,
            },
        }
    )
