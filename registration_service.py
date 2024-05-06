# file: registration_service.py
from flask import Flask, request, jsonify
import threading

# 使用一个线程安全的字典来模拟数据库，在真实场景中应替换为 PostgreSQL
# DB Schema: { "client_id": {"region": "region_name", "address": "client_address"}, ... }
DEVICE_REGISTRY = {}
# 锁，用于保证多线程环境下的数据一致性
db_lock = threading.Lock()

app = Flask(__name__)


@app.route('/register', methods=['POST'])
def register_device():
    """
    对应「边缘设备注册.html」中的注册流程。
    客户端调用此接口进行注册。
    """
    data = request.get_json()
    client_id = data.get('client_id')
    region = data.get('region')
    address = data.get('address')  # 客户端连接到区域服务器的地址

    if not all([client_id, region, address]):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    with db_lock:
        DEVICE_REGISTRY[client_id] = {"region": region, "address": address}

    print(f"[Registry] 设备注册成功: Client ID={client_id}, Region={region}, Address={address}")
    print(f"[Registry] 当前注册表: {DEVICE_REGISTRY}")

    # 真实场景中，这里会返回一个认证 Token
    return jsonify({"status": "success", "message": f"Device {client_id} registered in region {region}"})


@app.route('/get_clients_by_region', methods=['GET'])
def get_clients_by_region():
    """供区域服务器查询其下有哪些客户端。"""
    region = request.args.get('region')
    if not region:
        return jsonify({"status": "error", "message": "Region parameter is required"}), 400

    with db_lock:
        clients_in_region = {
            cid: info for cid, info in DEVICE_REGISTRY.items() if info['region'] == region
        }

    return jsonify(clients_in_region)


if __name__ == '__main__':
    print("[Registry] 注册服务正在启动，监听 0.0.0.0:5000...")
    # 在生产环境中应使用 Gunicorn 或 uWSGI
    app.run(host='0.0.0.0', port=5000)
