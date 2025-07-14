import hashlib
import time

from .redis_client import RedisClient
from .common_constants import REDIS_AUTH_TOKEN_PREFIX

def generate_auth_token(device_id, region):
    # 生成基础 token 字符串
    token = f"{device_id}-{region}-{int(time.time())}"

    # 使用 sha256 进行加密
    sha256_hash = hashlib.sha256(token.encode()).hexdigest()

    # 存储到 redis 中，用于后续验证
    RedisClient().set_key(REDIS_AUTH_TOKEN_PREFIX.format(sha256_hash), device_id, expire=3600)  # 1小时过期

    return sha256_hash