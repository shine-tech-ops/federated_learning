from django.conf import settings
from rest_framework.permissions import BasePermission

import hashlib
import time
from django.conf import settings
from rest_framework.permissions import BasePermission

from .redis_client import RedisClient
from .common_constant import REDIS_AUTH_TOKEN_PREFIX

def generate_auth_token(device_id, region):
    # 生成基础 token 字符串
    token = f"{device_id}-{region}-{int(time.time())}"

    # 使用 sha256 进行加密
    sha256_hash = hashlib.sha256(token.encode()).hexdigest()

    # 存储到 redis 中，用于后续验证
    RedisClient().set_key(REDIS_AUTH_TOKEN_PREFIX.format(sha256_hash), device_id, expire=3600)  # 1小时过期

    return sha256_hash


class PassAuthenticatedPermission(BasePermission):
    """
    Pass the authenticated
    """

    def has_permission(self, request, view):
        if request.method == "OPTIONS":
            return True
        if (
            request.headers.get("authorization")
            and request.headers.get("authorization") in settings.X_API_TOKENS
        ):
            return True
        else:
            return False