from django.conf import settings
from rest_framework.permissions import BasePermission
from rest_framework_simplejwt.authentication import JWTAuthentication

import hashlib
import time

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
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")

        # 优先尝试 Bearer/JWT
        if auth_header and auth_header.startswith("Bearer "):
            jwt_auth = JWTAuthentication()
            try:
                user_auth = jwt_auth.authenticate(request)
                if user_auth:
                    # DRF 会把 user、auth 挂在 request 上，后续视图可直接使用
                    request.user, request.auth = user_auth
                    return True
            except Exception:
                # JWT 解析失败则继续尝试静态 token
                pass

        # 兼容原先的静态白名单 token
        if auth_header and auth_header in settings.X_API_TOKENS:
            return True

        return False