# middleware/log_middleware.py
import json
from django.utils import timezone
from django.http import QueryDict
from django.db import models
from django.utils.deprecation import MiddlewareMixin
from learn_management.models import OperationLog
from loguru import logger


class OperationLogMiddleware:
    """
    使用 __call__ 实现的中间件，记录所有 API 请求日志
    """

    # 配置不需要记录日志的 URL 前缀
    FILTER_URLS = [
        '/api/auth/',
        '/api/docs/',
        '/api/schema/',
        '/api/health/',
        '/api/static/',
        '/api/admin/',
    ]

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_method = request.method
        # 判断是否需要记录日志
        if self._should_skip_logging(request.path, request_method):
            return self.get_response(request)

        # 保存请求信息
        self._save_request_info(request)

        # 继续处理请求
        response = self.get_response(request)

        # 记录日志
        self._log_request(request, response)

        return response

    def _should_skip_logging(self, path , request_method):
        """
        判断当前请求是否应该跳过日志记录
        """
        # 跳过特定 URL 前缀
        if any(path.startswith(prefix) for prefix in self.FILTER_URLS):
            return True

        # 跳过 GET 请求
        if request_method in ['GET', 'get']:
            return True

        return False

    def _save_request_info(self, request):
        """
        保存请求体或查询参数，便于后续记录日志
        """
        request._start_time = timezone.now()

        # 保存原始请求参数（用于日志记录）
        if request.method in ['POST', 'PUT']:
            content_type = request.META.get('CONTENT_TYPE', '')
            if 'application/json' in content_type:
                try:
                    request._body_data = json.loads(request.body.decode('utf-8'))
                except Exception:
                    request._body_data = {}
            elif 'application/x-www-form-urlencoded' in content_type:
                request._body_data = dict(request.POST)
            else:
                request._body_data = {}
        elif request.method in ['GET', 'DELETE']:
            # GET 和 DELETE 的参数在 query_params 中
            request._body_data = dict(request.GET)
        else:
            request._body_data = {}

    def _log_request(self, request, response):
        """
        实际记录日志的逻辑
        """
        try:
            # 获取用户
            user = request.user if request.user.is_authenticated else None

            # 获取请求信息
            ip = request.META.get('REMOTE_ADDR')
            method = request.method
            path = request.path

            # 获取 body（兼容各种请求方式）
            if hasattr(request, '_body_data'):
                if method in ['GET', 'DELETE']:
                    # GET/DELETE 参数记录为查询参数
                    body = json.dumps({
                        'query_params': request._body_data
                    })
                else:
                    # POST/PUT 记录为 body
                    body = json.dumps({
                        'body': request._body_data
                    })
            else:
                try:
                    body = request.body.decode('utf-8') if request.body else ''
                except Exception:
                    body = ''

            # 获取响应信息
            response_code = response.status_code
            try:
                response_body = response.content.decode('utf-8') if response.content else ''
            except Exception:
                response_body = ''

            # 创建日志记录
            OperationLog.objects.create(
                user=user,
                ip=ip,
                method=method,
                path=path,
                body=body,
                response_code=response_code,
                response_body=response_body
            )
        except Exception as e:
            logger.error("Failed to save operation log: %s", str(e), exc_info=True)

