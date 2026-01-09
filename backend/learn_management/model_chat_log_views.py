import traceback
from django.db.models import Q
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from backend.pagination import CustomPagination
from utils.common import PassAuthenticatedPermission
from .models import ModelChatLog
from .serializers import ModelChatLogSerializer


class ModelChatLogView(GenericAPIView):
    """模型对话日志：上传与查询"""

    queryset = ModelChatLog.objects.select_related('model_info', 'model_version').all()
    serializer_class = ModelChatLogSerializer
    pagination_class = CustomPagination
    permission_classes = [PassAuthenticatedPermission]

    def get(self, request, *args, **kwargs):
        try:
            queryset = self.get_queryset()

            model_id = request.query_params.get('model_id')
            model_version_id = request.query_params.get('model_version_id')
            keyword = request.query_params.get('keyword')
            start_time = request.query_params.get('start_time')
            end_time = request.query_params.get('end_time')

            if model_id:
                queryset = queryset.filter(model_info_id=model_id)
            if model_version_id:
                queryset = queryset.filter(model_version_id=model_version_id)
            if keyword:
                queryset = queryset.filter(
                    Q(input_text__icontains=keyword) | Q(output_text__icontains=keyword)
                )
            if start_time:
                queryset = queryset.filter(created_at__gte=start_time)
            if end_time:
                queryset = queryset.filter(created_at__lte=end_time)

            queryset = queryset.order_by('-created_at', '-id')

            page_queryset = self.paginate_queryset(queryset)
            serializer = self.get_serializer(page_queryset, many=True)
            return self.get_paginated_response(serializer.data)
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"code": status.HTTP_400_BAD_REQUEST, "msg": "查询模型对话日志失败", "data": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            user_obj = getattr(request, "user", None)
            created_by = user_obj if getattr(user_obj, "is_authenticated", False) else None

            if isinstance(data, list):
                logs, errors = [], []
                for item in data:
                    try:
                        serializer = self.get_serializer(data=item)
                        serializer.is_valid(raise_exception=True)
                        serializer.save(created_by=created_by)
                        logs.append(serializer.data)
                    except Exception as exc:
                        errors.append({"data": item, "error": str(exc)})

                return Response(
                    {
                        "code": status.HTTP_200_OK,
                        "msg": f"成功上传 {len(logs)} 条日志",
                        "data": {"success_count": len(logs), "error_count": len(errors), "logs": logs, "errors": errors or None},
                    }
                )

            serializer = self.get_serializer(data=data)
            serializer.is_valid(raise_exception=True)
            serializer.save(created_by=created_by)
            return Response(
                {"code": status.HTTP_200_OK, "msg": "日志上传成功", "data": serializer.data}
            )
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"code": status.HTTP_400_BAD_REQUEST, "msg": "上传模型对话日志失败", "data": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )


