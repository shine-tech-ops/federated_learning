
# region_node_view.py

import traceback
from rest_framework.response import Response
from rest_framework import status
from .models import OperationLog
from .serializers import OperationLogSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView


class SystemLogView(GenericAPIView):
    queryset = OperationLog.objects.all()
    serializer_class = OperationLogSerializer
    pagination_class = CustomPagination

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        if request.query_params.get("user"):
            queryset = queryset.filter(user__name__contains=request.query_params.get("user"))
        if request.query_params.get("method"):
            queryset = queryset.filter(method=request.query_params.get("method"))
        if request.query_params.get("path"):
            queryset = queryset.filter(path__contains=request.query_params.get("path"))
        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(instance=page_queryset, many=True)
        return self.get_paginated_response(serializer.data)

    def delete(self, request, *args, **kwargs):
        try:
            # 获取要删除的日志 ID 列表
            log_ids = request.data.get("ids")  # 支持批量删除
            if log_ids and isinstance(log_ids, list):
                # 批量删除
                self.queryset.filter(id__in=log_ids).delete()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": f"系统日志批量删除成功（{len(log_ids)}条）",
                    "data": {}
                }
            else:
                # 全部删除
                self.queryset.all().delete()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "系统日志已全部清空",
                    "data": {}
                }
            return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "系统日志删除失败",
                "data": str(e),
            }
            return Response(ret_data)
