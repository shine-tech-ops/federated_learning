import traceback

from rest_framework.response import Response
from rest_framework import status
from .models import FederatedTask
from .serializers import FederatedTaskSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView


class FederatedTaskView(GenericAPIView):

    queryset = FederatedTask.objects.all()
    serializer_class = FederatedTaskSerializer
    pagination_class = CustomPagination

    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            data["created_by"] = request.user.id
            serializer = self.get_serializer(data=data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "创建成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "创建失败",
                "data": str(e),
            }
            return Response(ret_data)

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        if request.query_params.get("id"):
            queryset = queryset.filter(id=request.query_params.get("id"))
        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(instance=page_queryset, many=True)
        return self.get_paginated_response(serializer.data)

    def put(self, request, *args, **kwargs):
        try:
            task_id = request.data.get("id")
            task = self.queryset.get(id=task_id)
            serializer = self.get_serializer(task, data=request.data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "更新成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "更新失败",
                "data": str(e),
            }
            return Response(ret_data)

    def delete(self, request, *args, **kwargs):
        try:
            task_id = request.data.get("id")
            task = self.queryset.get(id=task_id)
            task.delete()
            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "删除成功",
                "data": {},
            }
            return Response(ret_data)
        except Exception as e:
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "删除失败",
                "data": str(e),
            }
            return Response(ret_data)

class FederatedTaskPauseView(GenericAPIView):
    queryset = FederatedTask.objects.all()
    serializer_class = FederatedTaskSerializer

    def put(self, request, *args, **kwargs):
        try:
            task = self.queryset.get(id=request.data.get("id"))
            if task.status == "running":
                task.status = "paused"
                task.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "暂停成功",
                    "data": {},
                }
                return Response(ret_data)
            else:
                ret_data = {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "任务状态错误",
                    "data": {},
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "暂停失败",
                "data": str(e),
            }
            return Response(ret_data)


class FederatedTaskResumeView(GenericAPIView):
    queryset = FederatedTask.objects.all()
    serializer_class = FederatedTaskSerializer

    def put(self, request, *args, **kwargs):
        try:
            task = self.queryset.get(id=request.data.get("id"))
            if task.status == "paused":
                task.status = "running"
                task.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "恢复成功",
                    "data": {},
                }
                return Response(ret_data)
            else:
                ret_data = {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "任务状态错误",
                    "data": {},
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "msg": "恢复失败",
                "data": str(e),
            }
            return Response(ret_data)
