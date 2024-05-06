
# views/system_config_views.py
import traceback
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import GenericAPIView
from learn_management.models import SystemConfig
from learn_management.serializers import SystemConfigSerializer
from backend.pagination import CustomPagination
import utils.common_constant as const


class SystemConfigView(GenericAPIView):
    queryset = SystemConfig.objects.all()
    serializer_class = SystemConfigSerializer
    pagination_class = CustomPagination

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        if request.query_params.get("id"):
            queryset = queryset.filter(id=request.query_params.get("id"))

        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(page_queryset, many=True)
        return self.get_paginated_response(serializer.data)

    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            data["created_by"] = request.user.id
            serializer = self.get_serializer(data=data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response({
                    "code": status.HTTP_200_OK,
                    "msg": "创建成功",
                    "data": serializer.data
                })
        except Exception as e:
            traceback.print_exc()
            return Response({
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "创建失败",
                "data": str(e)
            })

    def put(self, request, *args, **kwargs):
        try:
            config_id = request.data.get("id")
            config = self.queryset.get(id=config_id)
            serializer = self.get_serializer(config, data=request.data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response({
                    "code": status.HTTP_200_OK,
                    "msg": "更新成功",
                    "data": serializer.data
                })
        except Exception as e:
            traceback.print_exc()
            return Response({
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "更新失败",
                "data": str(e)
            })

    def delete(self, request, *args, **kwargs):
        try:
            config_id = request.data.get("id")
            config = self.queryset.get(id=config_id)

            if config.is_active:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "不能删除已激活的配置",
                    "data": {}
                })

            config.delete()
            return Response({
                "code": status.HTTP_200_OK,
                "msg": "删除成功",
                "data": {}
            })
        except Exception as e:
            return Response({
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "删除失败",
                "data": str(e)
            })

class SystemConfigActivateView(GenericAPIView):
    queryset = SystemConfig.objects.all()
    serializer_class = SystemConfigSerializer

    def post(self, request, *args, **kwargs):
        try:
            config_id = request.data.get("id")
            if not config_id:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "缺少配置ID",
                    "data": {}
                })

            # 获取当前要激活的配置
            new_config = self.queryset.get(id=config_id)

            # 将所有配置设为非激活
            self.queryset.update(is_active=False)

            # 激活目标配置
            new_config.is_active = True
            new_config.save()

            return Response({
                "code": status.HTTP_200_OK,
                "msg": "激活成功",
                "data": self.get_serializer(new_config).data
            })
        except Exception as e:
            traceback.print_exc()
            return Response({
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "激活失败",
                "data": str(e),
            })

class AggregationMethodView(GenericAPIView):
    def get(self, request, *args, **kwargs):

       return Response({
           "code": status.HTTP_200_OK,
           "msg": "获取成功",
           "data": [key for key,_ in const.AGGREGATION_STRATEGIES_MAP.items()],
       })
