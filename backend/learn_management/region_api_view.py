import traceback
from rest_framework.response import Response
from rest_framework import status
from .models import TrainingRecord, FederatedTask, EdgeNode, RegionNode
from .serializers import TrainingRecordSerializer, FederatedTaskSerializer, EdgeNodeSerializer, RegionNodeSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView
from utils.common import PassAuthenticatedPermission
from utils.rabbitmq_client import RabbitMQClient
from utils.common import generate_auth_token
import utils.common_constant as consts
import conf.env as env


class DeviceRegisterView(GenericAPIView):
    queryset = EdgeNode.objects.all()
    serializer_class = EdgeNodeSerializer
    pagination_class = CustomPagination

    permission_classes = [PassAuthenticatedPermission]

    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            data["created_by"] = request.user.id
            serializer = self.get_serializer(data=data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                auth_token = generate_auth_token(serializer.data["device_id"], serializer.data["region_node"])
                ret = serializer.data
                ret["mqtt_config"] = {
                    "host": env.MQTT_BROKER_HOST,
                    "port": env.MQTT_BROKER_PORT,
                    "username": env.MQTT_USER,
                    "password": env.MQTT_PASSWORD,
                }
                ret["token"] = auth_token
                # 发送注册消息到mq
                RabbitMQClient().publisher(
                    consts.MQ_DEVICE_REG_EXCHANGE,
                    ret,
                )
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "边缘节点创建成功",
                    "data": ret,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "边缘节点创建失败",
                "data": str(e),
            }
            return Response(ret_data)

class RegionTaskView(GenericAPIView):
    queryset = FederatedTask.objects.select_related(
    'region_node', 'model_info', 'model_version'
    )
    serializer_class = FederatedTaskSerializer
    pagination_class = CustomPagination

    permission_classes = [PassAuthenticatedPermission]

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        region_id = request.query_params.get("region_id")
        task_id = request.query_params.get("task_id")

        if region_id:
            queryset = queryset.filter(region_node_id=region_id)
        if task_id:
            queryset = queryset.filter(id=task_id)
        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(instance=page_queryset, many=True)
        return self.get_paginated_response(serializer.data)

class DeviceTaskView(GenericAPIView):
    queryset = TrainingRecord.objects.select_related(
        'edge_node', 'region_node', 'model_info', 'model_version'
    ).all()
    serializer_class = TrainingRecordSerializer
    pagination_class = CustomPagination

    permission_classes = [PassAuthenticatedPermission]

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        region_id = request.query_params.get("region_id")
        device_id = request.query_params.get("device_id")
        task_id = request.query_params.get("task_id")

        if region_id:
            queryset = queryset.filter(region_node_id=region_id)
        if device_id:
            queryset = queryset.filter(device_id=device_id)
        if task_id:
            queryset = queryset.filter(federated_task_id=task_id)
        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(instance=page_queryset, many=True)
        return self.get_paginated_response(serializer.data)


    def post(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "设备训练记录创建成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "设备训练记录创建失败",
                "data": str(e),
            }
            return Response(ret_data)

    def put(self, request, *args, **kwargs):
        try:
            node_id = request.data.get("id")
            node = self.queryset.get(id=node_id)
            serializer = self.get_serializer(node, data=request.data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "设备训练记录更新成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "设备训练记录更新失败",
                "data": str(e),
            }
            return Response(ret_data)

    def delete(self, request, *args, **kwargs):
        try:
            node_id = request.data.get("id")
            node = self.queryset.get(id=node_id)
            node.delete()
            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "设备训练记录删除成功",
                "data": {},
            }
            return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "设备训练记录删除失败",
                "data": str(e),
            }
            return Response(ret_data)

class RegionNodeListView(GenericAPIView):
    queryset = RegionNode.objects.all()
    serializer_class = RegionNodeSerializer
    pagination_class = CustomPagination

    permission_classes = [PassAuthenticatedPermission]

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        if request.query_params.get("id"):
            queryset = queryset.filter(id=request.query_params.get("id"))
        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(instance=page_queryset, many=True)
        return self.get_paginated_response(serializer.data)