import traceback
import json
from datetime import datetime
from loguru import logger

from rest_framework.response import Response
from rest_framework import status, serializers
from .models import FederatedTask, ModelInfo, ModelVersion, RegionNode, SystemConfig
from .serializers import FederatedTaskSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView
from utils.rabbitmq_client import RabbitMQClient

class TaskStartSerializer(serializers.Serializer):
    id = serializers.IntegerField(help_text='联邦学习任务ID')


def get_system_config():
    sysconfig = SystemConfig.objects.filter(is_active=True).first()
    if not sysconfig:
       raise Exception("请先配置系统参数")
    config_data = sysconfig.config_data
    federated = config_data.get("federated")
    if not federated:
        raise Exception("请先配置联邦学习参数")
    return federated.get("rounds", 10), federated.get("aggregation", "fedavg"), federated.get("participationRate", 50)
class FederatedTaskView(GenericAPIView):

    queryset = FederatedTask.objects.select_related(
    'region_node', 'model_info', 'model_version', 'created_by'
    )
    serializer_class = FederatedTaskSerializer
    pagination_class = CustomPagination

    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            data["created_by"] = request.user.id
            rounds, aggregation_method, participation_rate = get_system_config()
            data["rounds"] = rounds
            data["aggregation_method"] = aggregation_method
            data["participation_rate"] = participation_rate
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
            data = request.data
            rounds, aggregation_method, participation_rate = get_system_config()
            data["rounds"] = rounds
            data["aggregation_method"] = aggregation_method
            data["participation_rate"] = participation_rate

            serializer = self.get_serializer(task, data=data, partial=True)
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




class FederatedTaskStartView(GenericAPIView):
    """
    联邦学习任务启动视图
    启动任务并将任务信息发送到 RabbitMQ 供 region node 消费
    """
    queryset = FederatedTask.objects.select_related(
        'region_node', 'model_info', 'model_version', 'created_by'
    )
    serializer_class = TaskStartSerializer

    def post(self, request, *args, **kwargs):
        """
        启动联邦学习任务
        1. 验证任务ID
        2. 检查任务状态
        3. 更新任务状态为 running
        4. 发送任务信息到 RabbitMQ
        """
        try:
            serializer = self.get_serializer(data=request.data)
            if not serializer.is_valid():
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "参数验证失败",
                    "data": serializer.errors,
                })

            task_id = serializer.validated_data['id']
            
            # 获取任务对象
            try:
                task = self.queryset.get(id=task_id)
            except FederatedTask.DoesNotExist:
                return Response({
                    "code": status.HTTP_404_NOT_FOUND,
                    "msg": "任务不存在",
                    "data": {},
                })

            # 检查任务状态
            if task.status not in ['pending', 'paused']:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": f"任务状态错误，当前状态: {task.get_status_display()}",
                    "data": {"current_status": task.status},
                })

            # 检查必要的关联对象
            if not task.region_node:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "任务未关联区域节点",
                    "data": {},
                })

            if not task.model_info or not task.model_version:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "任务未关联模型信息或模型版本",
                    "data": {},
                })

            # 更新任务状态
            task.status = 'running'
            task.updated_at = datetime.now()
            task.save()

            # 获取该区域节点的所有边缘设备
            edge_nodes = task.region_node.edge_nodes.all()
            edge_devices = []
            for edge_node in edge_nodes:
                edge_devices.append({
                    "device_id": edge_node.device_id,
                    "ip_address": edge_node.ip_address,
                    "device_context": edge_node.device_context,
                    "status": edge_node.status,
                    "last_heartbeat": edge_node.last_heartbeat.isoformat() if edge_node.last_heartbeat else None,
                    "description": edge_node.description,
                })

            # 准备发送到 RabbitMQ 的任务数据
            task_data = {
                "task_id": task.id,
                "task_name": task.name,
                "description": task.description,
                "rounds": task.rounds,
                "aggregation_method": task.aggregation_method,
                "participation_rate": task.participation_rate,
                "status": task.status,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "region_node": {
                    "id": task.region_node.id,
                    "name": task.region_node.name,
                    "ip_address": task.region_node.ip_address,
                    "description": task.region_node.description,
                },
                "edge_devices": edge_devices,  # 添加边缘设备信息
                "model_info": {
                    "id": task.model_info.id,
                    "name": task.model_info.name,
                    "description": task.model_info.description,
                },
                "model_version": {
                    "id": task.model_version.id,
                    "version": task.model_version.version,
                    "model_file": task.model_version.model_file,
                    "description": task.model_version.description,
                    "accuracy": task.model_version.accuracy,
                    "loss": task.model_version.loss,
                    "metrics": task.model_version.metrics,
                },
                "created_by": {
                    "id": task.created_by.id,
                    "name": task.created_by.name,
                },
                "message_type": "federated_task_start",
                "timestamp": datetime.now().isoformat(),
            }

            # 发送到 RabbitMQ
            try:
                rabbitmq_client = RabbitMQClient()
                # 使用区域节点ID作为Exchange名称
                exchange_name = f"federated_task_region_{task.region_node.id}"
                logger.info(f"发送到 Exchange: {exchange_name}")
                rabbitmq_client.publisher(exchange_name, task_data)
                
                return Response({
                    "code": status.HTTP_200_OK,
                    "msg": "任务启动成功，已发送到区域节点",
                    "data": {
                        "task_id": task.id,
                        "task_name": task.name,
                        "status": task.status,
                        "region_node": task.region_node.name,
                        "exchange_name": exchange_name,
                    },
                })
                
            except Exception as mq_error:
                # 如果 RabbitMQ 发送失败，回滚任务状态
                task.status = 'pending'
                task.save()
                
                return Response({
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "msg": f"任务启动失败，RabbitMQ 发送错误: {str(mq_error)}",
                    "data": {},
                })

        except Exception as e:
            traceback.print_exc()
            return Response({
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "msg": "任务启动失败",
                "data": str(e),
            })
   