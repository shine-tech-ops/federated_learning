import traceback

from rest_framework.response import Response
from rest_framework import status, serializers
from .models import FederatedTask, ModelInfo, ModelVersion, RegionNode, SystemConfig
from .serializers import FederatedTaskSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView
from .network.services import NetworkConfigService

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
    serializer_class = TaskStartSerializer
    network_config_service = NetworkConfigService()
    
    def post(self, request, *args, **kwargs):
        try:
            # 直接使用请求中的任务数据
            task_data = request.data
            
            # 获取区域节点配置
            region_id = task_data.get('region_node_detail').get('id')
            region_config = self.network_config_service.get_region_routes(region_id)
            if not region_config:
                raise Exception("区域节点配置不存在")
           
            # 检查任务状态
            if task_data['status'] in ["pending", "paused"]:
                # 准备任务消息
                task_message = task_data
                task_message['region_config'] = region_config
                
                # TODO:调用region服务器的接口，启动任务
                
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "开始成功",
                    "task_message": task_message,
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
                
        except FederatedTask.DoesNotExist:
            return Response({
                "code": status.HTTP_404_NOT_FOUND,
                "msg": "任务不存在",
                "data": {},
            })
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "开始失败",
                "data": str(e),
            }
            return Response(ret_data)

