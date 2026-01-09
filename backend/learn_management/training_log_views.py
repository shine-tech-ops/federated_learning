"""
联邦学习训练日志视图
"""
import traceback
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.permissions import AllowAny
from django.db.models import Q, Avg, Max, Min, Count
from django.utils import timezone
from datetime import timedelta

from .models import FederatedTrainingLog, FederatedTask, RegionNode
from .serializers import FederatedTrainingLogSerializer
from backend.pagination import CustomPagination
from utils.common import PassAuthenticatedPermission


class FederatedTrainingLogView(GenericAPIView):
    """训练日志查询和上传视图"""
    
    queryset = FederatedTrainingLog.objects.select_related('task', 'region_node').all()
    serializer_class = FederatedTrainingLogSerializer
    pagination_class = CustomPagination
    permission_classes = [AllowAny]
    
    def get(self, request, *args, **kwargs):
        """查询训练日志列表"""
        try:
            queryset = self.get_queryset()
            
            # 查询参数过滤
            task_id = request.query_params.get('task_id')
            region_id = request.query_params.get('region_id')
            device_id = request.query_params.get('device_id')
            round_num = request.query_params.get('round')
            phase = request.query_params.get('phase')
            level = request.query_params.get('level')
            start_time = request.query_params.get('start_time')
            end_time = request.query_params.get('end_time')
            
            if task_id:
                queryset = queryset.filter(task_id=task_id)
            if region_id:
                queryset = queryset.filter(region_node_id=region_id)
            if device_id:
                queryset = queryset.filter(device_id=device_id)
            if round_num:
                queryset = queryset.filter(round=int(round_num))
            if phase:
                queryset = queryset.filter(phase=phase)
            if level:
                queryset = queryset.filter(level=level)
            if start_time:
                queryset = queryset.filter(log_timestamp__gte=start_time)
            if end_time:
                queryset = queryset.filter(log_timestamp__lte=end_time)
            
            # 分页
            page_queryset = self.paginate_queryset(queryset)
            serializer = self.get_serializer(instance=page_queryset, many=True)
            return self.get_paginated_response(serializer.data)
            
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "查询训练日志失败",
                "data": str(e),
            }
            return Response(ret_data)
    
    def post(self, request, *args, **kwargs):
        """上传训练日志（支持单条或批量）"""
        try:
            data = request.data
            
            # 支持批量上传
            if isinstance(data, list):
                logs = []
                errors = []
                for log_data in data:
                    try:
                        serializer = self.get_serializer(data=log_data)
                        if serializer.is_valid(raise_exception=True):
                            log = serializer.save()
                            logs.append(serializer.data)
                    except Exception as e:
                        errors.append({"data": log_data, "error": str(e)})
                
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": f"成功上传 {len(logs)} 条日志",
                    "data": {
                        "success_count": len(logs),
                        "error_count": len(errors),
                        "logs": logs,
                        "errors": errors if errors else None
                    }
                }
                return Response(ret_data)
            else:
                # 单条上传
                serializer = self.get_serializer(data=data)
                if serializer.is_valid(raise_exception=True):
                    serializer.save()
                    ret_data = {
                        "code": status.HTTP_200_OK,
                        "msg": "日志上传成功",
                        "data": serializer.data,
                    }
                    return Response(ret_data)
                    
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "上传训练日志失败",
                "data": str(e),
            }
            return Response(ret_data)


class FederatedTrainingLogStatsView(GenericAPIView):
    """训练日志统计视图"""
    
    permission_classes = [PassAuthenticatedPermission]
    
    def get(self, request, *args, **kwargs):
        """获取训练日志统计信息"""
        try:
            task_id = request.query_params.get('task_id')
            if not task_id:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "缺少task_id参数",
                    "data": None
                })
            
            # 基础统计
            logs = FederatedTrainingLog.objects.filter(task_id=task_id)
            
            # 按轮次统计
            round_stats = logs.exclude(round__isnull=True).values('round').annotate(
                avg_loss=Avg('loss'),
                max_loss=Max('loss'),
                min_loss=Min('loss'),
                avg_accuracy=Avg('accuracy'),
                max_accuracy=Max('accuracy'),
                min_accuracy=Min('accuracy'),
                device_count=Count('device_id', distinct=True),
                log_count=Count('id')
            ).order_by('round')
            
            # 按设备统计
            device_stats = logs.exclude(device_id__isnull=True).values('device_id').annotate(
                avg_loss=Avg('loss'),
                avg_accuracy=Avg('accuracy'),
                log_count=Count('id'),
                latest_round=Max('round')
            ).order_by('-latest_round')
            
            # 按阶段统计
            phase_stats = logs.values('phase').annotate(
                log_count=Count('id')
            ).order_by('phase')
            
            # 按级别统计
            level_stats = logs.values('level').annotate(
                log_count=Count('id')
            ).order_by('level')
            
            # 总体统计
            total_stats = {
                'total_logs': logs.count(),
                'total_rounds': logs.exclude(round__isnull=True).aggregate(Max('round'))['round__max'] or 0,
                'unique_devices': logs.exclude(device_id__isnull=True).values('device_id').distinct().count(),
                'latest_log_time': logs.aggregate(Max('log_timestamp'))['log_timestamp__max'],
                'earliest_log_time': logs.aggregate(Min('log_timestamp'))['log_timestamp__min'],
            }
            
            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "统计信息获取成功",
                "data": {
                    "total_stats": total_stats,
                    "round_stats": list(round_stats),
                    "device_stats": list(device_stats),
                    "phase_stats": list(phase_stats),
                    "level_stats": list(level_stats),
                }
            }
            return Response(ret_data)
            
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "获取统计信息失败",
                "data": str(e),
            }
            return Response(ret_data)

