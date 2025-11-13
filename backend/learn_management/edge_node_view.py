
# edge_node_view.py

import traceback
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .models import EdgeNode
from .serializers import EdgeNodeSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView
from utils.common import PassAuthenticatedPermission


class EdgeNodeView(GenericAPIView):
    queryset = EdgeNode.objects.all()
    serializer_class = EdgeNodeSerializer
    pagination_class = CustomPagination

    def post(self, request, *args, **kwargs):
        try:
            data = request.data.copy()
            data["created_by"] = request.user.id
            
            # 处理 region_node 字段（前端可能传 region_id 或 region_node）
            if "region_id" in data and "region_node" not in data:
                data["region_node"] = data.pop("region_id")
            
            # device_id 必须提供，如果没有则生成一个唯一ID
            if not data.get("device_id"):
                import uuid
                data["device_id"] = f"device_{uuid.uuid4().hex[:8]}"
            
            serializer = self.get_serializer(data=data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "边缘节点创建成功",
                    "data": serializer.data,
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

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        if request.query_params.get("id"):
            queryset = queryset.filter(id=request.query_params.get("id"))
        # 支持 region_id 和 region_node 两种查询方式
        region_id = request.query_params.get("region_id") or request.query_params.get("region_node")
        if region_id:
            queryset = queryset.filter(region_node_id=region_id)
        if request.query_params.get("status"):
            queryset = queryset.filter(status=request.query_params.get("status"))
        if request.query_params.get("device_id"):
            queryset = queryset.filter(device_id=request.query_params.get("device_id"))
        page_queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(instance=page_queryset, many=True)
        return self.get_paginated_response(serializer.data)

    def put(self, request, *args, **kwargs):
        try:
            node_id = request.data.get("id")
            node = self.queryset.get(id=node_id)
            data = request.data.copy()
            
            # 处理 region_node 字段（前端可能传 region_id 或 region_node）
            if "region_id" in data and "region_node" not in data:
                data["region_node"] = data.pop("region_id")
            
            serializer = self.get_serializer(node, data=data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "边缘节点更新成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "边缘节点更新失败",
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
                "msg": "边缘节点删除成功",
                "data": {},
            }
            return Response(ret_data)
        except Exception as e:
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "边缘节点删除失败",
                "data": str(e),
            }
            return Response(ret_data)


class EdgeNodeHeartbeatView(GenericAPIView):
    queryset = EdgeNode.objects.all()
    serializer_class = EdgeNodeSerializer
    pagination_class = CustomPagination
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        """接收边缘设备心跳"""
        try:
            data = request.data
            device_id = data.get("device_id")
            region_id = data.get("region_id") or data.get("region_node")
            device_context = data.get("device_context", {})
            
            if not device_id:
                ret_data = {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "设备心跳失败：缺少device_id",
                    "data": None,
                }
                return Response(ret_data)
            
            if not region_id:
                ret_data = {
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "设备心跳失败：缺少region_id",
                    "data": None,
                }
                return Response(ret_data)
            
            # 更新或创建设备心跳记录
            from django.utils import timezone
            node, created = EdgeNode.objects.update_or_create(
                device_id=device_id,
                region_node_id=region_id,
                defaults={
                    "last_heartbeat": timezone.now(),
                    "status": "online",
                    "device_context": device_context
                }
            )
            
            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "设备心跳成功" if not created else "设备心跳成功（新设备已注册）",
                "data": {
                    "device_id": device_id,
                    "region_id": region_id,
                    "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
                    "status": node.status
                },
            }
            return Response(ret_data)
            
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "msg": f"设备心跳失败：{str(e)}",
                "data": None,
            }
            return Response(ret_data)