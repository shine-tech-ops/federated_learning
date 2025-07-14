# region_node_view.py

import traceback
from rest_framework.response import Response
from rest_framework import status
from .models import RegionNode
from .serializers import RegionNodeSerializer
from backend.pagination import CustomPagination
from rest_framework.generics import GenericAPIView


class RegionNodeView(GenericAPIView):
    queryset = RegionNode.objects.all()
    serializer_class = RegionNodeSerializer
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
                    "msg": "区域节点创建成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "区域节点创建失败",
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
            node_id = request.data.get("id")
            node = self.queryset.get(id=node_id)
            serializer = self.get_serializer(node, data=request.data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "区域节点更新成功",
                    "data": serializer.data,
                }
                return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "区域节点更新失败",
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
                "msg": "区域节点删除成功",
                "data": {},
            }
            return Response(ret_data)
        except Exception as e:
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "区域节点删除失败",
                "data": str(e),
            }
            return Response(ret_data)
