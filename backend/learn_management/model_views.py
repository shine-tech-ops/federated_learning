import traceback
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import GenericAPIView
from .models import ModelVersion, ModelInfo
from .serializers import ModelInfoSerializer, ModelVersionSerializer
from backend.pagination import CustomPagination


class ModelInfoView(GenericAPIView):
    queryset = ModelInfo.objects.all()
    serializer_class = ModelInfoSerializer

    def post(self, request, *args, **kwargs):
        data = request.data
        data["created_by"] = request.user.id
        serializer = self.get_serializer(data=data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(
                {"code": status.HTTP_200_OK, "msg": "创建成功", "data": serializer.data}
            )
        return Response(
            {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "创建失败",
                "data": serializer.errors,
            }
        )

    def get(self, request, *args, **kwargs):
        page = self.paginate_queryset(self.get_queryset())
        serializer = self.get_serializer(page, many=True)
        return self.get_paginated_response(serializer.data)

    def put(self, request, *args, **kwargs):
        try:
            version_id = request.data.get("id")
            instance = self.queryset.get(id=version_id)
            serializer = self.get_serializer(instance, data=request.data, partial=True)
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
            instance = self.queryset.get(id=request.data.get("id"))
            instance.delete()
            return Response({"code": status.HTTP_200_OK, "msg": "删除成功", "data": {}})
        except Exception as e:
            return Response(
                {"code": status.HTTP_400_BAD_REQUEST, "msg": "删除失败", "data": str(e)}
            )


class ModelVersionView(GenericAPIView):

    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
    pagination_class = CustomPagination

    def post(self, request, *args, **kwargs):
        try:
            data = request.data
            data["created_by"] = request.user.id
            serializer = self.get_serializer(data=data)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response(
                    {
                        "code": status.HTTP_200_OK,
                        "msg": "版本创建成功",
                        "data": serializer.data,
                    }
                )
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"code": status.HTTP_400_BAD_REQUEST, "msg": "版本创建失败", "data": str(e)}
            )

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        if request.query_params.get("model_id"):
            queryset = queryset.filter(model_info_id=request.query_params.get("model_id"))
        page = self.paginate_queryset(queryset)
        serializer = self.get_serializer(page, many=True)
        return self.get_paginated_response(serializer.data)

    def put(self, request, *args, **kwargs):
        try:
            instance = self.queryset.get(id=request.data.get("id"))
            serializer = self.get_serializer(instance, data=request.data, partial=True)
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                return Response(
                    {
                        "code": status.HTTP_200_OK,
                        "msg": "更新成功",
                        "data": serializer.data,
                    }
                )
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"code": status.HTTP_400_BAD_REQUEST, "msg": "更新失败", "data": str(e)}
            )

    def delete(self, request, *args, **kwargs):
        try:
            instance = self.queryset.get(id=request.data.get("id"))
            if instance.is_deployed:
                return Response({
                    "code": 400,
                    "msg": "上线版本不可删除"
                })
            instance.delete()
            return Response({"code": status.HTTP_200_OK, "msg": "删除成功", "data": {}})
        except Exception as e:
            traceback.print_exc()
            return Response(
                {"code": status.HTTP_400_BAD_REQUEST, "msg": "删除失败", "data": str(e)}
            )


class ModelRollbackView(GenericAPIView):
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer

    def post(self, request, *args, **kwargs):
        try:
            target_id = request.data.get("id")
            target = self.queryset.get(id=target_id)

            # 回滚：将所有部署模型设为非部署状态
            ModelVersion.objects.filter(is_deployed=True).update(
                is_deployed=False, status="archived"
            )

            # 设置目标模型为部署状态
            target.is_deployed = True
            target.status = "active"
            target.save()

            return Response(
                {
                    "code": status.HTTP_200_OK,
                    "msg": "回滚成功",
                    "data": ModelVersionSerializer(target).data,
                }
            )
        except Exception as e:
            traceback.print_exc()
            return Response(
                {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "msg": "回滚失败",
                    "data": str(e),
                }
            )



class ModelVersionDeployView(GenericAPIView):
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer

    def post(self, request, *args, **kwargs):
        try:
            version_id = request.data.get("id")
            is_deployed = request.data.get("is_deployed", False)
            instance = self.queryset.get(id=version_id)
            instance.is_deployed = is_deployed
            instance.save(update_fields=["is_deployed"])
            return Response({
                "code": status.HTTP_200_OK,
                "msg": f"{'上线' if is_deployed else '下线'}成功",
                "data": ModelVersionSerializer(instance).data
            })
        except Exception as e:
            traceback.print_exc()
            return Response({
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": "操作失败",
                "data": str(e)
            })
