import traceback
import time
import io
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.parsers import MultiPartParser, FormParser
from .models import ModelVersion, ModelInfo
from .serializers import ModelInfoSerializer, ModelVersionSerializer
from backend.pagination import CustomPagination
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import os
import mimetypes


class ModelInfoView(GenericAPIView):
    queryset = ModelInfo.objects.all()
    serializer_class = ModelInfoSerializer

    def post(self, request, *args, **kwargs):
        # 将 QueryDict 转换为可变的字典
        # 对于 FormData，request.data 是 QueryDict（不可变），需要转换为普通字典才能修改
        data = dict(request.data)
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
            # 将 QueryDict 转换为可变的字典
            # 对于 FormData，request.data 是 QueryDict（不可变），需要转换为普通字典才能修改
            # QueryDict 的值可能是列表，需要取第一个值（单值字段）
            data = {}
            for key, value in request.data.items():
                if isinstance(value, list) and len(value) > 0:
                    data[key] = value[0]  # 取第一个值
                else:
                    data[key] = value
            
            # 确保 model_info 是整数（ForeignKey 需要主键值）
            if 'model_info' in data:
                try:
                    data['model_info'] = int(data['model_info'])
                except (ValueError, TypeError):
                    pass  # 让序列化器处理错误
            
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


class ModelFileUploadView(GenericAPIView):
    """模型文件上传视图"""
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    @swagger_auto_schema(
        operation_summary='上传模型文件',
        operation_description='上传模型文件到存储系统（MinIO 或本地文件系统），返回文件路径用于创建模型版本。支持的文件格式：.pt, .pth, .zip, .pkl',
        manual_parameters=[
            openapi.Parameter(
                'file',
                openapi.IN_FORM,
                description='模型文件（.pt, .pth, .zip, .pkl 格式）',
                type=openapi.TYPE_FILE,
                required=True
            ),
        ],
        consumes=['multipart/form-data'],
        responses={
            200: openapi.Response(
                description='上传成功',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'code': openapi.Schema(type=openapi.TYPE_INTEGER, description='状态码', example=200),
                        'msg': openapi.Schema(type=openapi.TYPE_STRING, description='消息', example='上传成功'),
                        'data': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                'file_path': openapi.Schema(
                                    type=openapi.TYPE_STRING, 
                                    description='文件路径',
                                    example='models/1234567890_model.pt'
                                )
                            }
                        )
                    }
                )
            ),
            400: openapi.Response(
                description='上传失败',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'code': openapi.Schema(type=openapi.TYPE_INTEGER, example=400),
                        'msg': openapi.Schema(type=openapi.TYPE_STRING, example='未提供文件'),
                        'data': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        },
        tags=['模型管理']
    )
    def post(self, request, *args, **kwargs):
        """上传模型文件"""
        try:
            file = request.FILES.get("file")
            if not file:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "未提供文件",
                    "data": None
                })
            
            # 验证文件类型
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext not in ['.pt', '.zip', '.pth', '.pkl']:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "不支持的文件类型，仅支持 .pt, .pth, .zip, .pkl 格式",
                    "data": None
                })
            
            # 生成文件名（固定使用 models 目录）
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            new_filename = f"{timestamp}_{file.name}"
            object_name = f"models/{new_filename}"
            
            # 根据配置选择存储方式
            if settings.USE_MINIO_STORAGE:
                # 使用 MinIO 存储
                from utils.minio_client import get_minio_client
                minio_client = get_minio_client()
                # 读取文件内容
                file_data = io.BytesIO()
                for chunk in file.chunks():
                    file_data.write(chunk)
                file_data.seek(0)
                
                # 获取文件类型
                content_type = getattr(file, 'content_type', None) or 'application/octet-stream'
                
                # 上传到 MinIO
                minio_client.upload_file(
                    file_data=file_data,
                    object_name=object_name,
                    content_type=content_type
                )
                rel_path = object_name
            else:
                # 使用本地文件系统存储
                target_dir = os.path.join(settings.MEDIA_ROOT, "models")
                os.makedirs(target_dir, exist_ok=True)
                abs_path = os.path.join(target_dir, new_filename)
                rel_path = f"models/{new_filename}"
                # 流式写入，避免大文件占用内存
                with open(abs_path, 'wb') as dst:
                    for chunk in file.chunks():
                        dst.write(chunk)

            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "上传成功",
                "data": {
                    "file_path": rel_path,
                },
            }
            return Response(ret_data)
        except Exception as e:
            traceback.print_exc()
            ret_data = {
                "code": status.HTTP_400_BAD_REQUEST,
                "msg": f"上传失败: {str(e)}",
                "data": str(e),
            }
            return Response(ret_data)


class ModelVersionDownloadView(GenericAPIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, id, *args, **kwargs):
        instance = get_object_or_404(ModelVersion, id=id)
        rel_path = str(instance.model_file or '').lstrip('/')
        
        if not rel_path:
            return Response({
                "code": status.HTTP_404_NOT_FOUND,
                "msg": "文件路径为空",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)

        try:
            if settings.USE_MINIO_STORAGE:
                # 使用 MinIO 下载
                from utils.minio_client import get_minio_client
                minio_client = get_minio_client()
                
                # 检查文件是否存在
                if not minio_client.file_exists(rel_path):
                    return Response({
                        "code": status.HTTP_404_NOT_FOUND,
                        "msg": "文件不存在",
                        "data": None
                    }, status=status.HTTP_404_NOT_FOUND)
                
                # 获取文件流
                file_stream = minio_client.get_file_stream(rel_path)
                filename = os.path.basename(rel_path)
                content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                
                response = FileResponse(file_stream, content_type=content_type)
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                return response
            else:
                # 使用本地文件系统下载（原有逻辑）
                media_root = os.path.abspath(settings.MEDIA_ROOT)
                abs_path = os.path.abspath(os.path.join(media_root, rel_path))

                if not abs_path.startswith(media_root) or not os.path.exists(abs_path):
                    return Response({
                        "code": status.HTTP_404_NOT_FOUND,
                        "msg": "文件不存在",
                        "data": None
                    }, status=status.HTTP_404_NOT_FOUND)

                filename = os.path.basename(abs_path)
                content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                response = FileResponse(open(abs_path, 'rb'), content_type=content_type)
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                return response
        except Exception as e:
            traceback.print_exc()
            return Response({
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "msg": f"下载失败: {str(e)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
