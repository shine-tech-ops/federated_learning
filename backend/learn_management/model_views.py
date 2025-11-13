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
from rest_framework.permissions import IsAuthenticated, AllowAny
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import os
import mimetypes
from loguru import logger


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
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    @swagger_auto_schema(
        operation_summary='上传模型文件',
        operation_description='上传模型文件到存储系统（MinIO 或本地文件系统），返回文件路径用于创建模型版本。支持的文件格式：.pt, .pth, .zip, .pkl, .npz',
        manual_parameters=[
            openapi.Parameter(
                'file',
                openapi.IN_FORM,
                description='模型文件（.pt, .pth, .zip, .pkl, .npz 格式）',
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
                                    description='文件路径（相对路径）',
                                    example='models/1234567890_model.pt'
                                ),
                                'file_url': openapi.Schema(
                                    type=openapi.TYPE_STRING, 
                                    description='文件访问 URL（可直接下载的完整 URL）',
                                    example='http://minio:9000/models/1234567890_model.pt?X-Amz-Algorithm=...'
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
            if file_ext not in ['.pt', '.zip', '.pth', '.pkl', '.npz']:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "不支持的文件类型，仅支持 .pt, .pth, .zip, .pkl, .npz 格式",
                    "data": None
                })
            
            # 生成文件名（固定使用 models 目录）
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            new_filename = f"{timestamp}_{file.name}"
            object_name = f"models/{new_filename}"
            
            # 根据配置选择存储方式
            file_url = None
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
                # 生成预签名 URL（有效期 7 天）
                try:
                    file_url = minio_client.get_file_url(object_name, expires=7 * 24 * 3600)
                    logger.info(f"生成 MinIO URL 成功: {file_url}")
                except Exception as e:
                    logger.warning(f"生成 MinIO URL 失败: {e}")
                    # 如果生成 URL 失败，尝试构建一个基本的访问 URL
                    try:
                        # 构建 MinIO 的基本访问 URL（需要配置）
                        minio_endpoint = getattr(settings, 'MINIO_ENDPOINT', 'localhost')
                        minio_port = getattr(settings, 'MINIO_PORT', 9000)
                        minio_use_ssl = getattr(settings, 'MINIO_USE_SSL', False)
                        protocol = 'https' if minio_use_ssl else 'http'
                        bucket_name = getattr(settings, 'MINIO_BUCKET_NAME', 'models')
                        # 注意：这不是预签名 URL，可能需要认证才能访问
                        file_url = f"{protocol}://{minio_endpoint}:{minio_port}/{bucket_name}/{object_name}"
                        logger.info(f"使用基本 MinIO URL: {file_url}")
                    except Exception as e2:
                        logger.error(f"构建基本 MinIO URL 也失败: {e2}")
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
                
                # 生成本地文件访问 URL
                media_url = getattr(settings, 'MEDIA_URL', '/media/')
                if not media_url.endswith('/'):
                    media_url += '/'
                # 构建完整的 URL（如果配置了完整域名）
                base_url = getattr(settings, 'BASE_URL', '')
                # 如果没有配置 BASE_URL，尝试从请求中获取
                if not base_url:
                    try:
                        # 从请求中获取主机信息
                        request_scheme = request.scheme if hasattr(request, 'scheme') else 'http'
                        request_host = request.get_host() if hasattr(request, 'get_host') else 'localhost:8085'
                        base_url = f"{request_scheme}://{request_host}"
                    except Exception as e:
                        logger.warning(f"无法从请求获取 base_url: {e}")
                
                if base_url:
                    file_url = f"{base_url.rstrip('/')}{media_url.rstrip('/')}/{rel_path}"
                else:
                    # 如果没有配置 BASE_URL，返回相对路径
                    file_url = f"{media_url.rstrip('/')}/{rel_path}"
                
                logger.info(f"生成本地文件 URL: {file_url}")
            
            # 确保 file_url 不为 None，如果还是 None，至少返回一个相对路径
            if file_url is None:
                logger.warning("file_url 为 None，使用 file_path 作为备用")
                file_url = rel_path

            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "上传成功",
                "data": {
                    "file_path": rel_path,
                    "file_url": file_url,  # 添加可访问的 URL
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
