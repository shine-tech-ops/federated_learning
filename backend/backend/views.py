import os
import traceback
import time

from django.core.files.storage import default_storage
from rest_framework.generics import GenericAPIView
from django.core.files.base import ContentFile
from rest_framework.response import Response
from rest_framework import status
from rest_framework.serializers import Serializer


class UploadFileSerializer(Serializer):
    pass


class UploadFileView(GenericAPIView):
    serializer_class = UploadFileSerializer  # 添加序列化器类
    def post(self, request, *args, **kwargs):
        try:
            file = request.FILES.get("file")
            upload_to_path = request.data.get("upload_to_path", "uploads")  # 设置默认路径更安全
            if not file:
                return Response({
                    "code": status.HTTP_400_BAD_REQUEST,
                    "msg": "未提供文件",
                    "data": None
                })
            # 以 MEDIA_ROOT 为根目录，使用相对子目录
            from django.conf import settings
            safe_subdir = str(upload_to_path).strip().strip('/').replace('..', '') or 'uploads'
            target_dir = os.path.join(settings.MEDIA_ROOT, safe_subdir)
            os.makedirs(target_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            new_filename = f"{timestamp}_{file.name}"
            abs_path = os.path.join(target_dir, new_filename)
            rel_path = f"{safe_subdir}/{new_filename}"
            # 流式写入，避免大文件占用内存
            with open(abs_path, 'wb') as dst:
                for chunk in file.chunks():
                    dst.write(chunk)

            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "创建成功",
                "data": {
                    # 返回相对 MEDIA_ROOT 的路径
                    "file_path": rel_path,
                },
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
