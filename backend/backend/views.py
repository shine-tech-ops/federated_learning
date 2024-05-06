import os
import traceback
import time

from django.core.files.storage import default_storage
from rest_framework.generics import GenericAPIView
from django.core.files.base import ContentFile
from rest_framework.response import Response
from rest_framework import status


class UploadFileView(GenericAPIView):
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
            upload_to_path = f"/code/upload/{upload_to_path}"
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            new_filename = f"{timestamp}_{file.name}"
            path = os.path.join(upload_to_path, new_filename)
            # 确保 ContentFile 内容为 bytes 类型
            content = ContentFile(file.read()) if isinstance(file.read(), bytes) else ContentFile(file.read().encode('utf-8'))
            file_path = default_storage.save(path, content)

            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "创建成功",
                "data": {
                    "file_path": file_path,
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
