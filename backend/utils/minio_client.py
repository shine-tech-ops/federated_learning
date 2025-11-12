"""
MinIO 客户端工具类
用于文件上传、下载、删除等操作
"""
import io
from typing import Optional, BinaryIO
from minio import Minio
from minio.error import S3Error
from django.conf import settings
from loguru import logger


class MinIOClient:
    """MinIO 客户端封装类"""
    
    def __init__(self):
        """初始化 MinIO 客户端"""
        # 从 Django settings 获取配置
        self.endpoint = getattr(settings, 'MINIO_ENDPOINT', 'localhost')
        self.port = getattr(settings, 'MINIO_PORT', 9000)
        self.access_key = getattr(settings, 'MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = getattr(settings, 'MINIO_SECRET_KEY', 'minioadmin123')
        self.use_ssl = getattr(settings, 'MINIO_USE_SSL', False)
        self.bucket_name = getattr(settings, 'MINIO_BUCKET_NAME', 'models')
        
        # 创建 MinIO 客户端
        self.client = Minio(
            f"{self.endpoint}:{self.port}",
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.use_ssl
        )
        
        # 确保 bucket 存在
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """确保 bucket 存在，如果不存在则创建"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    def upload_file(
        self,
        file_data: BinaryIO,
        object_name: str,
        content_type: Optional[str] = None,
        length: Optional[int] = None
    ) -> str:
        """
        上传文件到 MinIO
        
        Args:
            file_data: 文件数据（文件对象或字节流）
            object_name: 对象名称（在 bucket 中的路径）
            content_type: 文件类型（MIME type）
            length: 文件大小（字节）
        
        Returns:
            对象的完整路径
        """
        try:
            # 如果 file_data 是文件对象，获取其大小
            if hasattr(file_data, 'seek') and hasattr(file_data, 'tell'):
                file_data.seek(0, io.SEEK_END)
                file_length = file_data.tell()
                file_data.seek(0)
            elif length:
                file_length = length
            else:
                # 尝试读取整个文件来获取大小
                file_data.seek(0)
                content = file_data.read()
                file_length = len(content)
                file_data = io.BytesIO(content)
            
            # 上传文件
            self.client.put_object(
                self.bucket_name,
                object_name,
                file_data,
                length=file_length,
                content_type=content_type or "application/octet-stream"
            )
            
            logger.info(f"Uploaded file: {object_name} to bucket: {self.bucket_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Error uploading file {object_name}: {e}")
            raise
    
    def download_file(self, object_name: str) -> bytes:
        """
        从 MinIO 下载文件
        
        Args:
            object_name: 对象名称（在 bucket 中的路径）
        
        Returns:
            文件内容的字节数据
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            logger.info(f"Downloaded file: {object_name} from bucket: {self.bucket_name}")
            return data
        except S3Error as e:
            logger.error(f"Error downloading file {object_name}: {e}")
            raise
    
    def get_file_stream(self, object_name: str) -> BinaryIO:
        """
        获取文件流（用于大文件下载）
        
        Args:
            object_name: 对象名称
        
        Returns:
            文件流对象
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response
        except S3Error as e:
            logger.error(f"Error getting file stream {object_name}: {e}")
            raise
    
    def delete_file(self, object_name: str) -> bool:
        """
        删除文件
        
        Args:
            object_name: 对象名称
        
        Returns:
            是否删除成功
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Deleted file: {object_name} from bucket: {self.bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting file {object_name}: {e}")
            return False
    
    def file_exists(self, object_name: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            object_name: 对象名称
        
        Returns:
            文件是否存在
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_file_url(self, object_name: str, expires: int = 3600) -> str:
        """
        获取文件的预签名 URL（用于临时访问）
        
        Args:
            object_name: 对象名称
            expires: URL 过期时间（秒），默认 1 小时
        
        Returns:
            预签名 URL
        """
        try:
            from urllib.parse import urlencode
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL for {object_name}: {e}")
            raise
    
    def list_files(self, prefix: str = "") -> list:
        """
        列出指定前缀的所有文件
        
        Args:
            prefix: 文件路径前缀
        
        Returns:
            文件列表
        """
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []


# 创建全局 MinIO 客户端实例
_minio_client: Optional[MinIOClient] = None


def get_minio_client() -> MinIOClient:
    """获取 MinIO 客户端单例"""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinIOClient()
    return _minio_client

