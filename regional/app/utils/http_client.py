"""
HTTP 客户端 - 用于向中央服务器上报状态
"""

import requests
import json
import time
from loguru import logger
from typing import Dict, Any, Optional, Union


class HTTPClient:
    """HTTP 客户端 - 向中央服务器上报状态"""
    
    def __init__(self, config=None):
        if config:
            # 支持两种配置格式：central_server_url 或 url
            self.base_url = config.get('central_server_url') or config.get('url', 'http://localhost:8085')
            self.timeout = config.get('timeout', 30)
            self.retry_attempts = config.get('retry_attempts', 3)
            self.retry_delay = config.get('retry_delay', 5)
        else:
            # 默认配置
            self.base_url = 'http://localhost:8085'
            self.timeout = 30
            self.retry_attempts = 3
            self.retry_delay = 5
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RegionalNode/1.0'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """发送 HTTP 请求"""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"发送 {method} 请求到 {url} (尝试 {attempt + 1}/{self.retry_attempts})")
                
                if method.upper() == 'GET':
                    response = self.session.get(url, timeout=self.timeout)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=data, timeout=self.timeout)
                elif method.upper() == 'PUT':
                    response = self.session.put(url, json=data, timeout=self.timeout)
                else:
                    raise ValueError(f"不支持的 HTTP 方法: {method}")
                
                response.raise_for_status()
                
                # 尝试解析 JSON 响应
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {'status': 'success', 'message': response.text}
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"HTTP 请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"HTTP 请求最终失败: {e}")
                    return None
        
        return None
    
    def report_task_status(self, task_id: str, status: str, region_id: str, details: Dict[str, Any] = None) -> bool:
        """上报任务状态到中央服务器"""
        # 如果是完成状态，调用完成接口
        if status == 'completed':
            # 从 details 中提取 model_file_path
            model_file_path = details.get('model_file_path') if details else None
            return self.complete_task(task_id, model_file_path=model_file_path)
        return True
    
    def complete_task(self, task_id: str, model_file_path: str = None) -> bool:
        """上报任务完成到中央服务器"""
        endpoint = "/api/v1/learn_management/federated_task/complete/"
        data = {'id': int(task_id)}
        if model_file_path:
            data['model_file_path'] = model_file_path
        
        result = self._make_request('POST', endpoint, data)
        if result and result.get('code') == 200:
            logger.info(f"任务完成上报成功: {task_id}")
            return True
        else:
            logger.error(f"任务完成上报失败: {task_id}, result: {result}")
            return False
    
    def report_device_result(self, device_id: str, result_data: Dict[str, Any], region_id: str) -> bool:
        """上报设备训练结果到中央服务器"""
        data = {
            'device_id': device_id,
            'result_data': result_data,
            'region_id': region_id,
            'timestamp': time.time()
        }
        
        # result = self._make_request('POST', '/api/regional/device-result/', data)
        # if result:
        #     logger.info(f"设备结果上报成功: {device_id}")
        #     return True
        # else:
        #     logger.error(f"设备结果上报失败: {device_id}")
        #     return False
        return True
    
    def report_device_status(self, device_id: str, status_data: Dict[str, Any], region_id: str) -> bool:
        """上报设备状态到中央服务器"""
        # data = {
        #     'device_id': device_id,
        #     'status_data': status_data,
        #     'region_id': region_id,
        #     'timestamp': time.time()
        # }
        
        # result = self._make_request('POST', '/api/regional/device-status/', data)
        # if result:
        #     logger.debug(f"设备状态上报成功: {device_id}")
        #     return True
        # else:
        #     logger.error(f"设备状态上报失败: {device_id}")
        #     return False
        return True

    def upload_training_logs(self, logs: Union[Dict[str, Any], list]) -> bool:
        """
        上传训练/聚合日志到中央服务器
        
        支持单条 dict 或 list（批量）格式，返回是否请求成功。
        """
        endpoint = "/api/v1/learn_management/training_log/"
        result = self._make_request('POST', endpoint, logs)
        if not result:
            return False

        code = result.get("code")
        if code == 200:
            logger.debug(f"训练日志上传成功: {result.get('msg')}")
            return True

        logger.warning(f"训练日志上传失败: {result.get('msg')}")
        return False
    
    def health_check(self) -> bool:
        """健康检查 - 检查与中央服务器的连接"""
        # result = self._make_request('GET', '/api/health/')
        # if result:
        #     logger.info("中央服务器连接正常")
        #     return True
        # else:
        #     logger.warning("中央服务器连接异常")
        #     return False
        return True
    
    def upload_model_file(self, file_path: str, task_id: str = None) -> Optional[Dict[str, Any]]:
        """上传模型文件到中央服务器"""
        try:
            import os
            
            if not os.path.exists(file_path):
                logger.error(f"模型文件不存在: {file_path}")
                return None
            
            url = f"{self.base_url.rstrip('/')}/api/v1/learn_management/model_version/upload/"
            
            # 添加任务ID作为元数据（如果需要）
            data = {}
            if task_id:
                data['task_id'] = task_id
            
            for attempt in range(self.retry_attempts):
                try:
                    logger.info(f"上传模型文件到 {url} (尝试 {attempt + 1}/{self.retry_attempts})")
                    
                    # 每次尝试都重新打开文件
                    with open(file_path, 'rb') as f:
                        files = {
                            'file': (os.path.basename(file_path), f, 'application/octet-stream')
                        }
                        
                        # 对于文件上传，使用 requests.post 而不是 session.post
                        # 这样可以避免 session 的 Content-Type: application/json header 干扰
                        # requests 在使用 files 参数时会自动设置正确的 multipart/form-data Content-Type（包含 boundary）
                        response = requests.post(
                            url,
                            files=files,
                            data=data,
                            headers={'User-Agent': 'RegionalNode/1.0'},  # 只保留必要的 header，Content-Type 由 requests 自动设置
                            timeout=self.timeout * 2  # 上传文件需要更长的超时时间
                        )
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    if result.get('code') == 200:
                        logger.info(f"模型文件上传成功: {file_path}, result: {result}")
                        return result.get('data', {})
                    else:
                        logger.error(f"模型文件上传失败: {result.get('msg', 'Unknown error')}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(self.retry_delay)
                        else:
                            return None
                            
                except requests.exceptions.RequestException as e:
                    logger.warning(f"模型文件上传失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"模型文件上传最终失败: {e}")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"上传模型文件异常: {e}")
            return None
    
    def close(self):
        """关闭 HTTP 客户端"""
        self.session.close()
        logger.info("HTTP 客户端已关闭")


if __name__ == "__main__":
    #  python -m app.utils.http_client
    config = {
        'central_server_url': 'http://localhost:8085',
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 5
    }
    http_client = HTTPClient(config)
    path = '/Users/vincent/code/federated_learning/regional/parameters/final_model_round_002.npz'
    http_client.upload_model_file(path)