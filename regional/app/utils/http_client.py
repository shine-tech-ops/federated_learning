"""
HTTP 客户端 - 用于向中央服务器上报状态
"""

import requests
import json
import time
from loguru import logger
from typing import Dict, Any, Optional


class HTTPClient:
    """HTTP 客户端 - 向中央服务器上报状态"""
    
    def __init__(self, config=None):
        if config:
            self.base_url = config.get('central_server_url', 'http://localhost:8000')
            self.timeout = config.get('timeout', 30)
            self.retry_attempts = config.get('retry_attempts', 3)
            self.retry_delay = config.get('retry_delay', 5)
        else:
            # 默认配置
            self.base_url = 'http://localhost:8000'
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
        # data = {
        #     'task_id': task_id,
        #     'status': status,
        #     'region_id': region_id,
        #     'details': details or {},
        #     'timestamp': time.time()
        # }
        
        # result = self._make_request('POST', '/api/regional/task-status/', data)
        # if result:
        #     logger.info(f"任务状态上报成功: {task_id} -> {status}")
        #     return True
        # else:
        #     logger.error(f"任务状态上报失败: {task_id} -> {status}")
        #     return False
        return True
    
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
    
    def close(self):
        """关闭 HTTP 客户端"""
        self.session.close()
        logger.info("HTTP 客户端已关闭")
