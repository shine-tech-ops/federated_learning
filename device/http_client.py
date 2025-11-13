"""
设备端 HTTP 客户端 - 用于向中央服务器发送心跳
"""

import requests
import json
import time
from loguru import logger
from typing import Dict, Any, Optional


class HTTPClient:
    """HTTP 客户端 - 向中央服务器发送心跳"""
    
    def __init__(self, base_url: str = 'http://localhost:8085', timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'EdgeDevice/1.0'
        })

    def send_heartbeat(self, device_id: str, region_node: int, device_context: Dict[str, Any] = None, 
                       ip_address: Optional[str] = None, status: str = "online", 
                       description: Optional[str] = None) -> bool:
        """发送设备心跳到中央服务器"""
        url = f"{self.base_url}/api/v1/learn_management/device/heartbeat/"
        data = {
            "device_id": device_id,
            "region_node": region_node,
            "device_context": device_context or {},
            "status": status
        }
        
        # 添加可选字段
        if ip_address:
            data["ip_address"] = ip_address
        if description:
            data["description"] = description
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            if result.get('code') == 200:
                logger.debug(f"设备 {device_id} 心跳发送成功")
                return True
            else:
                logger.warning(f"设备 {device_id} 心跳发送失败: {result.get('msg')}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"设备 {device_id} 心跳发送异常: {e}")
            return False
    
    def close(self):
        """关闭 HTTP 客户端"""
        self.session.close()

