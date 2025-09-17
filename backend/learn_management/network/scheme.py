# backend/learn_management/network/scheme.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DeviceRoute:
    """设备路由信息"""
    device_id: str         # 设备ID
    internal_ip: str       # 设备内网IP
    internal_port: int     # 设备端口
    status: str = 'online' # 设备状态

@dataclass
class RegionConfig:
    """区域节点配置"""
    # 中心服务器信息
    central_server: str    # 中心服务器地址 (例如: "http://central:8080")
    
    # 设备路由表
    device_routes: Dict[str, DeviceRoute]  # {device_id: route_info}

    def add_device(self, device_id: str, ip: str, port: int):
        """添加设备路由"""
        self.device_routes[device_id] = DeviceRoute(
            device_id=device_id,
            internal_ip=ip,
            internal_port=port
        )

    def remove_device(self, device_id: str):
        """移除设备路由"""
        if device_id in self.device_routes:
            del self.device_routes[device_id]