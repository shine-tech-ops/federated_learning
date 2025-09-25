# backend/learn_management/network/services.py
from typing import Optional, Dict
from .scheme import RegionConfig, DeviceRoute
from ..models import RegionNode, EdgeNode

from django.conf import settings


class NetworkConfigService:
    """网络配置服务"""
    
    def get_region_routes(self, region_id: str) -> Optional[RegionConfig]:
        """
        获取区域节点的路由配置
        
        Args:
            region_id: 区域节点ID
            
        Returns:
            RegionConfig: 区域节点配置，如果区域节点不存在返回 None
        """
        try:
            # 获取区域节点信息
            region = RegionNode.objects.get(id=region_id)
            
            # 获取该区域下所有在线的边缘设备
            edge_nodes = EdgeNode.objects.filter(
                region_node=region,
                status='online'  # 只获取在线设备
            )
            
            # 构建设备路由表
            device_routes = {}
            for edge in edge_nodes:
                if edge.device_id and edge.ip_address:  # 确保有必要信息
                    device_routes[edge.device_id] = DeviceRoute(
                        device_id=edge.device_id,
                        internal_ip=edge.ip_address,
                        internal_port=8000,  # 这个可能需要从配置或设备信息中获取
                        status=edge.status
                    )
            
            # 创建区域配置
            # central_server 地址可能需要从系统配置中获取
            return RegionConfig(
                central_server=f"http://{settings.CENTRAL_SERVER_HOST}:{settings.CENTRAL_SERVER_PORT}",  # 这个地址应该从配置中读取
                device_routes=device_routes
            )
            
        except RegionNode.DoesNotExist:
            return None
        except Exception as e:
            # 这里可能需要添加日志
            print(f"Error getting region routes: {e}")
            return None

    def get_all_region_routes(self) -> Dict[str, RegionConfig]:
        """
        获取所有区域节点的路由配置
        
        Returns:
            Dict[str, RegionConfig]: 区域节点ID到配置的映射
        """
        configs = {}
        for region in RegionNode.objects.all():
            config = self.get_region_routes(str(region.id))
            if config:
                configs[str(region.id)] = config
        return configs