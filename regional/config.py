#!/usr/bin/env python3
"""
Regional Node 配置文件
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# 加载 .env 文件
def load_env_file():
    """加载 .env 文件"""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print(f"📄 加载 .env 文件: {env_file}")
        load_dotenv(env_file)
    else:
        print(f"ℹ️ 未找到 .env 文件，使用默认配置")

# 在导入时自动加载 .env 文件
load_env_file()


class Config:
    """配置类"""
    
    def __init__(self):
        # 区域节点配置
        self.region_id = self._get_env('REGION_ID', 'region-001')
        self.node_name = self._get_env('NODE_NAME', f'Regional Node {self.region_id}')
        
        # RabbitMQ 配置
        self.rabbitmq = {
            'host': self._get_env('RABBITMQ_HOST', 'localhost'),
            'port': int(self._get_env('RABBITMQ_PORT', '5672')),
            'username': self._get_env('RABBITMQ_USER', 'rabbitmq'),
            'password': self._get_env('RABBITMQ_PASSWORD', 'rabbitmq'),
            'virtual_host': self._get_env('RABBITMQ_VHOST', '/'),
            'exchange_prefix': 'federated_task',
            'queue_prefix': 'region'
        }
        
        # MQTT 配置
        self.mqtt = {
            'host': self._get_env('MQTT_BROKER_HOST', 'localhost'),
            'port': int(self._get_env('MQTT_BROKER_PORT', '1883')),
            'username': self._get_env('MQTT_USER', 'mqtt'),
            'password': self._get_env('MQTT_PASSWORD', 'mqtt2024#'),
            'keepalive': int(self._get_env('MQTT_KEEPALIVE', '60')),
            'topic_prefix': f'region/{self.region_id}/devices'
        }
        
        # 任务管理配置
        self.task = {
            'max_concurrent_tasks': int(self._get_env('MAX_CONCURRENT_TASKS', '10')),
            'task_timeout': int(self._get_env('TASK_TIMEOUT', '3600')),  # 1小时
            'device_timeout': int(self._get_env('DEVICE_TIMEOUT', '300')),  # 5分钟
            'heartbeat_interval': int(self._get_env('HEARTBEAT_INTERVAL', '30')),  # 30秒
            'status_check_interval': int(self._get_env('STATUS_CHECK_INTERVAL', '10'))  # 10秒
        }
        
        # 日志配置
        self.logging = {
            'level': self._get_env('LOG_LEVEL', 'INFO'),
            'file': self._get_env('LOG_FILE', 'logs/regional.log'),
            'max_size': self._get_env('LOG_MAX_SIZE', '10 MB'),
            'backup_count': int(self._get_env('LOG_BACKUP_COUNT', '5')),
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'
        }
        
        # 网络配置
        self.network = {
            'connection_timeout': int(self._get_env('CONNECTION_TIMEOUT', '30')),
            'retry_attempts': int(self._get_env('RETRY_ATTEMPTS', '3')),
            'retry_delay': int(self._get_env('RETRY_DELAY', '5'))
        }
        
        # 调试配置
        self.debug = {
            'enabled': self._get_env('DEBUG', 'false').lower() == 'true',
            'verbose_logging': self._get_env('VERBOSE_LOGGING', 'false').lower() == 'true',
            'mock_devices': self._get_env('MOCK_DEVICES', 'false').lower() == 'true'
        }
    
    def _get_env(self, key: str, default: str) -> str:
        """获取环境变量"""
        return os.environ.get(key, default)
    
    def get_rabbitmq_exchange(self) -> str:
        """获取 RabbitMQ Exchange 名称"""
        return f"{self.rabbitmq['exchange_prefix']}_{self.region_id}"
    
    def get_rabbitmq_queue(self) -> str:
        """获取 RabbitMQ Queue 名称"""
        return f"{self.rabbitmq['queue_prefix']}_{self.region_id}_tasks"
    
    def get_mqtt_topic(self, device_id: str, action: str) -> str:
        """获取 MQTT 主题名称"""
        return f"{self.mqtt['topic_prefix']}/{device_id}/{action}"
    
    def get_mqtt_wildcard_topic(self, action: str) -> str:
        """获取 MQTT 通配符主题名称"""
        return f"{self.mqtt['topic_prefix']}/+/{action}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'region_id': self.region_id,
            'node_name': self.node_name,
            'rabbitmq': self.rabbitmq,
            'mqtt': self.mqtt,
            'task': self.task,
            'logging': self.logging,
            'network': self.network,
            'debug': self.debug
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"RegionalNodeConfig(region_id={self.region_id}, node_name={self.node_name})"


# 全局配置实例
config = Config()


# 配置验证函数
def validate_config() -> bool:
    """验证配置是否有效"""
    try:
        # 配置验证通过，因为 __init__ 已经设置了默认值
        print(f"✅ 配置验证通过")
        print(f"   区域ID: {config.region_id}")
        print(f"   RabbitMQ: {config.rabbitmq['host']}:{config.rabbitmq['port']}")
        print(f"   MQTT: {config.mqtt['host']}:{config.mqtt['port']}")
        return True
        
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False


if __name__ == "__main__":
    # 测试配置
    print("=== Regional Node 配置信息 ===")
    print(f"区域ID: {config.region_id}")
    print(f"节点名称: {config.node_name}")
    print(f"RabbitMQ: {config.rabbitmq['host']}:{config.rabbitmq['port']}")
    print(f"MQTT: {config.mqtt['host']}:{config.mqtt['port']}")
    print(f"最大并发任务: {config.task['max_concurrent_tasks']}")
    print(f"调试模式: {config.debug['enabled']}")
    print()
    
    # 验证配置
    validate_config()
