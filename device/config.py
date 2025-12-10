#!/usr/bin/env python3
"""
设备端配置文件
通过环境变量或 .env 文件驱动配置
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv


def load_env_file() -> None:
    """加载当前目录下的 .env 文件（如果存在）"""
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        print(f"📄 加载 device 配置: {env_file}")
        load_dotenv(env_file)
    else:
        print("ℹ️ 未找到 device/.env，使用默认配置")


# 导入时自动加载 .env
load_env_file()


class Config:
    """设备端配置"""

    def __init__(self) -> None:
        # 设备与区域信息
        self.device_id = self._get_env("DEVICE_ID", "device_001")
        self.region_id = int(self._get_env("REGION_ID", "1"))

        # MQTT 配置
        self.mqtt: Dict[str, Any] = {
            "host": self._get_env("MQTT_BROKER_HOST", "localhost"),
            "port": int(self._get_env("MQTT_BROKER_PORT", "1883")),
            "username": self._get_env("MQTT_USER", "mqtt"),
            "password": self._get_env("MQTT_PASSWORD", "mqtt2024#"),
            "keepalive": int(self._get_env("MQTT_KEEPALIVE", "60")),
        }

        # HTTP/中央服务器配置
        self.http: Dict[str, Any] = {
            "base_url": self._get_env("CENTRAL_SERVER_URL", "http://localhost:8085"),
            "timeout": int(self._get_env("HTTP_TIMEOUT", "10")),
        }

        # 心跳间隔（秒）
        self.heartbeat_interval = int(self._get_env("HEARTBEAT_INTERVAL", "30"))

        # 日志配置
        self.logging: Dict[str, Any] = {
            "level": self._get_env("LOG_LEVEL", "INFO"),
            "file": self._get_env("LOG_FILE", "logs/device_{device_id}.log"),
            "format": self._get_env(
                "LOG_FORMAT", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            ),
            "max_size": self._get_env("LOG_MAX_SIZE", "10 MB"),
        }

    def _get_env(self, key: str, default: str) -> str:
        """读取环境变量"""
        return os.environ.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "device_id": self.device_id,
            "region_id": self.region_id,
            "mqtt": self.mqtt,
            "http": self.http,
            "heartbeat_interval": self.heartbeat_interval,
            "logging": self.logging,
        }


# 全局配置实例
config = Config()


