import threading
from datetime import timedelta

from django.utils import timezone
from django.db import close_old_connections

import utils.common_constant as const
from loguru import logger


def check_offline_devices():
    try:
        logger.info("start check device status.")
        # 避免数据库连接超时
        close_old_connections()

        # 设置离线阈值：5 分钟未心跳
        threshold = timezone.now() - timedelta(seconds=const.HEARTBEAT_TIMEOUT)

        # 更新所有超过阈值的设备为离线
        from learn_management.models import EdgeNode
        EdgeNode.objects.filter(last_heartbeat__lt=threshold).update(status="offline")

        offline_count = EdgeNode.objects.filter(status="offline").count()
        logger.info(f"[{timezone.now()}] has marked {offline_count} devices to offline.")

    except Exception as e:
        logger.error(f"check device status error: {str(e)}")
    finally:
        # 5 分钟后再次执行
        start_background_task()

def start_background_task():
    # 使用 Timer 延迟执行
    timer = threading.Timer(const.HEARTBEAT_TIMEOUT, check_offline_devices)  # 300 秒 = 5 分钟
    timer.daemon = True  # 守护线程，主线程退出时自动结束
    timer.start()