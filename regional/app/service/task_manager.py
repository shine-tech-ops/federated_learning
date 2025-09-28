"""
任务管理器 - 管理联邦学习任务的执行
"""

import json
import time
from typing import Dict, Any, List
from loguru import logger


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.device_tasks: Dict[str, List[str]] = {}  # device_id -> task_ids
    
    def start_task(self, task_data: Dict[str, Any]):
        """开始任务"""
        task_id = task_data['task_id']
        task_name = task_data.get('task_name', '未知任务')
        rounds = task_data.get('rounds', 10)
        aggregation_method = task_data.get('aggregation_method', 'fedavg')
        
        logger.info(f"🎯 任务管理器启动任务")
        logger.info(f"   • 任务ID: {task_id}")
        logger.info(f"   • 任务名称: {task_name}")
        logger.info(f"   • 训练轮数: {rounds}")
        logger.info(f"   • 聚合方法: {aggregation_method}")
        
        # 保存任务信息
        self.active_tasks[task_id] = {
            **task_data,
            'status': 'running',
            'start_time': time.time(),
            'devices': [],
            'current_round': 0,
            'total_rounds': rounds
        }
        
        # 初始化设备任务映射
        self.device_tasks[task_id] = []
        
        logger.info(f"✅ 任务 {task_id} 已开始，共 {rounds} 轮")
    
    def pause_task(self, task_id: str):
        """暂停任务"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'paused'
            logger.info(f"任务 {task_id} 已暂停")
        else:
            logger.warning(f"任务 {task_id} 不存在")
    
    def resume_task(self, task_id: str):
        """恢复任务"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'running'
            logger.info(f"任务 {task_id} 已恢复")
        else:
            logger.warning(f"任务 {task_id} 不存在")
    
    def add_device_to_task(self, task_id: str, device_id: str):
        """添加设备到任务"""
        if task_id in self.active_tasks:
            if device_id not in self.device_tasks[task_id]:
                self.device_tasks[task_id].append(device_id)
                self.active_tasks[task_id]['devices'].append(device_id)
                logger.info(f"设备 {device_id} 已加入任务 {task_id}")
        else:
            logger.warning(f"任务 {task_id} 不存在")
    
    def remove_device_from_task(self, task_id: str, device_id: str):
        """从任务中移除设备"""
        if task_id in self.active_tasks and device_id in self.device_tasks[task_id]:
            self.device_tasks[task_id].remove(device_id)
            self.active_tasks[task_id]['devices'].remove(device_id)
            logger.info(f"设备 {device_id} 已从任务 {task_id} 中移除")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        return self.active_tasks.get(task_id, {})
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务"""
        return self.active_tasks.copy()
    
    def complete_task(self, task_id: str):
        """完成任务"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'completed'
            self.active_tasks[task_id]['end_time'] = time.time()
            logger.info(f"任务 {task_id} 已完成")
    
    def get_devices_for_task(self, task_id: str) -> List[str]:
        """获取任务的设备列表"""
        return self.device_tasks.get(task_id, [])
