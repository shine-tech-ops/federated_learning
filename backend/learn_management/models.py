from django.db import models

# Create your models here.
from django.db import models
from utils.common_constant import AGGREGATION_STRATEGIES_MAP

from user.models import AuthUserExtend
from timescale.db.models.managers import TimescaleManager
from timescale.db.models.fields import TimescaleDateTimeField

AGGREGATION_METHOD_CHOICES = [
    (key, value['label']) for key, value in AGGREGATION_STRATEGIES_MAP.items()
]


class FederatedTask(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('paused', 'Paused'),
        ('completed', 'Completed'),
    )
    name = models.CharField(max_length=255, verbose_name="任务名称")
    description = models.TextField(blank=True, null=True, verbose_name="描述")
    rounds = models.IntegerField(default=10, verbose_name="训练轮次")
    aggregation_method = models.CharField(
        max_length=50,
        default='fedavg',
        choices=AGGREGATION_METHOD_CHOICES,
        verbose_name="聚合方式",
        db_index=True,
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name="状态",
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(AuthUserExtend, on_delete=models.CASCADE, related_name='federated_task_created_by', verbose_name='创建人')

    class Meta:
        db_table = "federated_task"
        db_table_comment = "联邦学习任务表"
        verbose_name = "联邦学习任务"
        ordering = ["-id"]


class SystemConfig(models.Model):
    name = models.CharField(max_length=50, unique=True, verbose_name="配置名称", help_text="系统配置的唯一名称")
    description = models.TextField(blank=True, null=True, verbose_name="描述", help_text="配置的详细说明")
    config_data = models.JSONField(default=dict, verbose_name="配置内容", help_text="存储联邦学习相关参数的JSON结构")
    is_active = models.BooleanField(default=False, verbose_name="是否激活", help_text="标记该配置是否为当前生效配置")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间", help_text="配置创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="最后更新时间", help_text="配置最后一次修改时间")
    created_by = models.ForeignKey(AuthUserExtend, on_delete=models.CASCADE, related_name='system_config_create_by', verbose_name="创建人")

    class Meta:
        db_table = "system_config"
        db_table_comment = "系统配置表"
        verbose_name = "系统配置"
        ordering = ["-id"]


class ModelInfo(models.Model):
    name = models.CharField(max_length=255, unique=True, verbose_name="模型名称")
    description = models.TextField(blank=True, null=True, verbose_name="描述")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(AuthUserExtend, on_delete=models.CASCADE, related_name='model_info_create_by', verbose_name="创建人")


    class Meta:
        db_table = "model_info"
        verbose_name = "模型信息"
        ordering = ["-id"]


class ModelVersion(models.Model):
    STATUS_CHOICES = (
        ('pending', '待评估'),
        ('active', '已激活'),
        ('archived', '归档'),
    )
    model_info = models.ForeignKey(ModelInfo, on_delete=models.CASCADE, related_name='versions')
    version = models.CharField(max_length=100, verbose_name="版本号")
    model_file = models.CharField(max_length=255, verbose_name="模型文件")
    description = models.TextField(blank=True, null=True, verbose_name="描述")
    accuracy = models.FloatField(null=True, blank=True, verbose_name="准确率")
    loss = models.FloatField(null=True, blank=True, verbose_name="损失值")
    metrics = models.JSONField(default=dict, blank=True, null=True, verbose_name="其他指标")  # 如 precision, recall 等
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="状态")
    is_deployed = models.BooleanField(default=False, verbose_name="是否已部署")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="最后更新时间")
    created_by = models.ForeignKey(AuthUserExtend, on_delete=models.CASCADE, related_name='model_version_create_by', verbose_name="创建人")

    class Meta:
        db_table = "model_version"
        db_table_comment = "模型版本表"
        verbose_name = "模型版本"
        ordering = ["-id"]
        unique_together = ('model_info', 'version')  # 同一模型不能重复版本号

class RegionNode(models.Model):
    name = models.CharField(max_length=100, unique=True, verbose_name="区域节点名称")
    description = models.TextField(blank=True, null=True, verbose_name="描述")
    ip_address = models.GenericIPAddressField(null=True, blank=True,verbose_name="IP地址")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    created_by = models.ForeignKey(
        AuthUserExtend,
        on_delete=models.SET_NULL,
        null=True,
        related_name='region_nodes_created_by',
        verbose_name="创建人"
    )

    class Meta:
        db_table = "region_node"
        db_table_comment = "区域节点表"
        verbose_name = "区域节点"
        ordering = ["-id"]

class EdgeNode(models.Model):
    STATUS_CHOICES = (
        ('online', '在线'),
        ('offline', '离线'),
        ('maintenance', '维护中'),
    )
    region = models.ForeignKey(
        RegionNode,
        on_delete=models.CASCADE,
        related_name='edge_nodes',
        verbose_name="所属区域节点"
    )
    device_id = models.CharField(max_length=100, null=True, blank=True, verbose_name="边缘设备id")
    ip_address = models.GenericIPAddressField(null=True, blank=True,verbose_name="IP地址")
    device_context = models.JSONField(default=dict, blank=True, null=True, verbose_name="设备上下文")
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='offline',
        verbose_name="状态",
        db_index=True
    )
    last_heartbeat = models.DateTimeField(null=True, blank=True, verbose_name="最后心跳时间")
    description = models.TextField(blank=True, null=True, verbose_name="描述")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    created_by = models.ForeignKey(
        AuthUserExtend,
        on_delete=models.SET_NULL,
        null=True,
        related_name='edge_nodes_created_by',
        verbose_name="创建人"
    )

    class Meta:
        db_table = "edge_node"
        db_table_comment = "边缘节点表"
        verbose_name = "边缘节点"
        ordering = ["-id"]
        unique_together = ("device_id",  "region")  # 防止重复的节点配置


class OperationLog(models.Model):
    user = models.ForeignKey(AuthUserExtend, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="用户")
    ip = models.CharField(max_length=100, null=True, blank=True, verbose_name="IP地址")
    method = models.CharField(max_length=10, null=True, blank=True, verbose_name="请求方法")
    path = models.CharField(max_length=255, null=True, blank=True, verbose_name="请求路径")
    body = models.TextField(null=True, blank=True, verbose_name="请求体")
    response_code = models.IntegerField(null=True, blank=True, verbose_name="响应状态码")
    response_body = models.TextField(null=True, blank=True, verbose_name="响应体")
    created_at = TimescaleDateTimeField(interval="1 day", auto_now_add=True, verbose_name="创建时间")

    objects = TimescaleManager()

    class Meta:
        db_table = "operation_log"
        db_table_comment = "操作日志表"
        verbose_name = "操作日志"
        verbose_name_plural = "操作日志"
        ordering = ["-id"]