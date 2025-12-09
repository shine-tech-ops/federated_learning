from django.db import models

# Create your models here.
from django.db import models
from utils.common_constant import AGGREGATION_STRATEGIES_MAP

from user.models import AuthUserExtend
from django.utils import timezone
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

    participation_rate = models.IntegerField(default=50, verbose_name="参与率")

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name="状态",
        db_index=True,
    )

    model_info = models.ForeignKey('ModelInfo', on_delete=models.SET_NULL, null=True, blank=True, verbose_name="模型信息")
    model_version = models.ForeignKey('ModelVersion', on_delete=models.SET_NULL, null=True, blank=True, verbose_name="模型版本")
    region_node = models.ForeignKey('RegionNode', on_delete=models.SET_NULL, null=True, blank=True, verbose_name="区域服务器")


    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
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
    region_node = models.ForeignKey(
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
        unique_together = ("device_id",  "region_node")  # 防止重复的节点配置


class TrainingRecord(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('paused', 'Paused'),
        ('canceled', 'Canceled'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    device_id = models.CharField(max_length=100, null=True, blank=True, verbose_name="设备ID")
    edge_node = models.ForeignKey('EdgeNode', on_delete=models.CASCADE, verbose_name="边缘设备")
    federated_task = models.ForeignKey('FederatedTask', on_delete=models.CASCADE, verbose_name="联邦任务")
    model_info = models.ForeignKey('ModelInfo', on_delete=models.CASCADE, verbose_name="模型信息")
    model_version = models.ForeignKey('ModelVersion', on_delete=models.CASCADE, verbose_name="模型版本")
    region_node = models.ForeignKey('RegionNode', on_delete=models.CASCADE, verbose_name="区域节点")

    device_context = models.JSONField(default=dict, verbose_name="设备上下文")
    training_context = models.JSONField(default=dict, verbose_name="训练上下文")

    start_time = models.DateTimeField(auto_now_add=True, verbose_name="开始时间")
    end_time = models.DateTimeField(null=True, blank=True, verbose_name="结束时间")
    duration = models.FloatField(null=True, blank=True, verbose_name="训练时长（秒）")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="状态")

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")


    def save(self, *args, **kwargs):
        # 1. 状态流转检查
        if self.pk is not None:  # 说明是更新操作
            orig = TrainingRecord.objects.get(pk=self.pk)
            orig_status = orig.status
            new_status = self.status

            valid_transitions = {
                'pending': ['running'],
                'running': ['paused', 'canceled', 'completed', 'failed'],
                'paused': ['running', 'canceled'],
            }

            if new_status not in valid_transitions.get(orig_status, []):
                raise ValueError(f"Invalid status transition from {orig_status} to {new_status}")

        # 2. 如果状态是 completed 或 failed，更新 end_time 和 duration
        if self.status in ['completed', 'failed']:
            if not self.end_time:
                self.end_time = timezone.now()
            if self.start_time and self.end_time:
                self.duration = (self.end_time - self.start_time).total_seconds()

        # 3. 调用父类的 save()
        super().save(*args, **kwargs)

    class Meta:
        db_table = "training_record"
        db_table_comment = "训练记录表"
        verbose_name = "训练记录"
        ordering = ["-id"]
        unique_together = ("device_id", "edge_node", "federated_task", "model_info", "model_version", "region_node")



class ModelInferenceLog(models.Model):
 

    model_version = models.ForeignKey('ModelVersion', on_delete=models.CASCADE, related_name='inference_logs', verbose_name="模型版本")
    edge_node = models.ForeignKey('EdgeNode', on_delete=models.CASCADE, null=True, blank=True, related_name='inference_logs', verbose_name="边缘设备")
    input_data = models.JSONField(verbose_name="输入数据")
    output_data = models.JSONField(null=True, blank=True, verbose_name="输出数据")
    error_message = models.TextField(null=True, blank=True, verbose_name="错误信息")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    created_by = models.ForeignKey(
        AuthUserExtend, 
        on_delete=models.SET_NULL, 
        null=True,
        related_name='model_inference_logs_created_by',
        verbose_name="创建人"
    )

    class Meta:
        db_table = "model_inference_log"
        db_table_comment = "模型推理日志表"
        verbose_name = "模型推理日志"
        ordering = ["-id"]


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


class FederatedTrainingLog(models.Model):
    """联邦学习训练日志表"""
    
    LOG_LEVEL_CHOICES = (
        ('DEBUG', 'DEBUG'),
        ('INFO', 'INFO'),
        ('WARNING', 'WARNING'),
        ('ERROR', 'ERROR'),
    )
    
    LOG_PHASE_CHOICES = (
        ('train', '训练阶段'),
        ('upload', '上传阶段'),
        ('aggregate', '聚合阶段'),
        ('evaluate', '评估阶段'),
        ('system', '系统事件'),
    )
    
    # 基础信息
    task = models.ForeignKey('FederatedTask', on_delete=models.CASCADE, related_name='training_logs', verbose_name="联邦任务")
    region_node = models.ForeignKey('RegionNode', on_delete=models.CASCADE, null=True, blank=True, related_name='training_logs', verbose_name="区域节点")
    device_id = models.CharField(max_length=100, null=True, blank=True, db_index=True, verbose_name="设备ID")
    
    # 训练信息
    round = models.IntegerField(null=True, blank=True, db_index=True, verbose_name="训练轮次")
    phase = models.CharField(max_length=20, choices=LOG_PHASE_CHOICES, default='system', db_index=True, verbose_name="日志阶段")
    level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES, default='INFO', db_index=True, verbose_name="日志级别")
    
    # 训练指标
    loss = models.FloatField(null=True, blank=True, verbose_name="损失值")
    accuracy = models.FloatField(null=True, blank=True, verbose_name="准确率")
    num_examples = models.IntegerField(null=True, blank=True, verbose_name="样本数量")
    
    # 其他指标（JSON格式存储）
    metrics = models.JSONField(default=dict, blank=True, null=True, verbose_name="其他指标")
    
    # 日志消息
    message = models.TextField(blank=True, null=True, verbose_name="日志消息")
    error_message = models.TextField(null=True, blank=True, verbose_name="错误信息")
    
    # 时间信息
    log_timestamp = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name="日志时间戳")
    duration = models.FloatField(null=True, blank=True, verbose_name="耗时（秒）")
    
    # 扩展信息
    extra_data = models.JSONField(default=dict, blank=True, null=True, verbose_name="扩展数据")
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        db_table = "federated_training_log"
        db_table_comment = "联邦学习训练日志表"
        verbose_name = "训练日志"
        verbose_name_plural = "训练日志"
        ordering = ["-log_timestamp", "-id"]
        indexes = [
            models.Index(fields=['task', 'round', 'phase']),
            models.Index(fields=['device_id', 'round']),
            models.Index(fields=['level', 'log_timestamp']),
        ]