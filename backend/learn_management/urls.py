# learn_management/urls.py

from django.urls import path
from drf_yasg.utils import swagger_auto_schema

# 导入视图类
from .federated_task_views import (
    FederatedTaskView,
    FederatedTaskPauseView,
    FederatedTaskResumeView
)
from .system_config_views import SystemConfigView, SystemConfigActivateView, AggregationMethodView
from .model_views import ModelVersionView, ModelRollbackView, ModelInfoView, ModelVersionDeployView
from .region_node_view import RegionNodeView
from .edge_node_view import EdgeNodeView
from .system_log_view import SystemLogView

# 封装带 tag 的视图函数（支持多个 method）
def tagged_view(view_class, tag, methods=None):
    if methods is None:
        methods = ['get', 'post', 'put', 'delete', 'patch']
    return swagger_auto_schema(tags=[tag], methods=methods)(view_class.as_view())

urlpatterns = [
    # ======================
    # 系统日志相关 API - tag: "系统日志"
    path("system_log/", tagged_view(SystemLogView, "系统日志", methods=['get', 'delete']) , name="system-log"),
    # ======================
    # 联邦学习相关 API - tag: "联邦学习"
    # ======================
    path("federated_task/", tagged_view(FederatedTaskView, "任务管理", methods=['get', 'post', 'put', 'delete']), name="federated-task-manager"),
    path("federated_task/pause/", tagged_view(FederatedTaskPauseView, "任务管理", methods=['put']), name="federated-task-pause"),
    path("federated_task/resume/", tagged_view(FederatedTaskResumeView, "任务管理", methods=['put']), name="federated-task-resume"),

    # ======================
    # 系统管理相关 API - tag: "系统配置"
    # ======================
    path("system/config/", tagged_view(SystemConfigView, "系统配置", methods=['get', 'post', 'put', 'delete']), name="system-config"),
    path("system/config/activate/", tagged_view(SystemConfigActivateView, "系统配置", methods=['post']), name='system-config-activate'),
    path("system/aggregation_method/", tagged_view(AggregationMethodView, "系统配置", methods=['get']), name="aggregation-method"),

    # ======================
    # 模型管理相关 API - tag: "模型管理"
    # ======================
    path('model_info/', tagged_view(ModelInfoView, "模型管理", methods=['get', 'post', 'put', 'delete']), name='model_info'),
    path('model_version/', tagged_view(ModelVersionView, "模型管理", methods=['get', 'post', 'put', 'delete']), name='model_version'),
    path('model_rollback/', tagged_view(ModelRollbackView, "模型管理", methods=['post']), name='model_rollback'),
    path('model_version/deploy/', tagged_view(ModelVersionDeployView, "模型管理", methods=['post']), name='model_version_deploy'),

    # ======================
    # 节点管理相关 API - tag: "节点管理"
    # ======================
    path('region_nodes/', tagged_view(RegionNodeView, "区域节点管理", methods=['get', 'post', 'put', 'delete']), name='region-node-list'),
    path('edge_nodes/', tagged_view(EdgeNodeView, "边缘节点管理", methods=['get', 'post', 'put', 'delete']), name='edge-node-list'),
]