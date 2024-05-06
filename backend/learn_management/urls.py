from django.urls import path

# 导入联邦学习相关视图
from .federated_task_views import (
    FederatedTaskView,
    FederatedTaskPauseView,
    FederatedTaskResumeView
)
# 导入系统管理和模型管理相关视图
from .system_config_views import SystemConfigView, SystemConfigActivateView, AggregationMethodView
from .model_views import (
    ModelVersionView,
    ModelRollbackView,
    ModelInfoView,
    ModelVersionDeployView,
)

urlpatterns = [
    # 联邦学习相关 API
    path("federated_task/", FederatedTaskView.as_view(), name="federated-task-manager"),
    path("federated_task/pause/", FederatedTaskPauseView.as_view(), name="federated-task-pause"),
    path("federated_task/resume/", FederatedTaskResumeView.as_view(), name="federated-task-resume"),

    # 系统管理相关 API
    path("system/config/", SystemConfigView.as_view(), name="system-config"),
    path("system/config/activate/", SystemConfigActivateView.as_view(), name='system-config-activate'),

    path("system/aggregation_method/", AggregationMethodView.as_view(), name="aggregation-method"),

    # 模型管理相关 API
    path('model_version/', ModelVersionView.as_view(), name='model_version'),
    path('model_rollback/', ModelRollbackView.as_view(), name='model_rollback'),
    # 模型信息管理
    path('model_info/', ModelInfoView.as_view(), name='model_info'),

    # 模型版本管理
    path('model_version/', ModelVersionView.as_view(), name='model_version'),
    path('model_version/deploy/', ModelVersionDeployView.as_view(), name='model_version_deploy'),
]
