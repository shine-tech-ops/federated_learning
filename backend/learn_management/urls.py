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
from .model_views import ModelVersionView, ModelRollbackView, ModelInfoView, ModelVersionDeployView, ModelVersionDownloadView
from .region_node_view import RegionNodeView
from .edge_node_view import EdgeNodeView
from .system_log_view import SystemLogView
from .region_api_view import RegionTaskView, DeviceTaskView, DeviceRegisterView, RegionNodeListView


# 封装带 tag 的视图函数（支持多个 method）
def tagged_view(
        view_class, tag, methods=None, operation_summaries=None, operation_descriptions=None
):
    if methods is None:
        methods = ['get', 'post', 'put', 'delete', 'patch']

    # 为每个方法分别应用装饰器
    decorated_view = view_class.as_view()
    for method in methods:
        method_upper = method.upper()
        summary = None
        description = None

        if operation_summaries and isinstance(operation_summaries, dict) and method in operation_summaries:
            summary = operation_summaries[method]
        elif operation_summaries and isinstance(operation_summaries, str):
            summary = operation_summaries

        if operation_descriptions and isinstance(operation_descriptions, dict) and method in operation_descriptions:
            description = operation_descriptions[method]
        elif operation_descriptions and isinstance(operation_descriptions, str):
            description = operation_descriptions

        if summary or description:
            decorated_view = swagger_auto_schema(
                tags=[tag],
                methods=[method],
                operation_summary=summary,
                operation_description=description
            )(decorated_view)

    return decorated_view


urlpatterns = [
    # ======================
    # 系统日志相关 API - tag: "系统日志"
    path(
        "system_log/",
        tagged_view(
            SystemLogView,
            "系统日志",
            methods=['get', 'delete'],
            operation_summaries={
                'get': '查询系统日志列表',
                'delete': '删除系统日志记录'
            },
            operation_descriptions={
                'get': '查询系统中的操作日志和错误日志列表',
                'delete': '根据条件删除系统中的日志记录'
            }
        ),
        name="system-log"
    ),
    # ======================
    # 联邦学习相关 API - tag: "联邦学习"
    # ======================
    path(
        "federated_task/",
        tagged_view(
            FederatedTaskView,
            "任务管理",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询联邦任务列表',
                'post': '创建新的联邦任务',
                'put': '更新联邦任务信息',
                'delete': '删除联邦任务'
            },
            operation_descriptions={
                'get': '查询系统中所有的联邦学习任务，支持分页和条件筛选',
                'post': '创建一个新的联邦学习任务，设置任务参数和参与节点',
                'put': '更新指定联邦任务的配置信息和状态',
                'delete': '删除指定的联邦学习任务及其相关数据'
            }
        ),
        name="federated-task-manager"
    ),
    path(
        "federated_task/pause/",
        tagged_view(
            FederatedTaskPauseView,
            "任务管理",
            methods=['put'],
            operation_summaries={
                'put': '暂停联邦任务'
            },
            operation_descriptions={
                'put': '暂停正在运行的联邦学习任务，保存当前训练状态'
            }
        ),
        name="federated-task-pause"
    ),
    path(
        "federated_task/resume/",
        tagged_view(
            FederatedTaskResumeView,
            "任务管理",
            methods=['put'],
            operation_summaries={
                'put': '恢复联邦任务'
            },
            operation_descriptions={
                'put': '恢复被暂停的联邦学习任务，从上次保存的状态继续执行'
            }
        ),
        name="federated-task-resume"
    ),

    # ======================
    # 系统管理相关 API - tag: "系统配置"
    # ======================
    path(
        "system/config/",
        tagged_view(
            SystemConfigView,
            "系统配置",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询系统配置列表',
                'post': '创建新的系统配置',
                'put': '更新系统配置信息',
                'delete': '删除系统配置'
            },
            operation_descriptions={
                'get': '查询系统中的所有配置项，包括训练参数、网络配置等',
                'post': '创建新的系统配置项，用于定义系统运行参数',
                'put': '更新现有系统配置项的值和参数',
                'delete': '删除指定的系统配置项'
            }
        ),
        name="system-config"
    ),
    path(
        "system/config/activate/",
        tagged_view(
            SystemConfigActivateView,
            "系统配置",
            methods=['post'],
            operation_summaries={
                'post': '激活系统配置'
            },
            operation_descriptions={
                'post': '激活指定的系统配置，使其成为系统当前生效的配置'
            }
        ),
        name='system-config-activate'
    ),
    path(
        "system/aggregation_method/",
        tagged_view(
            AggregationMethodView,
            "系统配置",
            methods=['get'],
            operation_summaries={
                'get': '查询聚合方法列表'
            },
            operation_descriptions={
                'get': '获取系统支持的模型聚合方法列表，用于联邦学习中的模型参数聚合'
            }
        ),
        name="aggregation-method"
    ),

    # ======================
    # 模型管理相关 API - tag: "模型管理"
    # ======================
    path(
        'model_info/',
        tagged_view(
            ModelInfoView,
            "模型管理",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询模型信息列表',
                'post': '创建新的模型信息',
                'put': '更新模型信息',
                'delete': '删除模型信息'
            },
            operation_descriptions={
                'get': '查询系统中注册的所有模型基本信息，包括模型名称、类型等',
                'post': '注册一个新的模型到系统中，记录模型基本信息',
                'put': '更新模型的基本信息和元数据',
                'delete': '从系统中删除指定的模型信息'
            }
        ),
        name='model_info'
    ),
    path(
        'model_version/',
        tagged_view(
            ModelVersionView,
            "模型管理",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询模型版本列表',
                'post': '创建新的模型版本',
                'put': '更新模型版本信息',
                'delete': '删除模型版本'
            },
            operation_descriptions={
                'get': '查询指定模型的所有版本信息，包括版本号、创建时间等',
                'post': '为模型创建新的版本，上传模型文件和参数',
                'put': '更新模型版本的描述信息和状态',
                'delete': '删除指定的模型版本及其相关文件'
            }
        ),
        name='model_version'
    ),
    path(
        'model_rollback/',
        tagged_view(
            ModelRollbackView,
            "模型管理",
            methods=['post'],
            operation_summaries={
                'post': '模型版本回滚'
            },
            operation_descriptions={
                'post': '将模型回滚到指定的历史版本，恢复到之前的模型状态'
            }
        ),
        name='model_rollback'
    ),
    path(
        'model_version/deploy/',
        tagged_view(
            ModelVersionDeployView,
            "模型管理",
            methods=['post'],
            operation_summaries={
                'post': '部署模型版本'
            },
            operation_descriptions={
                'post': '将指定的模型版本部署到生产环境，供实际使用'
            }
        ),
        name='model_version_deploy'
    ),

    path(
        'model_version/<int:id>/download/',
        tagged_view(
            ModelVersionDownloadView,
            "模型管理",
            methods=['get'],
            operation_summaries={
                'get': '下载模型版本文件'
            },
            operation_descriptions={
                'get': '按ID下载对应的模型版本文件（基于存储路径）'
            }
        ),
        name='model_version_download'
    ),

    # ======================
    # 节点管理相关 API - tag: "节点管理"
    # ======================
    path(
        'region_nodes/',
        tagged_view(
            RegionNodeView,
            "区域节点管理",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询区域节点列表',
                'post': '创建新的区域节点',
                'put': '更新区域节点信息',
                'delete': '删除区域节点'
            },
            operation_descriptions={
                'get': '查询系统中所有的区域节点信息，包括节点状态、配置等',
                'post': '添加新的区域节点到系统中，设置节点参数',
                'put': '更新区域节点的配置信息和运行状态',
                'delete': '从系统中移除指定的区域节点'
            }
        ),
        name='region-node-list'
    ),
    path(
        'edge_nodes/',
        tagged_view(
            EdgeNodeView,
            "边缘节点管理",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询边缘节点列表',
                'post': '创建新的边缘节点',
                'put': '更新边缘节点信息',
                'delete': '删除边缘节点'
            },
            operation_descriptions={
                'get': '查询系统中所有的边缘节点信息，包括设备状态、资源等',
                'post': '注册新的边缘设备节点，配置设备参数',
                'put': '更新边缘节点的设备信息和运行配置',
                'delete': '从系统中注销指定的边缘节点'
            }
        ),
        name='edge-node-list'
    ),
    # ======================
    # 边缘设备使用 API - tag: "边缘设备使用API"
    # ======================
    # 获取所有可用region
    path(
        'region/available/',
        tagged_view(
            RegionNodeListView,
            "边缘设备使用API",
            methods=['get'],
            operation_summaries={
                'get': '查询所有可用region'
            },
            operation_descriptions={
                'get': '边缘设备查询系统中所有可用的region列表，用于任务选择'
            }
        ),
        name='region-available'
    ),

    # 获取所有训练任务列表
    path(
        'region/task/',
        tagged_view(
            RegionTaskView,
            "边缘设备使用API",
            methods=['get'],
            operation_summaries={
                'get': '查询区域训练任务列表'
            },
            operation_descriptions={
                'get': '区域服务器获取其管辖范围内的所有训练任务列表，用于任务调度和管理'
            }
        ),
        name='region-task'
    ),
    # 获取设备已加入任务列表
    path(
        'device/task/',
        tagged_view(
            DeviceTaskView,
            "边缘设备使用API",
            methods=['get', 'post', 'put', 'delete'],
            operation_summaries={
                'get': '查询设备任务列表',
                'post': '创建设备任务记录',
                'put': '更新设备任务状态',
                'delete': '删除设备任务记录'
            },
            operation_descriptions={
                'get': '设备查询自己参与的所有训练任务列表和状态信息',
                'post': '设备加入新的训练任务，创建任务参与记录',
                'put': '更新设备在任务中的执行状态和进度信息',
                'delete': '设备退出指定的训练任务，删除任务参与记录'
            }
        ),
        name='device-task'
    ),
    path(
        'device/register/',
        tagged_view(
            DeviceRegisterView,
            "边缘设备使用API",
            methods=['post'],
            operation_summaries={
                'post': '注册边缘设备'
            },
            operation_descriptions={
                'post': '边缘设备首次接入系统时进行注册，获取设备标识和配置信息'
            }
        ),
        name='device-register'
    ),
]