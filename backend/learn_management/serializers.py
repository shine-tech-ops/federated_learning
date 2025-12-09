from datetime import timedelta

from django.utils import timezone
from rest_framework import serializers
from .models import FederatedTask, SystemConfig, ModelVersion, ModelInfo, RegionNode, EdgeNode, TrainingRecord, OperationLog, ModelInferenceLog, FederatedTrainingLog
from user.serializers import CommonUserSerializer
import utils.common_constant as const

class SystemConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = SystemConfig
        fields = "__all__"


class ModelInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelInfo
        fields = "__all__"

class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = "__all__"

class RegionNodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegionNode
        fields = '__all__'

class FederatedTaskSerializer(serializers.ModelSerializer):

    def to_representation(self, obj):
        ret = super(FederatedTaskSerializer, self).to_representation(obj)
        ret['region_node_detail'] = RegionNodeSerializer(obj.region_node).data
        ret['model_info_detail'] = ModelInfoSerializer(obj.model_info).data
        ret['model_version_detail'] = ModelVersionSerializer(obj.model_version).data
        sysconfig = SystemConfig.objects.filter(is_active=True).first()
        ret['aggregation_method'] = sysconfig.config_data.get('federated', {}).get('aggregation')
        ret['rounds'] = sysconfig.config_data.get('federated', {}).get('rounds', 10)
        ret['participation_rate'] = sysconfig.config_data.get('federated', {}).get('participationRate', 0.5)
        return ret
    class Meta:
        model = FederatedTask
        fields = "__all__"



class EdgeNodeSerializer(serializers.ModelSerializer):
    def to_representation(self, obj):
        ret = super(EdgeNodeSerializer, self).to_representation(obj)
        ret['region_node_detail'] = RegionNodeSerializer(obj.region_node).data
        # 动态判断心跳是否过期，过期则视为离线返回
        last_heartbeat = obj.last_heartbeat
        is_stale = (
            (not last_heartbeat)
            or (timezone.now() - last_heartbeat > timedelta(seconds=const.HEARTBEAT_STALE_SECONDS))
        )
        if is_stale:
            ret['status'] = 'offline'
        ret['stale_after_seconds'] = const.HEARTBEAT_STALE_SECONDS
        return ret
    class Meta:
        model = EdgeNode
        fields = '__all__'


class TrainingRecordSerializer(serializers.ModelSerializer):
    edge_node_detail = serializers.SerializerMethodField()
    region_node_detail = serializers.SerializerMethodField()
    model_info_detail = serializers.SerializerMethodField()
    model_version_detail = serializers.SerializerMethodField()

    def get_edge_node_detail(self, obj):
        return EdgeNodeSerializer(obj.edge_node).data

    def get_region_node_detail(self, obj):
        return RegionNodeSerializer(obj.region_node).data

    def get_model_info_detail(self, obj):
        return ModelInfoSerializer(obj.model_info).data

    def get_model_version_detail(self, obj):
        return ModelVersionSerializer(obj.model_version).data

    class Meta:
        model = TrainingRecord
        fields = '__all__'

class OperationLogSerializer(serializers.ModelSerializer):
    user = CommonUserSerializer(read_only=True)
    class Meta:
        model = OperationLog
        exclude = ['response_body']

class ModelInferenceLogSerializer(serializers.ModelSerializer):
    model_version_detail = serializers.SerializerMethodField()
    edge_node_detail = serializers.SerializerMethodField()
    created_by_detail = CommonUserSerializer(source='created_by', read_only=True)

    def get_model_version_detail(self, obj):
        return ModelVersionSerializer(obj.model_version).data

    def get_edge_node_detail(self, obj):
        if obj.edge_node:
            return EdgeNodeSerializer(obj.edge_node).data
        return None

    class Meta:
        model = ModelInferenceLog
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at', 'created_by']


class FederatedTrainingLogSerializer(serializers.ModelSerializer):
    task_detail = serializers.SerializerMethodField()
    region_node_detail = serializers.SerializerMethodField()
    
    def get_task_detail(self, obj):
        if obj.task:
            return {
                'id': obj.task.id,
                'name': obj.task.name,
                'rounds': obj.task.rounds,
                'status': obj.task.status,
            }
        return None
    
    def get_region_node_detail(self, obj):
        if obj.region_node:
            return RegionNodeSerializer(obj.region_node).data
        return None
    
    class Meta:
        model = FederatedTrainingLog
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at']
