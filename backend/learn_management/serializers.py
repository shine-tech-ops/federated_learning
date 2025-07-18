from rest_framework import serializers
from .models import FederatedTask, SystemConfig, ModelVersion, ModelInfo, RegionNode, EdgeNode, TrainingRecord, OperationLog
from user.serializers import CommonUserSerializer

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
        return ret
    class Meta:
        model = FederatedTask
        fields = "__all__"



class EdgeNodeSerializer(serializers.ModelSerializer):
    def to_representation(self, obj):
        ret = super(EdgeNodeSerializer, self).to_representation(obj)
        ret['region_node_detail'] = RegionNodeSerializer(obj.region_node).data
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
