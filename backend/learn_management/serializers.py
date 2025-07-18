from rest_framework import serializers
from .models import FederatedTask, SystemConfig, ModelVersion, ModelInfo, RegionNode, EdgeNode, OperationLog
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
    region = RegionNodeSerializer(read_only=True)
    class Meta:
        model = EdgeNode
        fields = '__all__'

class OperationLogSerializer(serializers.ModelSerializer):
    user = CommonUserSerializer(read_only=True)
    class Meta:
        model = OperationLog
        exclude = ['response_body']
