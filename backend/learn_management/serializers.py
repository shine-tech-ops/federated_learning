from rest_framework import serializers
from .models import FederatedTask, SystemConfig, ModelVersion, ModelInfo, RegionNode, EdgeNode, OperationLog
from user.serializers import CommonUserSerializer


class FederatedTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = FederatedTask
        fields = "__all__"


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