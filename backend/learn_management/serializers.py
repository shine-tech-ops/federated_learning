from rest_framework import serializers
from .models import FederatedTask, SystemConfig, ModelVersion, ModelInfo

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
