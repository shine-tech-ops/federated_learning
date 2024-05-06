from rest_framework import serializers
from .models import (
    Permission,
    Role,
    RolePermission,
    UserRole,
    AuthUserExtend,
)


class PermissionSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        return Permission.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.name_en = validated_data.get("name_en", instance.name_en)
        instance.name_zh = validated_data.get("name_zh", instance.name_zh)
        instance.parent = validated_data.get("parent", instance.parent)
        instance.save()
        return instance

    class Meta:
        model = Permission
        fields = "__all__"


class RolePermissionSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        return RolePermission.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.role = validated_data.get("role", instance.role)
        instance.permission = validated_data.get("permission", instance.permission)
        instance.save()
        return instance

    def to_representation(self, value):
        ret_data = {
            "id": value.permission.id,
            "name_en": value.permission.name_en,
            "name_zh": value.permission.name_zh,
        }
        return ret_data

    class Meta:
        model = RolePermission
        fields = "__all__"


class RoleSerializer(serializers.ModelSerializer):
    permission = RolePermissionSerializer(many=True, read_only=True)

    def create(self, validated_data):
        return Role.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.name = validated_data.get("name", instance.name)
        instance.save()
        return instance

    class Meta:
        model = Role
        fields = "__all__"

    def to_representation(self, obj):
        ret_data = super(RoleSerializer, self).to_representation(obj)
        exclude_keys = ["created_at", "updated_at"]
        for key in exclude_keys:
            ret_data.pop(key, None)
        return ret_data


class UserRoleSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        return UserRole.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.user = validated_data.get("user", instance.user)
        instance.role = validated_data.get("role", instance.role)
        instance.save()
        return instance

    def to_representation(self, value):
        ret_data = {
            "id": value.role.id,
            "name": value.role.name,
        }
        return ret_data

    class Meta:
        model = UserRole
        fields = "__all__"


class UserSerializer(serializers.ModelSerializer):
    role = UserRoleSerializer(many=True, read_only=True)

    class Meta:
        model = AuthUserExtend
        fields = "__all__"

    def to_representation(self, obj):
        ret_data = super(UserSerializer, self).to_representation(obj)
        exclude_keys = ["password", "last_login", "date_joined"]
        for key in exclude_keys:
            ret_data.pop(key, None)
        # 添加角色及对应的权限
        raw_role_list = ret_data.pop("role", [])
        ret_data["role"] = []
        for role in raw_role_list:
            role_obj = Role.objects.filter(id=role["id"])
            role_serializer = RoleSerializer(role_obj, many=True, read_only=True)
            ret_data["role"] += role_serializer.data
        return ret_data
