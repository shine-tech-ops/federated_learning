import logging
import operator
import json
from pathlib import Path
from functools import reduce
from django.contrib.auth.hashers import make_password
from django.db.models import Q
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import BasePermission
from rest_framework.views import APIView
from .models import Permission, Role, AuthUserExtend, UserRole
from .serializers import (
    PermissionSerializer,
    RoleSerializer,
    RolePermission,
    UserSerializer,
)
from rest_framework_simplejwt.views import TokenObtainPairView


class PassAuthenticatedPermission(BasePermission):
    """
    本自定义auth类旨在开启验证后, 对个别接口需要跳过验证时使用
    """

    def has_permission(self, request, view):
        return True


logger = logging.getLogger(__name__)


class PermissionView(GenericAPIView):
    """
    权限视图：
    1. GET 获取权限列表；
    2. POST 添加新的权限
    """

    queryset = Permission.objects.all()
    serializer_class = PermissionSerializer

    def get_children(self, parent):
        queryset = self.get_queryset().filter(parent=parent)
        serializer = self.get_serializer(instance=queryset, many=True)
        return Response(serializer.data).data

    def get(self, request, *args, **kwargs):
        name_en = request.query_params.get("name_en")
        queryset = self.get_queryset()
        if name_en:
            queryset = queryset.filter(name_en=name_en)
        queryset = queryset.filter(parent=0)
        serializer = self.get_serializer(instance=queryset, many=True)
        for serializer_enum in serializer.data:
            parent_id = serializer_enum["id"]
            serializer_enum["children"] = self.get_children(parent_id)
            for child in serializer_enum["children"]:
                child["children"] = self.get_children(child["id"])
        data = [
            {
                "id": 0,
                "name_en": "all",
                "name_zh": "全部",
                "children": serializer.data,
            }
        ]
        ret_data = {
            "code": status.HTTP_200_OK,
            "msg": "success",
            "data": data,
        }
        return Response(ret_data)

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors)


class RoleView(GenericAPIView):
    """
    角色视图
    1. GET 获取角色列表
    2. POST 创建新的角色及对应的权限
    3. PUT 更新角色权限
    4. DELETE 删除角色
    """

    queryset = Role.objects.all()
    serializer_class = RoleSerializer

    def update_role_permission(self, request, *args, **kwargs):
        role_id = request.data.get("role_id")
        permission_list = request.data.get("permission")
        role_obj = self.get_queryset().get(id=role_id)
        role_permisson_obj = RolePermission.objects.filter(role=role_id)
        if role_permisson_obj.exists():
            role_permisson_obj.delete()
        for permission in permission_list:
            permission_id = permission["id"]
            permission_obj = Permission.objects.get(id=permission_id)
            RolePermission.objects.create(
                role=role_obj,
                permission=permission_obj,
            )

    def get(self, request, *args, **kwargs):
        role_id = request.query_params.get("id")
        name = request.query_params.get("name")
        queryset = self.get_queryset()
        if role_id:
            queryset = queryset.filter(id=role_id)

        if name:
            queryset = queryset.filter(name=name)

        serializer = self.get_serializer(instance=queryset, many=True)
        ret_data = {
            "code": status.HTTP_200_OK,
            "msg": "success",
            "data": serializer.data,
        }
        return Response(ret_data)

    def post(self, request, *args, **kwargs):
        ret_data = {}
        serializer = self.get_serializer(data=request.data)
        try:
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                request.data["role_id"] = serializer.data["id"]
                self.update_role_permission(request)
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "success",
                    "data": [serializer.data],
                }
        except Exception as e:
            ret_data = {
                "code": status.HTTP_406_NOT_ACCEPTABLE,
                "msg": "error",
                "data": str(e),
            }
        return Response(ret_data)

    def put(self, request, *args, **kwargs):
        ret_data = {}
        role_id = request.data.get("id")
        try:
            role_obj = self.get_queryset().get(id=role_id)
            serializer = self.get_serializer(
                instance=role_obj, data=request.data, partial=True
            )
            if serializer.is_valid(raise_exception=True):
                request.data["role_id"] = role_id
                self.update_role_permission(request)
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "success",
                    "data": [serializer.data],
                }
        except Exception as e:
            ret_data = {
                "code": status.HTTP_406_NOT_ACCEPTABLE,
                "msg": "error",
                "data": str(e),
            }
        return Response(ret_data)

    def delete(self, request, *args, **kwargs):
        role_id = request.data.get("id")
        role_obj = self.get_queryset().filter(id=role_id)
        if role_obj.exists():
            role_obj.delete()
            ret_data = {"code": status.HTTP_200_OK, "msg": "success", "data": []}
        else:
            ret_data = {
                "code": status.HTTP_404_NOT_FOUND,
                "msg": "error",
                "data": "The target record not exists.",
            }
        return Response(ret_data)


class AuthUserExtendView(GenericAPIView):
    """
    用户视图
    1. GET 获取用户列表
    2. POST 创建新的用户
    3. PUT 更新用户信息
    4. DELETE 删除用户
    """

    queryset = AuthUserExtend.objects.all()
    serializer_class = UserSerializer

    def update_user_role(self, request, *args, **kwargs):
        user_id = request.data.get("id")
        role_list = request.data.get("role")
        user_obj = self.get_queryset().get(id=user_id)
        user_role_obj = UserRole.objects.filter(user=user_id)
        if user_role_obj.exists():
            user_role_obj.delete()
        for role in role_list:
            role_id = role["id"]
            role_obj = Role.objects.get(id=role_id)
            UserRole.objects.create(
                user=user_obj,
                role=role_obj,
            )

    def get(self, request, *args, **kwargs):
        user_id = request.query_params.get("id")
        name = request.query_params.get("name")
        mobile = request.query_params.get("mobile")
        email = request.query_params.get("email")
        is_admin = request.query_params.get("is_admin")

        queryset = self.get_queryset()
        if user_id:
            queryset = queryset.filter(id=user_id)

        params = []
        if name:
            params.append(Q(name=name))

        if mobile:
            params.append(Q(mobile=mobile))

        if email:
            params.append(Q(email=email))

        if is_admin:
            params.append(Q(is_admin=is_admin))

        if params:
            queryset = queryset.filter(reduce(operator.and_, params))

        serializer = self.get_serializer(instance=queryset, many=True)
        ret_data = {
            "code": status.HTTP_200_OK,
            "msg": "success",
            "data": serializer.data,
        }
        return Response(ret_data)

    def post(self, request, *args, **kwargs):
        ret_data = {}
        request.data["password"] = make_password(request.data.get("password"))

        serializer = self.get_serializer(data=request.data)

        try:
            if serializer.is_valid(raise_exception=True):
                serializer.save()
                request.data["id"] = serializer.data["id"]
                self.update_user_role(request)
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "success",
                    "data": [serializer.data],
                }
        except Exception as e:
            ret_data = {
                "code": status.HTTP_406_NOT_ACCEPTABLE,
                "msg": "error",
                "data": str(e),
            }
        return Response(ret_data)

    def put(self, request, *args, **kwargs):
        ret_data = {}
        user_id = request.data.get("id")
        user_pw = request.data.get("password")
        role = request.data.get("role")
        if user_pw:
            request.data["password"] = make_password(request.data.get("password"))

        try:
            instance = AuthUserExtend.objects.get(id=user_id)

            serializer = self.get_serializer(
                instance=instance, data=request.data, partial=True
            )
            if serializer.is_valid(raise_exception=True):
                if role:
                    self.update_user_role(request)
                serializer.save()
                ret_data = {
                    "code": status.HTTP_200_OK,
                    "msg": "success",
                    "data": [serializer.data],
                }
        except Exception as e:
            ret_data = {
                "code": status.HTTP_406_NOT_ACCEPTABLE,
                "msg": "error",
                "data": str(e),
            }
        return Response(ret_data)

    def delete(self, request, *args, **kwargs):
        user_id = request.data.get("id")
        user_obj = None
        if user_id:
            user_obj = self.get_queryset().filter(id=user_id, is_superuser=0)

        if user_obj.exists():
            user_obj.delete()
            ret_data = {"code": status.HTTP_200_OK, "msg": "success", "data": []}
        else:
            ret_data = {
                "code": status.HTTP_404_NOT_FOUND,
                "msg": "error",
                "data": "The target record not exists.",
            }
        return Response(ret_data)


class CurrentUserView(GenericAPIView):
    serializer_class = UserSerializer

    def get(self, request, *args, **kwargs):
        user = request.user
        if not user:
            ret_data = {
                "code": status.HTTP_404_NOT_FOUND,
                "msg": "error",
                "data": "The target record not found.",
            }
        else:
            queryset = AuthUserExtend.objects.filter(id=user.id)
            serializer = self.get_serializer(instance=queryset, many=True)
            ret_data = {
                "code": status.HTTP_200_OK,
                "msg": "success",
                "data": serializer.data,
            }
        return Response(ret_data)


class LoginView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        """
        重写登录接口，自定义登录失败时的响应
        """
        serializer = self.get_serializer(data=request.data)
        try:
            if serializer.is_valid(raise_exception=True):
                ret_data = serializer.validated_data
                return Response(ret_data, status=status.HTTP_200_OK)
        except Exception as e:
            ret_data = {
                "code": 400,
                "msg": "用户名或密码错误，请检查后重新登录",
                "data": [],
            }
            return Response(ret_data, status=status.HTTP_400_BAD_REQUEST)
