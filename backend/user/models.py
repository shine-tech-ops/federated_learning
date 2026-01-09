from django.db import models
from django.contrib.auth.models import AbstractBaseUser, UserManager


class Permission(models.Model):
    id = models.AutoField(primary_key=True)
    name_en = models.CharField(
        unique=True, max_length=80, blank=False, null=False, verbose_name="权限英文名称"
    )
    name_zh = models.CharField(
        max_length=80, blank=False, null=False, verbose_name="权限中文名称"
    )
    parent = models.IntegerField(blank=True, null=True, verbose_name="父级权限ID")

    class Meta:
        db_table = "permission"
        db_table_comment = "权限表"
        verbose_name = "权限表"
        ordering = ["id"]


class Role(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True, verbose_name="角色名")
    created_at = models.DateTimeField(blank=True, null=True, auto_now_add=True)
    updated_at = models.DateTimeField(blank=True, null=True, auto_now=True)

    class Meta:
        db_table = "role"
        db_table_comment = "角色表"
        verbose_name = "角色表"
        ordering = ["id"]


class RolePermission(models.Model):
    id = models.AutoField(primary_key=True)
    role = models.ForeignKey(
        Role,
        related_name="permission",
        on_delete=models.CASCADE,
        verbose_name="角色ID",
    )
    permission = models.ForeignKey(
        Permission, on_delete=models.CASCADE, verbose_name="权限ID"
    )

    class Meta:
        db_table = "role_permission"
        db_table_comment = "角色权限关联表"
        verbose_name = "角色权限关联表"
        ordering = ["id"]


class AuthUserExtend(AbstractBaseUser):
    id = models.AutoField(primary_key=True)
    name = models.CharField(
        unique=True, max_length=20, blank=True, null=True, verbose_name="用户名"
    )
    mobile = models.CharField(
        unique=False, max_length=20, blank=True, null=True, verbose_name="手机"
    )
    email = models.EmailField(unique=False, blank=True, null=True, verbose_name="邮箱")
    password = models.CharField(max_length=100, verbose_name="密码")
    is_active = models.BooleanField(
        default=True, blank=True, null=False, verbose_name="是否启用"
    )
    is_superuser = models.BooleanField(
        default=False, blank=True, null=False, verbose_name="超级管理员"
    )
    is_admin = models.BooleanField(
        default=False, blank=True, null=False, verbose_name="管理员"
    )
    last_login = models.DateTimeField(
        blank=True, null=True, verbose_name="最近登录时间"
    )
    date_joined = models.DateTimeField(
        blank=True, null=True, auto_now_add=True, verbose_name="创建时间"
    )
    objects = UserManager()

    USERNAME_FIELD = "name"
    REQUIRED_FIELDS = ["username", "password"]

    class Meta:
        db_table = "auth_user_extend"
        db_table_comment = "用户信息表"
        verbose_name = "用户信息表"
        ordering = ["id"]


class UserRole(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(
        AuthUserExtend,
        related_name="role",
        on_delete=models.CASCADE,
        verbose_name="用户",
    )
    role = models.ForeignKey(Role, on_delete=models.CASCADE, verbose_name="角色ID")

    class Meta:
        db_table = "user_role"
        db_table_comment = "用户角色关联表"
        verbose_name = "用户角色关联表"
        ordering = ["id"]
