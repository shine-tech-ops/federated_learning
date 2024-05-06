# DjangoBase


#### 1. 只启用本地开发环境

1. 克隆或下载项目代码
2. 进入项目目录，创建虚拟环境`python -m venv env`，并激活虚拟环境
3. 更新pip到最新版本`python -m pip install --upgrade pip`
4. 安装依赖`pip install -r requirements.txt`
5. 新建数据库: backend
6. 修改 backend/settings.py 中 DATABASES 的配置为实际的连接信息, 数据库名, 用户名, 密码, HOST及端口号
```json
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "backend",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "HOST": "127.0.0.1",
        "PORT": "5432",
    }
}

```
7. 数据库初始化`python manage.py migrate`
8. 更新数据库及表字段的注释`python manage.py migratecomment`
9. 运行服务`python manage.py runserver 0.0.0.0:8000`


#### 2. 使用docker方式运行
 
1. 克隆或下载项目代码
2. 进入项目目录, 执行`docker-compose up -d`


#### 3. 主要技术栈
> [初始化时使用的版本，理论上前后几个版本都可以]
- Python >= 3.12
- Django >= 5.0.6
- djangorestframework >= 3.12


#### 4. 基础项目集成的功能

- jwt 登录验证
- 跨域配置
- Django 日志配置
- 数据库字段注释功能
- 代码格式化插件 black  # https://black.readthedocs.io/en/stable/


#### 5. 其它可能使用的插件或功能

1. 多租户功能，参考：django-tenants  # https://github.com/django-tenants/django-tenants/tree/master


#### 6. 系统初始化后内置账号

- 超级管理员账号：superadmin/superadmin
- 管理员账号：admin/admin

#### 7. 已实现接口

- /api/v1/account/login/         # 登录接口
- /api/v1/account/permission/    # 权限接口
- /api/v1/account/user_role/     # 用户角色接口
- /api/v1/account/user/          # 用户接口
- /api/v1/account/current_user/  # 当前用户信息接口
- /api/v1/account/token_refresh/ # token刷新接口
