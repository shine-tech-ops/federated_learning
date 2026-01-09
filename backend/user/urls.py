from django.urls import path
from .views import (
    PermissionView,
    RoleView,
    AuthUserExtendView,
    CurrentUserView,
    LoginView,
)
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path("permission/", PermissionView.as_view()),
    path("user_role/", RoleView.as_view()),
    path("user/", AuthUserExtendView.as_view()),
    path("current_user/", CurrentUserView.as_view()),
    path("login/", LoginView.as_view(), name="login"),
    path("token_refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]
