# -*- coding: utf-8 -*-

from django.urls import path
from . import views

urlpatterns = [
    path("login/", views.login, name="login"), # 로그인 페이지
    path("manage/", views.manage, name="manage"), # 관리 페이지
    path("logout/", views.logout, name="logout"), # 로그아웃
]
