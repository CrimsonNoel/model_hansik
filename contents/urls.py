# -*- coding: utf-8 -*-

from django.urls import path
from . import views

urlpatterns = [
    path("start/", views.start, name="start"), # 시작 페이지
    path("loading/", views.loading, name="loading"), # 로딩 페이지
    path("result/", views.result, name="result"), # 결과 페이지
    path("wrong/", views.wrong, name="wrong"), # 결과 오류 페이지
]
