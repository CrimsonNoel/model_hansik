# -*- coding: utf-8 -*

"""
Created on Fri Sep  8 10:37:23 2023

@author: KITCOOP
"""

from django.urls import path
from . import views
# from . import views  -> member.views

# projectname/member/ ' index/ '
# kicwebpro urls.py에서 member.urls로 보냇다. 
# mvc처럼 덧붙여서 나가면된다

urlpatterns = [
    path("index/", views.index,name = "index"), 
    path("join/", views.join,name="join"), 
    path("login/", views.login,name="login"),
    path("logout/", views.logout,name="logout"),
    path("info/<str:id>/", views.info,name="info"),
    path("update/<str:id>/", views.update,name="update"),
    path("delete/<str:id>/", views.delete,name="delete"),
    path("passchg/", views.passchg,name="passchg"),
    path("list/", views.list,name="list"),
    path("picture/", views.picture,name="picture"),
    ]                                          
                                                 