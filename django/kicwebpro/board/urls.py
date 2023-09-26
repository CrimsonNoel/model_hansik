# -*- coding: utf-8 -*

"""
Created on Fri Sep  8 10:37:23 2023

@author: KITCOOP
"""

from django.urls import path
from . import views

# projectname/member/ ' index/ '
# mvc처럼 덧붙여서 나가면된다

urlpatterns = [
    path("write/", views.write,name = "write"),
    path("list/", views.list,name = "list"),
    path("comment/<int:num>/", views.comment,name = "comment"),
    path("commentpro/<str:comment>/<int:num>/", views.commentpro,name = "commentpro"),
    path("update/<int:num>/", views.update,name = "update"),
    path("delete/<int:num>/", views.delete,name="delete"),
    ]                                          
                                                 