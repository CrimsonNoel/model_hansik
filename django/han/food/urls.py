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
    path("front/", views.front,name = "front")
    ]                                          
                                                 