from django.shortcuts import render
from .models import Food
from django.http import HttpResponseRedirect
from django.contrib import auth
import time


def front(request):
    return render(request,'food/front.html')  # template 위치
