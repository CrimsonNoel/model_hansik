from django.shortcuts import render
from .models import Food
from django.http import HttpResponseRedirect
from django.contrib import auth
import time


def front(request):
    return render(request,'food/front.html')  # template 위치
    '''
    #if request.method != "POST":
    else:
        # 파일받아서 업로드?
        # 검증과정 거쳐야할듯
        #
        return HttpResponseRedirect("food/result.html")

    '''