from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.utils import timezone
from django.contrib import auth
from .models import Manager
from contents.models import Food, Feedback
import time

# Create your views here.
def login(request):
    if request.method != "POST":
        return render(request, 'management/login.html')
    else:
        id1 = request.POST["id"]
        pass1 = request.POST["pass"]
        try:
            manager = Manager.objects.get(id = id1) # select 문장 실행
        except: # db에 아이디 정보 X
            context = {"msg":"아이디 없음", "url":"/management/login"}
            return render(request, "alert.html", context)
        else: # db에 아이디 정보가 조회된 경우
            if manager.errcount <= 5: # 비밀번호 오류 횟수가 5회 이하인 경우
                if manager.pass1 == pass1 : # 비밀번호 일치
                    manager.errcount = 0
                    manager.save()
                    request.session["id"] = id1 # session 객체에 아이디 등록
                    time.sleep(1)
                    print("2:",request.session.session_key)
                    return HttpResponseRedirect("/management/manage")
                else: # 비밀번호 오류
                    manager.errcount = manager.errcount + 1
                    manager.save()
                    context = {"msg":"비밀번호 오류: "+str(manager.errcount)+"회, 5회 초과 시 로그인 불가"\
                               ,"url":"/management/login"}
                    return render(request,"alert.html",context)
            else: # 비밀번호 오류 횟수 10회 초과 시
                context = {"msg":"비밀번호 오류 횟수 5회 초과로 로그인 불가", "url":"/management/login"}
                return render(request,"alert.html",context)

def logout(request):
    auth.logout(request)
    return HttpResponseRedirect("/management/login/")


def manage(request):
    try :
        login = request.session["id"]
        manager = Manager.objects.get(id = login)
    except :
        context = {"msg":"관리자 로그인 필수","url":"/management/login"}
        return render(request, "alert.html", context)
    return render(request, 'management/manage.html', {"manager":manager})