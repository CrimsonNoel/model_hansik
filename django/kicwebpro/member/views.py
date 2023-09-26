# Create your views here.


from django.shortcuts import render
from .models import Member
from django.http import HttpResponseRedirect
from django.contrib import auth
import time


def index(request):
    return render(request,'member/index.html')  # template 위치

def join(request):
    if request.method != 'POST' :
        return render(request,'member/joinForm.html') # template/member/joinForm
    else :
        member = Member(id = request.POST["id"],
                        pass1 = request.POST["pass"],
                        name = request.POST["name"],
                        gender =request.POST["gender"],
                        tel=request.POST["tel"],
                        email=request.POST["email"],
                        picture = request.POST["picture"])
        member.save()  # insert,update(같은 id) 문장 실행. query?
        return HttpResponseRedirect("/member/login")
    
def login(request)    :
    if request.method !="POST"    :
        return render(request, 'member/loginForm.html')
    else :
        id1 = request.POST["id"]
        pass1 = request.POST["pass"]
        try :
            # 입력된 id값으로 Member객체에서 조회
            member = Member.objects.get(id=id1) # select 문장 실행
        except : # db에 아이디 정보가 없을때
            context = {"msg" : "아이디를 확인하세요.","url":"/member/login"}
            return render(request, "alert.html",context)
        else : # 정상작동인 경우, 아이디 조회가 된 경우  그다음
               # member.pass1 : db에 등록된 비밀번호
               # pass1 : 입력된 비밀번호
             if member.pass1 == pass1 : # 비밀번호 확인 일치하면
                 request.session["id"] = id1
                 time.sleep(1)
                 print("2: ",request.session.session_key)
                 return HttpResponseRedirect("/member/index")
             else : # 비밀범호 틀리면
               context = {"msg" : "비밀번호가 틀립니다.","url":"/member/login"}
               return render(request, "alert.html",context)
           
            
def info(request,id):
    member = Member.objects.get(id=id)
    return render(request,'member/memberInfo.html',{"mem":member})           
            
def update(request,id):
    member = Member.objects.get(id=id)
    if request.method !="POST"    :
        return render(request,'member/memberUpdateForm.html',{"mem":member})
    
    else : 
        # 비밀번호 검증.
        # 비밀번호 오류시 비밀번호 오류 메세지, update.html 페이지 출력
        # member.pass1 : db에 등록된 비밀번호
        # request.POST["pass"] : 입력된 비밀번호
        if request.POST["pass"] == member.pass1:
            member = Member(id = id,
                            pass1 = request.POST["pass"],
                            name = request.POST["name"],
                            gender =request.POST["gender"],
                            tel=request.POST["tel"],
                            email=request.POST["email"],
                            picture = request.POST["picture"])
        # id값 존재시 update 없으면 insert 자동으로 해줌(sqlite3)
            member.save()
            return HttpResponseRedirect("/member/info/"+id+"/")
        else:
            context = {"msg" : "비밀번호가 틀립니다.","url":"/member/update/"+id+"/"}
            return render(request, "alert.html",context)
    
    
def delete(request,id):
    login=request.session["id"]
    if request.method !="POST" :
         return render(request,'member/memberDeleteForm.html',{"id":id})
    elif login == "admin" :
        context={"msg":"관리자 계정은 탈퇴가 안됩니다","url":"/member/info/"+id+"/"}
        return render(request, "alert.html",context)
    else:
        member = Member.objects.get(id=id)
        if member.pass1 == request.POST["pass"]:
           member.delete()
           auth.logout(request)
           context={"msg":"탈퇴완료","url":"/member/login/"}
           return render(request, "alert.html",context)
        else :
            context={"msg":"비밀번호 오류","url":"/member/delete/"+id+"/"}
            return render(request, "alert.html",context)
           
def passchg(request):
    login=request.session["id"]
    if request.method !="POST" :
       return render(request,'member/memberPassForm.html')
    else:
        member = Member.objects.get(id=login)
        if member.pass1 == request.POST["pass"] : # 비밀번호 비교
           member.pass1 = request.POST["chgpass1"] # 변경할 비밀번호로 교체/수정
           member.save() # 변경내용 저장
           context={"msg":"비밀번호 수정 완료","url":"/member/info/"+login+"/"}
           return render(request, "alert.html",context)
        else: # 비밀번호 오류
            context={"msg":"비밀번호 오류","url":"member/passchg"}
            return render(request, "alert.html",context)

def logout(request):
        auth.logout(request)
        return HttpResponseRedirect("/member/login/")  

def list(request):
    try:
        login = request.session["id"]
    except:
        context={"msg":"로그인 하세요","url":"/member/login/"}
        return render(request, "alert.html",context)
    else:
        if login != "admin":
            context={"msg":"관리자만 가능합니다","url":"/member/index/"}
            return render(request, "alert.html",context)
        # mlist 요소 : Member 객체
        mlist = Member.objects.all() # 모든 데이터 리턴
    return render(request,'member/memberList.html',{"mlist":mlist})
               
def picture(request):
    if request.method !="POST" :
        return render(request,"member/pictureimgForm.html")              
    else :
        # request.FILES["picture"] : 업로드한 파일 객체
        # filename = 파일이름
        filename = request.FILES["picture"].name
        handle_upload(request.FILES["picture"])
        return render(request,"member/picturePro.html",{"filename":filename})       
               
def handle_upload(f) : # f : 업로드된 이미지파일의 내용
    with open("file/picture/"+f.name,"wb") as dest : # close 생략가능
        # f.chunks() : 이진파일 읽기
        for ch in f.chunks() :
            dest.write(ch)
               
               
               
               
               
               
               
               