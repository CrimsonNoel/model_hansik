from django.shortcuts import render
from .models import Board, Comment
from django.http import HttpResponseRedirect,HttpResponse
from django.utils import timezone
from django.core.paginator import Paginator
import math


def write(request):
    if request.method != "POST":
       return render(request,"board/boardForm.html")
    else :
        # 파일 업로드
        try :
            filename = request.FILES["file1"].name # 업로드 파일의 이름
            handle_upload(request.FILES["file1"]) # 업로드 파일이름으로 파일 저장
            # request.FILES["file1"] : file1 의 내용
        except : 
            filename = ""            
            # timezone.now() : 현재시간
            # num 컬럼은 설정하지 않음 : num은 기본키, auto_increments로 설정됨.
        b = Board(name = request.POST["name"],
                  pass1 = request.POST["pass"],
                  subject = request.POST["subject"],
                  content = request.POST["content"],
                  regdate = timezone.now(),
                  readcnt = 0,
                  file1 = filename
                  )
        b.save() # insert
        return HttpResponseRedirect("/board/list")


def list(request):
    # Get 방식의 pageNum 파라미터 조회.
    # pageNum 파라미터값이 없는 경우 1로 리턴
    pageNum= int(request.GET.get("pageNum",1))
    # Board.objects.all() : Board 데이터 전부 리턴.
    # order_by("-num") : num 컬럼의 내림차순 정렬
    all_boards = Board.objects.all().order_by("-num")
    pageview = 3
    paginator = Paginator(all_boards,pageview) # Paginator ( 불러올 queryset , 한페이지당 볼 페이지수)
    board_list = paginator.get_page(pageNum)
    listcount = Board.objects.count() # 게시물 등록선수
       
    # pageing
    pagestart = (pageNum-1)//pageview*pageview + 1
    pageend = pagestart + pageview -1
    maxpage = math.trunc(listcount/pageview)
    if listcount % pageview != 0:
        maxpage = maxpage +1
        
    content = {'boardlist':board_list, 'listcount':listcount,'pageNum':pageNum,'pageview':pageview,\
               'pagestart':pagestart,'maxpage':maxpage,'pageend':pageend} # dictionary 형태로 보냄
    print(content)

    return render(request, "board/boardList.html",content)

def comment(request,num):
    board = Board.objects.get(num = num) # num에 해당하는 게시물 데이터 조회
    board.readcnt +=1
    board.save() #조회수 증가 수정
    commentlist = Comment.objects.filter(num = num).order_by("-ser") # num에 해당하는 게시물 데이터 조회
    #query = (f"SELECT num, regdate FROM board_comment WHERE num = {num} ORDER BY ser DESC;",'num')
    #comments = Comment.objects.raw(query)
    # regdate 입력시간 출력 포기
    return render(request,"board/boardComment.html",{'b':board,'commentlist':commentlist})

def commentpro(request,comment,num):
    com = Comment(num = num, content=comment,regdate = timezone.now())
    com.save() # insert
    return HttpResponse(com)

def update(request,num):
    board = Board.objects.get(num=num)
    pass1 = board.pass1
    if request.method != "POST":
      return render(request,"board/boardUpdateForm.html",{"board":board})
    else :
       # 파일 업로드
       try :
               filename = request.FILES["file2"].name
               handle_upload(request.FILES["file2"])
              
       except : 
           filename = board.file1            

       if pass1 == request.POST["pass"] :
            b = Board(num=num,
                name = board.name,
              pass1 = request.POST["pass"],
              subject = request.POST["subject"],
              content = request.POST["content"],
              regdate = timezone.now(),
              readcnt = board.readcnt,
              file1 = filename  )
            b.save() # insert
            return HttpResponseRedirect("/board/comment/"+str(num)+"/",{"b":board})
       else:
            context = {"msg" : "비밀번호가 틀립니다.","url":"/board/update/"+str(num)+"/"}
            return render(request, "alert.html",context)

def handle_upload(f) : # file/board 폴더 생성. # f : 업로드된 이미지파일의 내용
    with open("file/images/"+f.name,"wb") as dest : # close 생략가능
        # f.chunks() : 이진파일 읽기
        for ch in f.chunks() :
            dest.write(ch)

def delete(request,num):
    b = Board.objects.get(num=num)
    if request.method !="POST" :
         return render(request,'board/boardDeleteForm.html',{"b":b})
    else:
        if b.pass1 == request.POST["pass"]:
           b.delete()
           context={"msg":"삭제 완료","url":"/board/list/"}
           return render(request, "alert.html",context)
        else :
            context={"msg":"비밀번호 오류","url":"/board/delete/"+str(num)+"/"}
            return render(request, "alert.html",context)











