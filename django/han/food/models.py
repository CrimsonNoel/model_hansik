from django.db import models

class Food(models.Model):
    # num 값이 없으면 자동증가함 -> auto increaments
    num = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30)
    #subject = models.CharField(max_length=100)
    #content = models.CharField(max_length=4000)    
    regdate = models.DateTimeField(auto_now_add=True,blank=True) # 오늘날짜적용
    #regdate = models.DateTimeField(null=True) # null 허용
    readcnt = models.IntegerField(default=0)
    file1 = models.CharField(max_length=300)
    
    # def __repr__(self) : 같은 함수
    def __str__(self) :
        return str(self.num)+":"+self.name
    
