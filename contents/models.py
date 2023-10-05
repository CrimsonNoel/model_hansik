from django.db import models

# Create your models here.

class Food(models.Model):
    num = models.IntegerField(primary_key=True) # num
    name = models.CharField(max_length=100) # 이름
    efficacy = models.CharField(max_length=100) # 효능
    origin = models.CharField(max_length=300) # 유래
    count = models.IntegerField(default=0) # 검색 횟수
    
    def __str__(self):
        return self.name + " : " + str(self.num)
    

class Feedback(models.Model):
    num = models.BigAutoField(primary_key=True)
    fnum = models.ForeignKey("Food", related_name="food", 
            on_delete=models.CASCADE, db_column="fnum") # food num (외래키)
    answer = models.IntegerField(default=0) # 0: 긍정(default), 1: 부정, 2: 관리자 승인완료
    image1 = models.CharField(max_length=300) # 업로드 이미지 경로
    regdate = models.DateTimeField(auto_now_add=True, blank=True)
    
    def __str__(self):
        return self.num + " : " + self.fnum + " : " + self.answer