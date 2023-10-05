from django.db import models

# Create your models here.

class Manager(models.Model):
    id = models.CharField(max_length=20, primary_key=True)
    pass1 = models.CharField(max_length=20)
    name = models.CharField(max_length=20)
    errcount = models.IntegerField(default=0) # 10회 이상 오류 시 로그인 불가 처리

    def __str__(self):
        return self.id + " : " + self.name + " : " + self.pass1