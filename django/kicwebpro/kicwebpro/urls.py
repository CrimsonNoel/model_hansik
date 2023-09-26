"""
URL configuration for kicwebpro project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path("member/", include("member.urls")), # = namespace   # projectname/ ' member/ '
    path("board/", include("board.urls")),
]


# resourse file
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


''' # 7 ..
urlpatterns = [
    path("admin/", admin.site.urls),
]
'''
# anaconda prompt 가서
# python manage.py startapp member 하면 폴더생성
# board 같은것도 마찬가지