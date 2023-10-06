from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.utils import timezone
from .models import Food, Feedback
from foodmodels.load import load_food_model
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import tensorflow as tf
from PIL import Image
import os
import numpy as np
from pathlib import Path
import shutil

def process_image(input_image):
    img = Image.open(input_image)
    img = img.convert("RGB")
    new_width, new_height = 200, 200
    img = img.resize((new_width, new_height))
    return img

def predict_image(last_file_name):
   food_model = load_food_model()
   input_file_path = os.path.join(settings.MEDIA_ROOT, last_file_name)
   img = tf.keras.utils.load_img(input_file_path)
   print("==type: ",type(img))
   img_array = tf.keras.utils.img_to_array(img)
   #img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
   image_shape = img_array.shape
   #img_array = img_array / 255.0
   print("img_array: ",img_array)
   predictions = food_model.predict(img_array)
   score = tf.nn.softmax(predictions[0])
    
   categories = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83']
   categories_index = ['갈치구이','고등어구이', '더덕구이', '장어구이', '조개구이', '조기구이', '황태구이', '훈제오리', '계란국', '떡국_만두국',
     '무국', '미역국', '북엇국', '시래기국', '육개장', '콩나물국', '콩자반', '갓김치', '깍두기', '무생채', '배추김치', '백김치',
     '부추김치', '열무김치', '오이소박이', '총각김치', '파김치', '가지볶음', '고사리나물', '미역줄기볶음', '숙주나물', '시금치나물',
     '애호박볶음', '수제비', '열무국수', '잔치국수', '꽈리고추무침', '도라지무침', '도토리묵', '잡채', '콩나물무침', '김치볶음밥',
     '비빔밥', '새우볶음밥', '알밥', '감자채볶음', '건새우볶음', '고추장진미채볶음', '두부김치', '멸치볶음', '어묵볶음', '오징어채볶음',
     '주꾸미볶음', '깻잎장아찌', '감자전', '김치전', '동그랑땡', '생선전', '파전', '호박전', '갈치조림', '감자조림', '고등어조림',
     '꽁치조림', '두부조림', '땅콩조림', '연근조림', '우엉조림', '코다리조림', '전복죽', '호박죽', '닭계장', '동태찌개', '순두부찌개',
     '계란찜', '김치찜', '해물찜', '갈비탕', '감자탕', '곰탕_설렁탕', '매운탕', '삼계탕', '추어탕']  # 카테고리 리스트
   category = categories_index[np.argmax(score)]
   print("shape:",image_shape)
   print("predictions: ",np.argmax(score),predictions[0])
   return category   # , max_score

def start(request):
    try:
        food_model = load_food_model()
        print(' ==model load==')
    except Exception as e:
        print('Failed to load food model:',str(e))
        
    if request.method == 'POST':
        uploaded_file = request.FILES.get('chooseFile')
        # 50,50,3
        if not uploaded_file:
            context = {"msg": '이미지가 업로드되지 않았습니다.', "url": "/contents/start/"}
            print('이미지가 업로드되지 않았습니다.')
            return render(request, 'alert.html', context)
        else:
            try:
                fs = FileSystemStorage()
                file_list = fs.listdir('')
                filename = uploaded_file.name.split('.')[0]+'.jpg'
                file_save = fs.save(filename, uploaded_file)
            except:
                pass
            file_path = Path(os.path.join(settings.MEDIA_ROOT, filename))
            print("100: ", file_path)
            food_dir = Path("kfoodpro/static/img/food")
            shutil.copy(file_path,food_dir)
            # result.html 에  static load 되어있으니 static/img/food 부분에 copy후에 copy이미지 업로드 하려했으나..
            
            ## 이미지 전처리 원할시 수정해야함
            #uploaded_file_image = process_image(uploaded_file)
            #input_file_path = os.path.join(settings.MEDIA_ROOT, filename)
            #uploaded_file_image.save(input_file_path)
            
            return HttpResponseRedirect("/contents/result/", filename)
    
    return render(request, 'contents/start.html')

def result(request):
    fs = FileSystemStorage()
    file_list = fs.listdir('')
    print('file_list: ',file_list)
    # 이미지 파일 확인작업
    if not file_list[1]:  # 파일이 업로드되지 않은 경우
        context = {"msg": '이미지가 업로드되지 않았습니다.', "url": "/contents/start/"}
        print('이미지가 업로드되지 않았습니다.')
        return render(request, 'alert.html', context)
    
    last_file_name = file_list[1][-1]  
    input_file_path = os.path.join(settings.MEDIA_ROOT, last_file_name)
    print('inputpath: ',input_file_path)
    try: # 가능한 이미지인지 확인
        img = Image.open(input_file_path)
    except Exception as e:
        context = {"msg": '지원하는 파일이 아닙니다', "url": "/contents/start/"}
        print('지원하는 파일이 아닙니다:', str(e))
        return render(request, 'alert.html', context)
    
    if not img:
        context = {"msg": '이미지가 업로드되지 않았습니다.', "url": "/contents/start/"}
        print('이미지가 업로드되지 않았습니다.')
        return render(request, 'alert.html', context)
    
    # predicted_category, max_score = predict_image(img)
    category= predict_image(last_file_name)
    food_dir = Path("kfoodpro/static/img/food")
    file_list = os.listdir(food_dir)
    foodimage = file_list[-1]
    print(category)
    # context = {"category": category}
    food = Food.objects.get(name=category)
    return render(request, 'contents/result.html', {"food":food, "foodimage": foodimage})

def wrong(request):
    return render(request, 'contents/wrong.html')