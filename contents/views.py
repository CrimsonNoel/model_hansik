from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.utils import timezone
from foodmodels.load import load_food_model
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import tensorflow as tf
from PIL import Image
import os
from .models import Food, Feedback
import numpy as np

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
    
    
    categories = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083']
    categories_index = ['갈치구이','고등어구이', '더덕구이', '장어구이', '조개구이', '조기구이', '황태구이', '훈제오리', '계란국', '떡국_만두국',
     '무국', '미역국', '북엇국', '시래기국', '육개장', '콩나물국', '콩자반', '갓김치', '깍두기', '무생채', '배추김치', '백김치',
     '부추김치', '열무김치', '오이소박이', '총각김치', '파김치', '가지볶음', '고사리나물', '미역줄기볶음', '숙주나물', '시금치나물',
     '애호박볶음', '수제비', '열무국수', '잔치국수', '꽈리고추무침', '도라지무침', '도토리묵', '잡채', '콩나물무침', '김치볶음밥',
     '비빔밥', '새우볶음밥', '알밥', '감자채볶음', '건새우볶음', '고추장진미채볶음', '두부김치', '멸치볶음', '어묵볶음', '오징어채볶음',
     '주꾸미볶음', '깻잎장아찌', '감자전', '김치전', '동그랑땡', '생선전', '파전', '호박전', '갈치조림', '감자조림', '고등어조림',
     '꽁치조림', '두부조림', '땅콩조림', '연근조림', '우엉조림', '코다리조림', '전복죽', '호박죽', '닭계장', '동태찌개', '순두부찌개',
     '계란찜', '김치찜', '해물찜', '갈비탕', '감자탕', '곰탕_설렁탕', '매운탕', '삼계탕', '추어탕'] 
    
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
            fs = FileSystemStorage()
            filename = uploaded_file.name.split('.')[0]+'.jpg'
            file_save = fs.save(filename, uploaded_file)
            #uploaded_file_image = process_image(uploaded_file)
            #input_file_path = os.path.join(settings.MEDIA_ROOT, filename)
            #uploaded_file_image.save(input_file_path)
            return HttpResponseRedirect("/contents/loading/")
    
    return render(request, 'contents/start.html')

def loading(request):
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
    print(category)
    context = {"category": category}
    
    return render(request, 'contents/result.html', context)

def result(request):
    
# =============================================================================
#     # 피드백 아니요시 파일 삭제할부분 
#     try:
#         os.remove(input_file_path)
#         print("삭제")
#     except:
#         print("안돼")
#         pass
# =============================================================================
    return render(request, 'contents/result.html')

def wrong(request):
    return render(request, 'contents/wrong.html')