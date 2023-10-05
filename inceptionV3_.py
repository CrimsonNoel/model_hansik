# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import LeakyReLU, Input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
from keras import metrics
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pathlib


labels =['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083']
data_dir =pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb4/")
data_dir_test = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/test_rgb4/")

data_dir
data_dir_test

img_height = 299
img_width = 299  # InceptionV3 모델의 입력 크기
batch_size = 128
num_classes = len(labels)  # 분류할 클래스 수

# 조기 학습 종료 콜백 정의
early_stopping = EarlyStopping(
    monitor='val_loss',  # 모니터링할 지표 선택 (검증 손실을 기준으로)
    patience=10,  # 손실이 더 이상 감소하지 않아도 학습을 최대 10번까지 더 진행
    verbose=1,  # 로그 출력 설정 (1: 출력, 0: 미출력)
    restore_best_weights=True  # 조기 종료 시 가장 좋은 가중치로 복원
)


# 데이터 augmentation 설정 (선택사항)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.5),
])

# 데이터 로딩 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 이미지 픽셀 값을 [0, 1] 범위로 조정
    preprocessing_function=data_augmentation,
    validation_split=0.2  # 검증 데이터 분할 비율
)


# 훈련 데이터셋 생성
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # 교차검증을 위해 필요하다
    subset="validation",
    seed=123,
    #image_size=(img_height, img_width),
    batch_size=batch_size)

# 검증 데이터셋 생성
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    validation_split=0.2,
    subset="validation",
    seed=123,
    #image_size=(img_height, img_width), # default
    batch_size=batch_size
    )

# InceptionV3 모델 로드
inception_model = tf.keras.applications.InceptionV3(
    include_top=False,  # Fully Connected Layer를 포함하지 않음
    weights="imagenet",  # ImageNet 데이터셋으로 사전 훈련된 가중치 사용
    #input_shape=(img_height, img_width, 3),  # 입력 이미지 크기 설정
)

# 사용자 정의 출력 레이어 추가
output_layer = tf.keras.layers.GlobalAveragePooling2D()(inception_model.output)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(output_layer) # L2 정규화 추가
# output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(output_layer) # 원-핫 인코딩

# 모델 정의
model = tf.keras.Model(inputs=inception_model.input, outputs=output_layer)


# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 모델 요약 정보 출력
inception_model.summary()

# 모델 학습
epochs=30
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    #callbacks=[early_stopping]  # 조기 종료 콜백
)


model.save("C:/Users/KITCOOP/kicpython/hansik",overwrite=True)

# 모델 불러오기
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

loaded_model = tf.keras.models.load_model("C:/Users/KITCOOP/kicpython/hansik/models/2")
loaded_model.summary()
loaded_model
# 모델의 가중치 가져오기
weights = loaded_model.get_weights()
# 가중치 확인
for i, layer_weights in enumerate(weights):
    print(f"Layer {i} weights shape: {layer_weights.shape}")

# 추후 새로운 폴더 만들어서 검증할 부분    
X=[]
categories = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083']
categories_index = ['갈치구이','고등어구이', '더덕구이', '장어구이', '조개구이', '조기구이', '황태구이', '훈제오리', '계란국', '떡국_만두국',
 '무국', '미역국', '북엇국', '시래기국', '육개장', '콩나물국', '콩자반', '갓김치', '깍두기', '무생채', '배추김치', '백김치',
 '부추김치', '열무김치', '오이소박이', '총각김치', '파김치', '가지볶음', '고사리나물', '미역줄기볶음', '숙주나물', '시금치나물',
 '애호박볶음', '수제비', '열무국수', '잔치국수', '꽈리고추무침', '도라지무침', '도토리묵', '잡채', '콩나물무침', '김치볶음밥',
 '비빔밥', '새우볶음밥', '알밥', '감자채볶음', '건새우볶음', '고추장진미채볶음', '두부김치', '멸치볶음', '어묵볶음', '오징어채볶음',
 '주꾸미볶음', '깻잎장아찌', '감자전', '김치전', '동그랑땡', '생선전', '파전', '호박전', '갈치조림', '감자조림', '고등어조림',
 '꽁치조림', '두부조림', '땅콩조림', '연근조림', '우엉조림', '코다리조림', '전복죽', '호박죽', '닭계장', '동태찌개', '순두부찌개',
 '계란찜', '김치찜', '해물찜', '갈비탕', '감자탕', '곰탕_설렁탕', '매운탕', '삼계탕', '추어탕']
def model_check(cate,num):
    img = ""
    image_path = "C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb4/"+cate+"/"
    file_name_in = num + ".jpg"
    input_file_path = os.path.join(image_path, file_name_in)
    
    img = tf.keras.utils.load_img(
        input_file_path, target_size=(200,200)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array.shape
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "이미지는  {} with a {:.2f} percent 확신합니다."
        .format(categories_index[np.argmax(score)], 100 * np.max(score))
    )
    

from pathlib import Path
for cate in categories:
    data_dir = Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb4/"+cate+"/")
    image_files = list(data_dir.glob('*.[jJpPgG][pPnNbB]*'))[-1]
    file_name = os.path.basename(image_files)
    last_4_digits = file_name[:-4]
    image_count = int(last_4_digits) 
    cnt=1
    for _ in range(image_count):
        try:
            if cnt > 0 and cnt <10 : num = "00"+str(cnt)
            elif cnt >= 10 and cnt <100 : num = "0"+str(cnt)
            else : num = str(cnt)
            model_check(cate,num)
            cnt += 1  
        except:
            print("이미지 파일이 없습니다","category : ",cate,"_",num)
            cnt += 1
    print("완료된 image 수 : ",cnt)


# matplot 시각화

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## test

test_path="C:/Users/KITCOOP/kicpython/hansik/kfood_test/다운로드 (2).jpg"
Image.open(test_path)


#이미지 자동 사이징 한다 
img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array.shape


img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_names = train_ds.class_names
print(
    "이미지는  {} with a {:.2f} percent 확신합니다."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


#########################
###loaded_model test
print(input_file_path)
img = image.load_img(input_file_path)
img = img.resize((299,299))
x = image.img_to_array(img)
x = preprocess_input(x)
x = tf.expand_dims(x, axis=0)
predictions = loaded_model.predict(x)
top_prediction_index = tf.argmax(predictions, axis=-1)[0].numpy()
# 클래스 이름과 확률을 출력합니다.
predicted_class_name = categories_index[top_prediction_index]
predicted_class_name
X.append(predicted_class_name)
print(f"Predicted class: {predicted_class_name}")



