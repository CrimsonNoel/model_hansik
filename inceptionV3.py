# -*- coding: utf-8 -*-

from PIL import Image

import math

import keras
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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pathlib


labels =['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083']
data_dir =pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb3/")
data_dir_test = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/test_rgb3/")

data_dir
data_dir_test

img_height = 299
img_width = 299  # InceptionV3 모델의 입력 크기
batch_size = 128
num_classes = len(labels)  # 분류할 클래스 수

# 데이터 augmentation 설정 (선택사항)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.5),
])

# 데이터 로딩 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 이미지 픽셀 값을 [0, 1] 범위로 조정
    preprocessing_function=data_augmentation,0
    validation_split=None  # 검증 데이터 분할 비율
)


# 훈련 데이터셋 생성
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# 검증 데이터셋 생성
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

# InceptionV3 모델 로드
inception_model = tf.keras.applications.InceptionV3(
    include_top=False,  # Fully Connected Layer를 포함하지 않음
    weights="imagenet",  # ImageNet 데이터셋으로 사전 훈련된 가중치 사용
    input_shape=(img_height, img_width, 3),  # 입력 이미지 크기 설정
)

# 사용자 정의 출력 레이어 추가
output_layer = tf.keras.layers.GlobalAveragePooling2D()(inception_model.output)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(output_layer)

# 모델 정의
model = tf.keras.Model(inputs=inception_model.input, outputs=output_layer)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 원-핫 인코딩 수행
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(output_layer)


# 모델 요약 정보 출력
inception_model.summary()

# 모델 학습
epochs=2
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)


model.save("C:/Users/KITCOOP/kicpython/hansik",overwrite=True,save_format=None,
           include_optimizer=True,signatures=None,options=None,save_traces=True)

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