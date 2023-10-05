import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib



# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

'''
    데이터 검사 및 이해하기
    입력 파이프라인 빌드하기
    모델 빌드하기
    모델 훈련하기
    모델 테스트하기
    모델을 개선하고 프로세스 반복하기
'''

#data_dir = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning/")
#data_dir_test = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/test/")
data_dir = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb1/")
data_dir_test = pathlib.Path("C:/Users/KITCOOP/kicpython/hansik/kfood_new/test_rgb1/")

image_count = len(list(data_dir.glob('*/*.jpg')))
image_test_count = len(list(data_dir_test.glob('*/*.jpg')))
# image_count = len(list(data_dir.glob('*/*.jpg')))
# image_test_count = len(list(data_dir_test.glob('*/*.jpg')))
print(image_count)
print(image_test_count)
print(list(data_dir.glob('*/*.jpg'))[:5])

# =============================================================================
# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0]))
# =============================================================================

gui000 = list(data_dir.glob('000/*'))
Image.open(str(gui000[0]))
Image.open(str(gui000[1]))

gui001 = list(data_dir.glob('001/*'))
Image.open(str(gui001[0]))
Image.open(str(gui001[1]))

batch_size = 512
img_height = 50
img_width = 50

data_dir
data_dir_test


# 훈련 데이터셋 생성
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 검증 데이터셋 생성
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=None,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


#성능 높이기
# Dataset.cache()는 첫 epoch 동안 디스크에서 이미지를 로드한 후 이미지를 메모리에 유지
# Dataset.prefetch는 훈련하는 동안 데이터 전처리 및 모델 실행을 중첩
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#데이터 표준화 하기
normalization_layer = layers.Rescaling(1./255)



normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


num_classes = len(class_names)
num_classes

# data_augmentation  =============================================================================
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


#학습
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.evaluate(val_ds)

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

print(
    "이미지는  {} with a {:.2f} percent 확신합니다."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


## 부록 원본 & 참고용 코드 주석

# =============================================================================
# #BatchDataset으로 학습후에 이미지로 test한다 
# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=None, # 검증 데이터가 이미 분류되어있다.
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
# 
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir_test, # 추가할 부분 
#   validation_split=None,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
# 
# ERROR : ValueError: If `subset` is set, `validation_split` must be set, and inversely.
# =============================================================================

# =============================================================================
# 이미지 파일 확인하는 부분
# import matplotlib.pyplot as plt
# 
# t1 =train_ds.take(1)
# len(train_ds)
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
#     
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break
# 
# =============================================================================

#  수작업 data_augmentation
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = data_augmentation(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")



#예측

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)




