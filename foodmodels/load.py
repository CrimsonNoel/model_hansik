# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:02:08 2023

@author: KITCOOP
"""
from tensorflow.keras.applications import InceptionV3
import tensorflow as tf

class load_food_model:
    _instance = None
    model = None  # model 클래스 변수를 정의합니다.

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # InceptionV3 모델을 초기화하고 로드하는 코드
            cls._instance.model = tf.keras.models.load_model("kfoodpro/foodmodels/8")
            print(" == Food model loaded success == ")
        return cls._instance
    
    
    def predict(self, img_array):
        predictions = self.model.predict(tf.expand_dims(img_array, axis=0))
        
        return predictions
