# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:27:23 2023

@author: KITCOOP
"""
from tensorflow import keras
from flask import Flask, render_template, request
import cv2
import pathlib
import numpy as np

#######################################################

name="flask01"

app = Flask(__name__)
static = 'static/'

@app.route('/',methods=['GET','POST'])
def upload():
    if request.method != 'POST':
        return render_template('index.html',name=name)
    else:
        file = request.files['upload']
        filename = file.filename
        file.save(static+filename)
        maxno=prediction(static+filename)
        return render_template('index.html', image_file=filename, num=maxno)
    

# pip install tensorflow
# pip install opencv-python
def prediction(filename):
    img1 = cv2.imread(filename) # 1. read file - 원본 이미지 읽기
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # 2. shape 수정
    img3 = np.expand_dims(img2, axis=0) # 3
    print(str(img1.shape)+"->"+str(img2.shape)+"->"+str(img3.shape))
    model = keras.models.load_model("templates/mnist.model")
    predictions = model.predict(img3) #예측
    print(predictions)
    maxno = np.argmax(predictions[0])
    print(maxno)
    return maxno


if __name__ == '__main__':
     app.run(debug=True, port="5002")























