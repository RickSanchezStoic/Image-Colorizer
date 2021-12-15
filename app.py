from flask import Flask , request, render_template, send_file
import numpy as np
import pandas as pd
import pickle
import base64
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import keras
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
app= Flask(__name__)
model=keras.models.load_model('colorizer.h5')
#import pickle
#filename='model.pkl'
#pickle.dump(model, open(filename, 'wb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['image'].read()## byte value
    image=np.fromstring(file,np.uint8)
    image=cv2.imdecode(image,cv2.IMREAD_COLOR)
    
    print(image.shape)
    img_lab=image/255.0

    
    
    
    img_lab=rgb2lab(img_lab)
    
    img_lab=cv2.resize(img_lab,(224,224),interpolation=cv2.INTER_AREA)

    #img_lab=img_lab.reshape(img_lab.shape+(1,))
    print("this is shape:",img_lab.shape)
    img_lab_x=img_lab[:,:,0]
    img_lab_y=img_lab[:,:,1:]/128
    img_lab_x=img_lab_x.reshape(img_lab_x.shape+(1,))
    img_lab_x=np.expand_dims(img_lab_x,axis=0)
    print('hi')
    print(img_lab_x.shape)
    print('hi')
    predicted=model.predict(img_lab_x)
    predicted=np.squeeze(predicted)
    print(predicted.shape)
    complete_predicted_image=np.zeros((224,224,3))
    complete_predicted_image[:,:,0]=img_lab_x[0][:,:,0]
    complete_predicted_image[:,:,1:]=predicted*128.0
    imsave('output.png',lab2rgb(complete_predicted_image))
    l=cv2.imread('output.png')
    RGB_img = cv2.cvtColor(l, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png',RGB_img)

    b64_string=[]
    with open("output.png", "rb") as img_file:
     b64_string = base64.b64encode(img_file.read())
    #return send_file('output.png',mimetype='image/png')
    return b64_string
