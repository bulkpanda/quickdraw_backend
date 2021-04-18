# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:38:18 2021

@author: Kunal Patel
"""
import os 
from flask import Flask, request
import json
from flask_cors import CORS
import base64

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

allClasses=['Bird', 'Flower', 'Hand', 'House', 'Mug', 'Pencil', 'Spoon', 'Sun', 'Tree', 'Umbrella']
model_path='./saved_models/model.onnx'
ort_session=ort.InferenceSession(model_path)

#===========================================Play=====================================================================
def process(path, pathhistory):
    pathhistory=np.array(pathhistory)
    image=Image.fromarray(plt.imread(path)[:,:,3])
    xmin=int(min(pathhistory[:,0]))
    xmax=int(max(pathhistory[:,0]))
    ymin=int(min(pathhistory[:,1]))
    ymax=int(max(pathhistory[:,1]))
    w=xmax-xmin
    h=ymax-ymin
    x=xmin
    y=ymin
    image=np.array(image)
    if h>w:
      xmin=int(x-(h-w)/2)
      xmax=int(x+(h+w)/2)
      print('After x: ',x)
      if xmin>=0:
        cropped=image[y:y+h,xmin:xmax]
      else:
        cropped=image[y:y+h,x:x+h]
        temp=np.zeros((h,h))
        left=int((h-w)/2)
        right=int((h+w)/2)
        temp[:,left:right]=cropped[:,0:w]
        cropped=temp
      # if x>=0:
      #   cropped=im[y:y+h,x:x+h] 
    elif h<w:
      ymin=int(y-(w-h)/2)
      ymax=int(y+(h+w)/2)
      print('After y: ',y)
      if ymin>=0:
        cropped=image[ymin:ymax,x:x+w] 
      else:
        cropped=image[y:y+w,x:x+w]
        temp=np.zeros((w,w))
        top=int((w-h)/2)
        bottom=int((h+w)/2)
        temp[top:bottom,:]=cropped[0:h,]
        cropped=temp
    else:
      m=max(w,h)
      cropped=image[y:y+m,x:x+m] 
    image=Image.fromarray(cropped)
    image=np.array(image.resize((64,64)))
    image=(np.array(image)>0.1).astype(np.float32)[None,:,:] #get image of shape(1,64,64)
    #plt.imshow(np.moveaxis(image,0,-1), cmap='gray')
    #plt.show()
    return image[None]

def test(path, pathhistory):
    image = process(path, pathhistory)
    
    output = ort_session.run(None,{'data': image})[0].argmax()
    return allClasses[output]


app = Flask(__name__)
cors = CORS(app)
datasetPath='inference'

@app.route('/api/play', methods=['POST'])
def play():
    data= json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    pathhistory = data['path']
    os.makedirs(f'inference/image', exist_ok=True)
    with open(f'inference/image/imagetoinfer.png', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))
    path='inference/image/'+'imagetoinfer.png' # change this value to test your own images
    return test(path,pathhistory)    

#============================================Create Dataset=======================================================================
datasetPath='D:/quickdraw/traincode/data1'

@app.route('/api/upload_canvas', methods=['POST'])
def upload_canvas():
    data= json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    filename = data['filename']
    fname=filename.split('.')[0]
    className = data['className']
    path = data['path']
    #print(path)
    os.makedirs(f'{datasetPath}/{className}/image', exist_ok=True)
    with open(f'{datasetPath}/{className}/image/{filename}', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))
    os.makedirs(f'{datasetPath}/{className}/path', exist_ok=True)
    np.save(f'{datasetPath}/{className}/path/{fname}.npy', path)
    return "Got the image " + className   

#=========================Saving wrong predictions================================================
datasetPath='./wrongpred'
@app.route('/api/upload_wrongpred', methods=['POST'])
def upload_wrongpred():
    data= json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    filename = data['filename']
    fname=filename.split('.')[0]
    className = data['className']
    path = data['path']
    #print(path)
    os.makedirs(f'{datasetPath}/{className}/image', exist_ok=True)
    with open(f'{datasetPath}/{className}/image/{filename}', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))
    os.makedirs(f'{datasetPath}/{className}/path', exist_ok=True)
    np.save(f'{datasetPath}/{className}/path/{fname}.npy', path)
    return "Got the correct class " + className     
