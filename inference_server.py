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
model_path='./saved_models/100model.onnx'
ort_session=ort.InferenceSession(model_path)

#===========================================Play=====================================================================
def process(path):
    image=Image.fromarray(plt.imread(path)[:,:,3])
    image=np.array(image.resize((64,64)))
    image=(np.array(image)>0.1).astype(np.float32)[None,:,:] #get image of shape(1,64,64)
    return image[None]

def test(path):
    image = process(path)
    output = ort_session.run(None,{'data': image})[0].argmax()
    return allClasses[output]


app = Flask(__name__)
cors = CORS(app)
datasetPath='inference'

@app.route('/api/play', methods=['POST'])
def play():
    data= json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    os.makedirs(f'{datasetPath}/image', exist_ok=True)
    with open(f'{datasetPath}/image/imagetoinfer.png', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))
    path=datasetPath + '/image/'+'imagetoinfer.png' # change this value to test your own images
    return test(path)    

#============================================Create Dataset=======================================================================
datasetPath='data'

@app.route('/api/upload_canvas', methods=['POST'])
def upload_canvas():
    data= json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    filename = data['filename']
    fname=filename.split('.')[0]
    className = data['className']
    path = data['path']
    print(path)
    os.makedirs(f'{datasetPath}/{className}/image', exist_ok=True)
    with open(f'{datasetPath}/{className}/image/{filename}', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))
    os.makedirs(f'{datasetPath}/{className}/path', exist_ok=True)
    np.save(f'{datasetPath}/{className}/path/{fname}.npy', path)
    return "Got the image " + className    