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

model_path='./saved_models/model.onnx' # filepath of the onnx model being used for inference
ort_session=ort.InferenceSession(model_path) # creating inference session

#===========================================Play=====================================================================
# Part of the code that gives inference  
# process function takes image path and sequence of coordinates as input and returns (1,1,64,64) as output. Here since we have coordinates 
# they are used to crop the image
# path variable refers to the filename or path
# pathhistory variable is the coordinate sequence
def process(path, pathhistory):
    pathhistory=np.array(pathhistory) # pathhistory is time sequence of coordinates
    image=Image.fromarray(plt.imread(path)[:,:,3]) # reading the alpha channel of image

    ###### finding bounding box of images #####
    xmin=int(min(pathhistory[:,0]))
    xmax=int(max(pathhistory[:,0]))
    ymin=int(min(pathhistory[:,1]))
    ymax=int(max(pathhistory[:,1]))
    w=xmax-xmin
    h=ymax-ymin
    x=xmin
    y=ymin
    ########################################
    image=np.array(image) # converting image from PIL to array

    # cropping if height is more than width
    if h>w:
      xmin=int(x-(h-w)/2) # offset of image along x-axis
      xmax=int(x+(h+w)/2)
     
      if xmin>=0: # center image if minimum x is positive 
        cropped=image[y:y+h,xmin:xmax]
      else: # if minimum x is negative, then shift image towards right by (h-w)/2 amount to center it
        cropped=image[y:y+h,x:x+h]
        temp=np.zeros((h,h))
        left=int((h-w)/2)
        right=int((h+w)/2)
        temp[:,left:right]=cropped[:,0:w]
        cropped=temp

    #similar if width is more than height  
    elif h<w:
      ymin=int(y-(w-h)/2)
      ymax=int(y+(h+w)/2)
  
      if ymin>=0:
        cropped=image[ymin:ymax,x:x+w] 
      else:
        cropped=image[y:y+w,x:x+w]
        temp=np.zeros((w,w))
        top=int((w-h)/2)
        bottom=int((h+w)/2)
        temp[top:bottom,:]=cropped[0:h,]
        cropped=temp

    # if both are same then simple cropping
    else:
      m=max(w,h)
      cropped=image[y:y+m,x:x+m] 


    image=Image.fromarray(cropped)
    
    image=np.array(image.resize((64,64))) #resizing image to 64x64
    image=(np.array(image)>0.1).astype(np.float32)[None,:,:] #get image of shape(1,64,64)

    return image[None] # adding one more dimension, now image is (1,1,64,64)


# Test function takes filename or path and coordinate time sequence as input and gives the inferred class as output 
# path variable is path to input image file
# pathhistory is 2D array of coordinate sequence
def test(path, pathhistory):
    image = process(path, pathhistory) # call process function to get (1,1,64,64) shaped image
    
    output = ort_session.run(None,{'data': image})[0].argmax() # calling inference session for inference, getting index value of class
    return allClasses[output] # getting class from the index value


app = Flask(__name__)
cors = CORS(app)

#====play function takes no input directly but returns the inferred class===
# post method means it can receive data
@app.route('/api/play', methods=['POST'])
def play():
    data= json.loads(request.data.decode('utf-8')) # decoding the received data in 'utf-8' format
    image_data = data['image'].split(',')[1].encode('utf-8') # receive image data 
    pathhistory = data['path'] # receive coordinate sequence data
    os.makedirs(f'inference/image', exist_ok=True) # make directory to store inferred image
    with open(f'inference/image/imagetoinfer.png', 'wb') as fh: 
        fh.write(base64.decodebytes(image_data)) # saving image to the inference directory
    filepath='inference/image/imagetoinfer.png' 
    return test(filepath,pathhistory)    


#============================================Create Dataset=======================================================================
# part for receiving training data from dataset-creation module
# datasetPath refers to the base directory where you want to receive data
# coordinate sequence is stored in path variable
datasetPath='./data'

# upload_canvas receives data from the create dataset module and saves it to the base directory
@app.route('/api/upload_canvas', methods=['POST'])
def upload_canvas():
    data= json.loads(request.data.decode('utf-8')) # decoding received data
    image_data = data['image'].split(',')[1].encode('utf-8') # encoding image array and saving it to image_data folder
    filename = data['filename'] # getting the filename
    fname=filename.split('.')[0] # getting filename without extension to save coordinate sequence data
    className = data['className'] # getting class name
    pathhistory = data['path'] # getting coordinate sequence
    os.makedirs(f'{datasetPath}/{className}/image', exist_ok=True) # creating directory to store image
    with open(f'{datasetPath}/{className}/image/{filename}', 'wb') as fh:
        fh.write(base64.decodebytes(image_data)) # saving image to the respective class directories
    os.makedirs(f'{datasetPath}/{className}/path', exist_ok=True)
    np.save(f'{datasetPath}/{className}/path/{fname}.npy', pathhistory) # saving coordinate sequence as .npy file
    return "Got the image " + className   


#=========================Saving wrong and correct predictions================================================
# similar to create dataset part but here we store the wrong predictions made by the model under correct directory
# datasetPath2 refers to the base directory for storing wrong prediction data
# no difference in implementation except the datasetPath value

@app.route('/api/upload_pred', methods=['POST'])
def upload_pred():
    data= json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    filename = data['filename']
    fname=filename.split('.')[0]
    className = data['className']
    pathhistory = data['path']
    if data['iscorrect']:
      datasetPath2='./correctpred'
    else:
      datasetPath2='./wrongpred'
    os.makedirs(f'{datasetPath2}/{className}/image', exist_ok=True)
    with open(f'{datasetPath2}/{className}/image/{filename}', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))
    os.makedirs(f'{datasetPath2}/{className}/path', exist_ok=True)
    np.save(f'{datasetPath2}/{className}/path/{fname}.npy', pathhistory)
    return "Got the correct class " + className     
