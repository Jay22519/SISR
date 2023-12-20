"""
Importing all modules 

"""

### Importing numpy and pandas 
import numpy as np
import pandas as pd


## Importing fastai modules
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *


## Importing torch and tensorflow modules 
from torchvision.models import *
from torchvision.utils import save_image
import tensorflow as tf

## Importing Moviepy modules 
import moviepy.editor as mp
from moviepy.editor import *


## Importing cv2 modules
import cv2

### Importing os modules 
import os
from os import listdir
from os.path import isfile, join

## Importing other modules 
import subprocess
from pip import main
import dill
import weakref
import pathlib
from subprocess import check_output
import time

## Importing warning modules
import warnings
warnings.filterwarnings("ignore")

### Importing PIL modules 
from PIL import Image, ImageEnhance , ImageOps 
from PIL import ImageFilter

### Importing super_image modules
from super_image import EdsrModel, ImageLoader


### Importing Matplotlib modules
import matplotlib.pyplot as plt
from pylab import rcParams



"""
Defining global models where will store our DL model which was sent by audio-video.py using model_calling() function
"""
model32: fastai.basic_train.Learner
model64: fastai.basic_train.Learner
model128: fastai.basic_train.Learner
model256: fastai.basic_train.Learner
edsr_model_x2 = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)

"""
Getting current working directory 
"""
cwd = os.getcwd()


"""
EDSR_SISR function . This is the main function will upscales and helps in superresolution of a single image 
Parameters are -> Image (PIL.Image) 
               -> model (which model is to be used for superresolution)
               -> size of Image

First it gives us a superesolution image with the help of our model and then upscales it using edsr_model_x2

Issue -> Our model is trained on normalised images and thus it returns images with in normalised range (by this I don't necessarily mean 
pixels between 0 and 1) and may also contain negative pixels . That's why we can't change our image into normal [0,255] using usual normalise 
techniques . So to solve this what I'm doing is saving the predicted Image ( which is in form of Tensor) into image with the help of 
save_image and then again load it and perform upscaling
"""
def EDSR_SISR(img, model, size):
    img_fastai = fastai.vision.image.Image(pil2tensor(img, dtype=np.float32).div_(255))
    p, img_pred, b = model.predict(img_fastai)

    model_upsample = edsr_model_x2

    save_image(img_pred, "img.png")

    img_pred = Image.open("img.png")

    inputs = ImageLoader.load_image(img_pred)
    preds = model_upsample(inputs)
    ImageLoader.save_image(preds, "output.png")
    return Image.open("output.png")



       
"""
EDSR function . This funtion just performs the EDSR Upscaling on Images . 
Parameters are -> Image (PIL.Image) 
               -> size of Image
Reason I've created this as a seperate function is beacuse for high resolution images ( images with input pixel > 512) , we don't need to perform 
superresolution and simple Upscaling would suffice . So for those type of images this function is called for superresolution
"""
def EDSR(img, size):
    model_upsample = edsr_model_x2
    inputs = ImageLoader.load_image(img)
    preds = model_upsample(inputs)
    ImageLoader.save_image(preds, "output.png")
    return Image.open("output.png")
    
"""
Superresolution function . This function handles the main logic behind superesolution that is deciding which model to use, which function to call 
for superresolution ,etc 
Parameters are -> Image (PIL.Image) 

Operation -> First based on image.size it will decide will model to call and how to resize the image accordinly . Then call the required 
SISR function

Extra point -> We are also maintaining the aspect ratio of Image 

"""
def Superresolution(img_input):
    global model32, model64, model128, model256, edsr_model_x2

    limit = 32
    size = min(img_input.size[0], img_input.size[1])
    if size < 32:
        size = max(img_input.size[0], img_input.size[1])
    if size < 32:
        size = 32
    aspect_ratio = img_input.size[0] / img_input.size[1]
    model = model32
    if size < 64:
        limit = 32
    elif size >= 64 and size < 128:
        limit = 64
        model = model64
    elif size >= 128 and size < 256:
        limit = 128
        model = model128
    elif size >= 256 and size < 512:
        limit = 256
        model = model256
    elif size >= 512:
        limit = 512 * (size // 512)

    img_to_send = img_input.resize((limit, limit))
    answer = ""
    if size < 512:
        answer = EDSR_SISR(img_to_send, model, limit)
    else:
        answer = EDSR(img_to_send, limit)

    l, h = answer.size
    answer = answer.resize(((int(l * aspect_ratio)), int(h)))

    return answer



"""
model_calling function . This function will be called in audio-video.py and will return the models which were loaded there
"""
def model_calling(model32_1  , model64_1  , model128_1  , model256_1   , edsr_model_x2_1) :
    global model32  , model64  , model128  , model256   , edsr_model_x2
    model32 = model32_1
    model64 = model64_1
    model128 = model128_1
    model256 = model256_1
    edsr_model_x2 = edsr_model_x2_1


"""
Upscaling_image  function .
Function to upscale image and save it in save_path
"""
def Upscaling_image(input_path ,factor , save_path) :
    answer = Image.open(input_path)
    print(answer.size) 
    while(factor != 1) :
        answer = Superresolution(answer) 
        factor = factor//2
    answer.save(save_path)

 
    # answer = answer.filter(ImageFilter.MedianFilter)  #ImageFilter.MinFilter
    # answer = answer.filter(ImageFilter.GaussianBlur(radius = 3))

    print("save_path is : " , save_path)

    return answer 

