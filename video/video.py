# import numpy as np
# import pandas as pd

## Importing warning modules
import warnings

# import os
from os import listdir
from os.path import isfile, join

import fastai
from fastai import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.vision import *

# import moviepy.editor as mp
from moviepy.editor import *
from torchvision.models import *
from torchvision.utils import save_image

# import tensorflow as tf


# import cv2


# import subprocess
# from pip import main
# import dill
# import weakref
# import pathlib
# from subprocess import check_output
# import time


warnings.filterwarnings("ignore")

### Importing PIL modules
from PIL import Image  # , ImageEnhance, ImageOps

### Importing super_image modules
from super_image import EdsrModel, ImageLoader

# ### Importing Matplotlib modules
# import matplotlib.pyplot as plt
# from pylab import rcParams

cwd = os.getcwd()


from video.interpolation import *


def EDSR_SISR(img, model, size):
    img_fastai = fastai.vision.image.Image(
        fastai.vision.image.pil2tensor(img, dtype=np.float32).div_(255)
    )
    p, img_pred, b = model.predict(img_fastai)

    model_upsample = edsr_model_x2

    save_image(img_pred, "img.png")

    img_pred = Image.open("img.png")

    inputs = ImageLoader.load_image(img_pred)
    preds = model_upsample(inputs)
    ImageLoader.save_image(preds, "output.png")
    return Image.open("output.png")


def EDSR(img, size):
    model_upsample = edsr_model_x2
    inputs = ImageLoader.load_image(img)
    preds = model_upsample(inputs)
    ImageLoader.save_image(preds, "output.png")
    return Image.open("output.png")


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


def Superres_calling(img, factor):
    answer = img
    while factor != 1:
        answer = Superresolution(answer)
        factor = factor // 2

    return answer


"""
Defining global models where will store our DL model which was sent by audio-video.py using model_calling() function
"""
model32: fastai.basic_train.Learner
model64: fastai.basic_train.Learner
model128: fastai.basic_train.Learner
model256: fastai.basic_train.Learner
edsr_model_x2 = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)


# def load_model(pixel_size=256):
#     name_model = os.path.join(
#         os.getcwd(), "Pretrained Model", "model_" + str(pixel_size) + ".pkl"
#     )
#     print(name_model, "\n\n")
#     model = dill.load(open(name_model, "rb"))

#     return model


# model32 = load_model(32)
# model64 = load_model(64)
# model128 = load_model(128)
# model256 = load_model(256)


def model_calling(model32_1, model64_1, model128_1, model256_1, edsr_model_x2_1):
    global model32, model64, model128, model256, edsr_model_x2
    model32 = model32_1
    model64 = model64_1
    model128 = model128_1
    model256 = model256_1
    edsr_model_x2 = edsr_model_x2_1


def reading_video(video_path, save_path, factor=2):

    clip = VideoFileClip(video_path)
    # frames = clip.iter_frames()
    audio = clip.audio

    image_list = []
    for v in clip.iter_frames():
        image_list.append(Image.fromarray(v))

    ##### Rishabh ka function call

    image_list = interpolate(image_list, 30)

    frame_path = os.path.join(
        cwd, "video", "frames"
    )  # Use this if running audio-video.py or main.py

    ## Make frames folder in video folder
    os.makedirs(frame_path, exist_ok=True)

    # frame_path = os.path.join(cwd, , "frames")                 #Use this is running this file directly

    count = 0
    total_len = len(image_list)
    print(total_len, "is len of image list")
    output_image_list = []
    for i, j in image_list:
        count += j
        upscaled = np.array(Superres_calling(i, factor))
        output_image_list.extend([upscaled] * j)
        Image.fromarray(output_image_list[count - 1]).save(
            os.path.join(frame_path, str(count - 1) + ".png")
        )
        print(count, end="\r")

    size = (image_list[0][0].size[0] * factor, image_list[0][0].size[0] * factor)

    onlyfiles = [f for f in listdir(frame_path) if isfile(join(frame_path, f))]

    for i in range(len(onlyfiles)):
        onlyfiles[i] = os.path.join(frame_path, onlyfiles[i])

    onlyfiles.sort(key=lambda f: int(re.sub("\D", "", f)))

    # print(onlyfiles)

    fps = round(total_len / audio.duration)
    clip = ImageSequenceClip(onlyfiles, fps=fps)

    clip.write_videofile(save_path)

    return audio


# if __name__ == "__main__":
#     reading_video(
#         "C:/Users/JPG/Desktop/Low_videos/video5_150_200.mp4",
#         "C:/Users/JPG/Desktop/temp.mp4",
#         2,
#     )
