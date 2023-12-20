import os

# import pathlib
## Importing other modules
import subprocess

# import time
import warnings

# import dill
# import fastai
import numpy as np
import pandas as pd
from fastai import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.vision import *

# import moviepy.editor as mp
from moviepy.editor import *
from torchvision.models import *

# import weakref
# from os import listdir
# from os.path import isfile, join
# from subprocess import check_output


# from torchvision.utils import save_image
# import tensorflow as tf


# import cv2


warnings.filterwarnings("ignore")

# import matplotlib.pyplot as plt

### Importing PIL modules
# from PIL import Image, ImageEnhance, ImageOps
# from pylab import rcParams
# from super_image import EdsrModel, ImageLoader

from audio.audio import *
from video.video import *


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = (
            [
                "pixel",
            ]
            + [f"feat_{i}" for i in range(len(layer_ids))]
            + [f"gram_{i}" for i in range(len(layer_ids))]
        )

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [
            base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.feat_losses += [
            base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


"""
Getting current working directory 
"""
cwd = os.getcwd()


from config import settings

"""
Input , output locations for Audio and Video
"""
# INPUT_VIDEO_PATH = "video/test.mp4"
OUTPUT_VIDEO_PATH_NOAUDIO = "output_noaudio.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"
OUTPUT_IMAGE_PATH = "output_final.png"
EXTRACTED_AUDIO_PATH = "audio/extracted_audio.wav"
PROCESSED_OUTPUT_AUDIO_PATH = "audio/processed_output.wav"


"""
Cutoff frequency which will be used by audio.py
"""
CUTOFF_FREQUENCY = 800.0


# load_model and class Model will be part of settings.py
###############
# def load_model(pixel_size=256):
#     name_model = os.path.join(
#         os.getcwd(), "video", "Pretrained Model", "model_" + str(pixel_size) + ".pkl"
#     )
#     print(name_model, "\n\n")
#     model = dill.load(open(name_model, "rb"))

#     return model


# def loading_model():
#     global model32, model64, model128, model256, edsr_model_x2
#     # model32 = load_model(32)
#     # model64 = load_model(64)
#     # model128 = load_model(128)
#     # model256 = load_model(256)
#     # edsr_model_x2 = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)

#     model32 = settings.model32
#     model64 = settings.model64
#     model128 = settings.model128
#     model256 = settings.model256
#     edsr_model_x2 = settings.edsr_model_x2


def run_command_windows(cmd):
    cmd = cmd.split()
    output = subprocess.Popen(
        ["powershell", "-Command"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    std_out, std_err = output.communicate()
    std_out, std_err = std_out.decode("utf-8").split("\n"), std_err.decode(
        "utf-8"
    ).split("\n")

    return std_out, std_err


def run_command_linux(cmd):
    cmd = cmd.split()
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_out, std_err = output.communicate()
    std_out, std_err = std_out.decode("utf-8").split("\n"), std_err.decode(
        "utf-8"
    ).split("\n")

    return std_out, std_err


"""
Our main function . It'll perform operations in following order 
        1) Read models 
        2) Call reading_video function of Video.py and this function will perform Superresolution of Video and save it in OUTPUT_VIDEO_PATH_NOAUDIO
        3) Reading_video will also return the original audio which will be enhanced using write_audiofile of Audio.py 
        4) Finally audio and video(with no audio) will be merged and saved in OUTPUT_VIDEO_PATH
"""


def Work(INPUT_PATH, factor, audio, type):

    ## That is input is video
    if type == 1:
        ## Passing models to video.py
        model_calling(
            settings.model32,
            settings.model64,
            settings.model128,
            settings.model256,
            settings.edsr_model_x2,
        )
        audio = reading_video(INPUT_PATH, OUTPUT_VIDEO_PATH_NOAUDIO, factor)
        if audio == 1:
            audio.write_audiofile(EXTRACTED_AUDIO_PATH)
            wave_stuff(
                EXTRACTED_AUDIO_PATH, PROCESSED_OUTPUT_AUDIO_PATH, CUTOFF_FREQUENCY
            )

            videoclip = VideoFileClip(OUTPUT_VIDEO_PATH_NOAUDIO)
            audioclip = AudioFileClip(PROCESSED_OUTPUT_AUDIO_PATH)

            new_audioclip = CompositeAudioClip([audioclip])
            videoclip.audio = new_audioclip
            videoclip.write_videofile(OUTPUT_VIDEO_PATH)

            return OUTPUT_VIDEO_PATH
        else:
            videoclip = VideoFileClip(OUTPUT_VIDEO_PATH_NOAUDIO)
            videoclip.audio = audio
            videoclip.write_videofile(OUTPUT_VIDEO_PATH)
            return OUTPUT_VIDEO_PATH

    ## Input is photo
    else:
        audio = 0
        # model_reading()
        model_calling(model32, model64, model128, model256, edsr_model_x2)
        image = Upscaling_image(INPUT_PATH, factor, OUTPUT_IMAGE_PATH)
        return OUTPUT_IMAGE_PATH


if __name__ == "__main__":
    loading_model()
    print("Done with loading\n\n")
    Work("C:/Users/JPG/Desktop/Low_videos/video5_150_200.mp4", 2, 1, 1)
