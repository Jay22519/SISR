def rms(frame1, frame2):
    """
    Takes 2 frames and calculates root mean square difference between them.
    """

    rms = np.sqrt(np.mean(frame1**2 - frame2**2))
    return rms


from PIL import Image  # No need for ImageChops
import math
import numpy as np
import cv2
# from skimage import img_as_float
# from skimage import compare_mse
from PIL import ImageChops
import moviepy.editor as mp
from moviepy.editor import *


# def rmsdiff(im1, im2):
#     """Calculates the root mean square error (RSME) between two images"""
#     return math.sqrt(compare_mse(img_as_float(im1), img_as_float(im2)))
def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return rms


def interpolate(arr, threshold):
    start = 0
    termcount=1
    ts=0
    temp = []
    count = 0

    for i in range(1, len(arr)):

        diff = rmsdiff(arr[i], arr[start])
        # print(diff)
        if diff > threshold or termcount>=3:
            temp.append((arr[start],termcount))
            ts+=termcount
            start = i
            count += 1
            termcount=1
        else:
            termcount+=1
    if(termcount>1):
        temp.append((arr[start],termcount))
        ts+=termcount
    print(ts)
    print(count, " is unique length")

    retval=[]
    for j in range(len(temp)-1):
        img,count=temp[j]
        nextimg,_=temp[j+1]
        img=np.asarray(img)
        nextimg=np.asarray(nextimg)
        retval.append((Image.fromarray(img),1))
        per=1/(count)
        for i in range(count):
            retval.append((Image.fromarray(cv2.addWeighted(img,i*per,nextimg,1-(i*per),0)),1))

    # print(retval)
    return retval

def main(video_path):
    clip = VideoFileClip(video_path)
    frames = clip.iter_frames()
    audio = clip.audio

    image_list = []
    for v in frames:
        image_list.append(Image.fromarray(v))

    # print(len(image_list))
    image_list = interpolate(image_list,30)

    print(len(image_list))


if __name__ == "__main__":
    main("D:\\final-year-project\input2.mp4")
