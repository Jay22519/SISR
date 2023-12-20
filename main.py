## Importing uvicorn to run fast api
# import os

## Importing fastai modules
# import fastai
import tensorflow as tf
import uvicorn
from fastai import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.vision import *

## Importing fastapi modules
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.templating import Jinja2Templates

## Importing torch and tensorflow modules
from torchvision.models import *
from torchvision.utils import save_image

# from super_image import EdsrModel, ImageLoader


## Importing pydantic for BaseSettings and BaseModel
# from pydantic import BaseSettings, BaseModel


# from pydantic import BaseSettings


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


## Importing settings where all the models are loaded
from audio_video import *

##Creating FastAPI object
app = FastAPI()
templates = Jinja2Templates(directory="templates/")


### Utility function to save a file
def save_file(filename, data):
    with open(filename, "wb") as f:
        f.write(data)


### Home page
@app.get("/")
async def intro():
    return {"Welcome to SISR"}


"""
This is the call which is made initially and will re directed us to index.html for user to upload video and other things
"""


@app.get("/predict")
async def info(request: Request):
    result = "Enter your video"
    return templates.TemplateResponse(
        "index.html", context={"request": request, "result": result}
    )


# video_file: UploadFile


"""
This is where index.html will redirect where the form is submitted . 
It extracts the video as UploadFile object and will save it locally and it's path will be sent to video.py via audio_video
It also gets the factor of upscaling , type of input and audio upscaling 
"""


@app.post("/predict")
async def info(
    request: Request,
    video_file: List[UploadFile] = File(..., description="Uploading video"),
    factor: int = Form(...),
    type_input: int = Form(...),
    audio: int = Form(...),
):

    factor = factor  ## Factor of upscaling
    type_input = type_input  ## Type = 0 for Image , 1 for video
    audio = audio  ## 0 for audio disable , 1 for enable

    result = {"Name": list(), "factor": factor, "type": type_input, "audio": audio}

    # print("Here redirected ...\n")
    filename = ""
    for f in video_file:
        contents = await f.read()
        save_file(f.filename, contents)

        print("Name : ", f.filename)
        result["Name"].append(f.filename)
        filename = f.filename

    output_path = Work(filename, factor, audio, type_input)

    return templates.TemplateResponse(
        "index.html", context={"request": request, "result": result}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
