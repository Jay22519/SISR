from pydantic import BaseSettings

## Importing fastai modules
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from fastai.vision import *


## Importing torch and tensorflow modules 
from torchvision.models import *
from torchvision.utils import save_image
import tensorflow as tf

from super_image import EdsrModel, ImageLoader

import dill

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()





def load_model(pixel_size=256):
    name_model = os.path.join(os.getcwd() , "video" ,"Pretrained Model" , "model_" + str(pixel_size) + ".pkl" )
    print(name_model,"\n\n")
    model = dill.load(open(name_model, "rb"))

    return model

class Settings():

    model32 : fastai.basic_train.Learner =  load_model(32)
    model64 : fastai.basic_train.Learner = load_model(64)
    model128 : fastai.basic_train.Learner =  load_model(128)
    model256 : fastai.basic_train.Learner =  load_model(256)

    edsr_model_x2  = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

    name : str = "Jay"

settings = Settings()