import io
import fastai
import requests
import torch
from fastai.vision import open_image, ImageDataBunch, \
    imagenet_stats, cnn_learner
from fastai.vision.models import densenet121
from fastai.vision.transform import get_transforms
from fastai.vision import *
from sklearn.metrics import roc_auc_score
import  torch
torch.nn.Module.dump_patches = True

import warnings 
warnings.filterwarnings('ignore')


def auc_score(y_pred,y_true,tens=True):
    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score = tensor(score)
    return score

CLASES = ['Benign', 'Malignent']
MODEL = densenet121


CUSTOM_MODEL_PATH = 'models' 

def load_image(url):
    return open_image(io.BytesIO(requests.get(url).content))

fastai.torch_core.defaults.device = torch.device('cpu')



#img = load_image('https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg')
img = open_image('be73269e9f8f034cc0c0680a7ca849798783c0c6.tif')
learn = load_learner(CUSTOM_MODEL_PATH)

pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)

#data = ImageDataBunch.single_from_classes('', CLASES, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
#learn = cnn_learner(data, MODEL).load(CUSTOM_MODEL_PATH)
#pred_class,pred_idx,outputs = learn.predict(img)

#print(pred_class)