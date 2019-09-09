import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler

import json
from matplotlib.ticker import FormatStrFormatter
import copy
from collections import OrderedDict


class_dict = {1:'Malignent',0:'Benign'}


class SimpleCNN(nn.Module):
    def __init__(self):
        # ancestor constructor call
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * 1 * 1, 2) # !!!
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) # first convolutional layer then batchnorm, then activation then pooling layer.
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))
        x = self.avg(x)
        #print(x.shape) # lifehack to find out the correct dimension for the Linear Layer
        x = x.view(-1, 512 * 1 * 1) # !!!
        x = self.fc(x)
        return x


# torch.save(model.state_dict(), 'model.ckpt')

def load_model(path):
    checkpoint = torch.load(path,map_location='cpu')
    model = SimpleCNN()
    model.load_state_dict(checkpoint)
    return model
    
loaded_model = load_model('model.ckpt')



def process_image(image):
    img_pil  = cv2.imread(image)
    print(img_pil)
    adjustments = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

    
    img_tensor = adjustments(img_pil)
    #print(img_tensor.shape)
    return img_tensor
    

img = '/home/bat/934f42de720b8f477c22f01e7aed751c00bf1118.tif'
#img_tensor = process_image(img)



def predict(image_path, model):
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    model.eval()
    with torch.no_grad():
        output = model.forward(img_torch)
    print(output)
    #_, predicted = torch.max(output.data, 1)    
    probability = F.softmax(output.data,dim=1)        
    prob,clas = probability.topk(1)
    return prob,clas
prob,clas = predict(img, loaded_model)
#print(type(classs[0][0].numpy()[0]))
print('Class : %s ,Probability : %s' %(class_dict[clas[0][0].tolist()],prob[0][0].tolist()))

