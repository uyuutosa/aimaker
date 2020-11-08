import os

import torch
from aimaker.models.base_model import BaseModel
from torch.nn import functional as F
import torch.nn as nn
from .MobileNetV2 import MobileNetV2
from numpy import *

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, act=nn.PReLU()):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            act
        )

    def forward(self, X):
        return self.conv(X)




class PyramidModelGenerator(BaseModel):
    def __init__(self, settings):
        super(PyramidModelGenerator, self).__init__(settings)
        self.model_path = os.path.join('/'.join(__file__.split('/')[:-1]), '../../trained_models/mobilenet_v2.pth.tar')
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        net = MobileNetV2()
        net.load_state_dict(state_dict)
        self.feature = torch.nn.Sequential(*list(list(net.children())[0:1][0].children()))
        self.in_channels_lst = [list(x.children())[0][0].in_channels for x in self.feature[1:-1]]
        self.out_channels_lst = [list(x.children())[0][-1].num_features for x in self.feature[1:-1]]
        #self.k_lst   = [1,1,4,4,4,4,3,3,3,3,3,3,3,3,3]
        #self.pad_lst = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]
        #self.m_lst   = [1,1,4,4,4,4,9,9,9,9,9,9,9,21,21]
        self.k_lst   = [3,3,4,6,6,6,8,8,8,8,8,8,8,4,4]
        self.pad_lst = [1,1,0,1,1,1,0,0,0,0,0,0,0,0,0]
        self.m_lst   = [1,1,3,4,4,4,8,8,8,8,8,8,8,17,17]
        # (h,w) = (128,128) case
        #self.k_lst   = [1,1,4,4,4,4,3,3,3,3,3,3,3,3,3]
        #self.pad_lst = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]
        #self.m_lst   = [1,1,4,4,4,4,9,9,9,9,9,9,9,21,21]
        self.up_lst  = [UpSample(x,y, kernel_size=k, stride=m, padding=p).to(self.gpu_ids[0]) for x,y,k,m,p in\
                       zip(self.out_channels_lst[:-1],self.in_channels_lst[1:], self.k_lst, self.m_lst, self.pad_lst)]
        self.output = UpSample(960, 52, 4, 2, 1, act=nn.Tanh())


    def forward(self, X):
        self.lst = []
        for i in range(len(self.feature)-3):
            X = self.feature[i](X)
            if i > 1 and i < len(self.feature):
                self.lst += [self.up_lst[i-1](X)]
        self.lst = self.lst[2:]
        return self.output(torch.cat(self.lst, dim=1))
