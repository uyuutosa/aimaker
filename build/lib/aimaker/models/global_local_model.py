# -*- coding: utf-8 -*-
"""Inspired 
TODO:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import aimaker.utils.util as util
from aimaker.models import BaseModel
from aimaker.models import ActivationFactory


def swish(x):
    return x * F.sigmoid(x)

class GLPatchGANModel(BaseModel):
    def __init__(self, config):
        super(GLPatchGANModel, self).__init__(config)

        n = 1
        if config['global settings']['controller'] == 'pix2pix':
            n = 2 # for image pooling
        input_nc = int(config['global local model settings']['numberOfInputImageChannels']) * n

        self.conv1 = nn.Conv2d(input_nc, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # global laysers
        self.global1    = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.global_bn1 = nn.BatchNorm2d(256)
        self.global2    = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.global_bn2 = nn.BatchNorm2d(512)
        self.global3    = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.global_bn3 = nn.BatchNorm2d(512)

        # local layres
        self.local1    = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.local_bn1 = nn.BatchNorm2d(256)
        self.local2    = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.local_bn2 = nn.BatchNorm2d(512)
        self.local3    = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.local_bn3 = nn.BatchNorm2d(512)

        self.shuffle  = nn.PixelShuffle(2 ** 3)


        # concated layres
        self.concated1    = nn.Conv2d(1024, 256, 3, stride=1, padding=1)
        self.concated_bn1 = nn.BatchNorm2d(256)

        # last activation
        self.last_activation = ActivationFactory(config)\
                .create(config['global local model settings']['last activation'])

        self.o = torch.autograd.Variable(torch.ones(2**6))
        #self.o = torch.autograd.Variable(torch.ones(2**6).cuda(ch.getGPUID()[0]))



    def cuda(self, device=None):
        if device != -1:
            self.o = self.o.cuda(device)
        return super(GLPatchGANModel, self).cuda(device)


    def forward(self, x):
        x = swish(self.conv1(x))
        self._setFeatureForView("conv_1", x)
        x = swish(self.bn2(self.conv2(x)))
        self._setFeatureForView("conv_2", x)
        x = swish(self.bn3(self.conv3(x)))
        self._setFeatureForView("conv_3", x)
        x = swish(self.bn4(self.conv4(x)))
        self._setFeatureForView("conv_4", x)

        # global layers
        global_x = swish(self.global_bn1(self.global1(x)))
        global_x = swish(self.global_bn2(self.global2(global_x)))
        global_x = swish(self.global_bn3(self.global3(global_x)))

        # local layers
        local_x = swish(self.local_bn1(self.local1(x)))
        local_x = swish(self.local_bn2(self.local2(local_x)))
        local_x = swish(self.local_bn3(self.local3(local_x)))

        # concatenate global and local feature
        h, w = global_x.shape[2:]
        global_x = self.shuffle((global_x[...,None] * self.o)\
                    .view(-1, 512*2**6, h, w))

        concated_x = torch.cat((global_x, local_x), 1)

        concated_x = swish(self.concated_bn1(self.concated1(concated_x)))

        return self.last_activation(concated_x)

class GLGeneratorModel(BaseModel):
    def __init__(self, config):
        super(GLGeneratorModel, self).__init__(config)

        ch = util.ConfigHandler(config)
        n = 1
        if config['global settings']['controller'] == 'pix2pix':
            n = 2 # for image pooling
        input_nc = int(config['global local model settings']['numberOfInputImageChannels']) * n

        self.conv1 = nn.Conv2d(input_nc, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # global laysers
        self.global1    = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.global_bn1 = nn.BatchNorm2d(256)
        self.global2    = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.global_bn2 = nn.BatchNorm2d(512)
        self.global3    = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.global_bn3 = nn.BatchNorm2d(512)

        # local layres
        self.local1    = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.local_bn1 = nn.BatchNorm2d(256)
        self.local2    = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.local_bn2 = nn.BatchNorm2d(512)
        self.local3    = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.local_bn3 = nn.BatchNorm2d(512)

        self.shuffle = nn.PixelShuffle(2 ** 3)


        # concated layres
        self.concated1    = nn.Conv2d(1024, 256, 3, stride=1, padding=1)
        self.concated_bn1 = nn.BatchNorm2d(256)

        # last activation
        self.last_activation = ActivationFactory(config)\
                .create(config['global local model settings']['last activation'])

        self.o = torch.autograd.Variable(torch.ones(2**6))
        #self.o = torch.autograd.Variable(torch.ones(2**6).cuda(ch.getGPUID()[0]))

        self.shuffle2 = nn.PixelShuffle(4)
        self.last_conv = nn.Conv2d(16, 3, 3, 1, 1)

    def cuda(self, device=None):
        if device != -1:
            self.o = self.o.cuda(device)
        return super(GLGeneratorModel, self).cuda(device)


    def forward(self, x):
        x = swish(self.conv1(x))
        self._setFeatureForView("conv_1", x)
        x = swish(self.bn2(self.conv2(x)))
        self._setFeatureForView("conv_2", x)
        x = swish(self.bn3(self.conv3(x)))
        self._setFeatureForView("conv_3", x)
        x = swish(self.bn4(self.conv4(x)))
        self._setFeatureForView("conv_4", x)

        # global layers
        global_x = swish(self.global_bn1(self.global1(x)))
        global_x = swish(self.global_bn2(self.global2(global_x)))
        global_x = swish(self.global_bn3(self.global3(global_x)))

        # local layers
        local_x = swish(self.local_bn1(self.local1(x)))
        local_x = swish(self.local_bn2(self.local2(local_x)))
        local_x = swish(self.local_bn3(self.local3(local_x)))

        # concatenate global and local feature
        h, w = global_x.shape[2:]
        global_x = self.shuffle((global_x[...,None] * self.o)\
                    .view(-1, 512*2**6, h, w))

        concated_x = torch.cat((global_x, local_x), 1)

        concated_x = swish(self.concated_bn1(self.concated1(concated_x)))

        concated_x = self.shuffle2(concated_x)
        concated_x = self.last_conv(concated_x)

        return self.last_activation(concated_x)
