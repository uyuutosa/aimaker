# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802
TODO:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import aimaker.utils.util as util
from aimaker.models.base_model import BaseModel


def swish(x):
    return x * F.sigmoid(x)


class residualBlock(BaseModel):
    def __init__(self, config, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__(config)

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(BaseModel):
    # Implements resize-convolution
    def __init__(self, config, in_channels, out_channels):
        super(upsampleBlock, self).__init__(config)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class SRGANGenerator(BaseModel):
    def __init__(self, config):
        super(SRGANGenerator, self).__init__(config)
        self.n_residual_blocks = int(config['SRGANGenerator settings']['nResidualBlocks'])
        self.upsample_factor   = int(config['SRGANGenerator settings']['upsampleFactor'])
                                                
        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4) # not change the width and hieght

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock(config))

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(config, 64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)
    

    def forward(self, x):
        x = swish(self.conv1(x))
        self._setFeatureForView("conv_1", x)
        self._setParameterForView("conv_1", self.conv1)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)
            self._setFeatureForView("residual_block_{}".format(i+1), y)

        x = self.bn2(self.conv2(y)) + x
        self._setFeatureForView("conv_2", x)
        self._setParameterForView("conv_2", self.conv2)

        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample' + str(i+1))(x)
            self._setFeatureForView("upsample_{}".format(i+1), x)

        x = self.conv3(x)
        self._setFeatureForView("conv_3", x)
        self._setParameterForView("conv_3", self.conv3)
        
        return x

class SRGANDiscriminator(BaseModel):
    def __init__(self, config):
        super(SRGANDiscriminator, self).__init__(config)

        ch = util.ConfigHandler(config)
        n = 1
        if config['global settings']['controller'] == 'pix2pix':
            n = 2 # for image pooling
        input_nc = ch.getNumberOfOutputImageChannels() * n
        self.conv1 = nn.Conv2d(input_nc, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))
        self._setFeatureForView("conv_1", x)

        x = swish(self.bn2(self.conv2(x)))
        self._setFeatureForView("conv_2", x)
        x = swish(self.bn3(self.conv3(x)))
        self._setFeatureForView("conv_3", x)
        x = swish(self.bn4(self.conv4(x)))
        self._setFeatureForView("conv_4", x)
        x = swish(self.bn5(self.conv5(x)))
        self._setFeatureForView("conv_5", x)
        x = swish(self.bn6(self.conv6(x)))
        self._setFeatureForView("conv_6", x)
        x = swish(self.bn7(self.conv7(x)))
        self._setFeatureForView("conv_7", x)
        x = swish(self.bn8(self.conv8(x)))
        self._setFeatureForView("conv_8", x)

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
