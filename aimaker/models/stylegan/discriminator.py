### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from torch.nn.init import kaiming_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np

from aimaker.models.base_model import BaseModel

class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = f[:, :, ::-1, ::-1]
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x

            
class StyleDiscriminator(BaseModel):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 f=None,         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 settings=None
                 **kwarg
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__(settings)
        if settings is not None:
            resolution = settings['resolution']
            fmap_base = settings['fmap_base']
            num_channels = settings['num_channels']
            structure = settings['structure']
            fmap_max = settings['fmap_max']
            fmap_decay = settings['fmap_decay']
            f = settings['f']

        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.resolution_log2-1), kernel_size=1)
        self.structure = structure

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-5), kernel_size=2, stride=2)
        self.down22 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), kernel_size=2, stride=2)
        self.down23 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), kernel_size=2, stride=2)
        self.down24 = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), kernel_size=2, stride=2)

        # conv1: padding=same
        self.conv1 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-4), kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(self.nf(self.resolution_log2-4), self.nf(self.resolution_log2-5), kernel_size=3, padding=(1, 1))
        self.conv6 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-6), kernel_size=3, padding=(1, 1))
        self.conv7 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), kernel_size=3, padding=(1, 1))
        self.conv8 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), kernel_size=3, padding=(1, 1))

        # calculate point:
        self.conv_last = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(1), kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 2. 512 x 512 -> 256 x 256
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 3. 256 x 256 -> 128 x 128
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 4. 128 x 128 -> 64 x 64
            res -= 1
            x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 5. 64 x 64 -> 32 x 32
            res -= 1
            x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)

            # 6. 32 x 32 -> 16 x 16
            res -= 1
            x = F.leaky_relu(self.conv6(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down22(self.blur2d(x)), 0.2, inplace=True)

            # 7. 16 x 16 -> 8 x 8
            res -= 1
            x = F.leaky_relu(self.conv7(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down23(self.blur2d(x)), 0.2, inplace=True)

            # 8. 8 x 8 -> 4 x 4
            res -= 1
            x = F.leaky_relu(self.conv8(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down24(self.blur2d(x)), 0.2, inplace=True)

            # 9. 4 x 4 -> point
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            # N x 8192(4 x 4 x nf(1)).
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x


