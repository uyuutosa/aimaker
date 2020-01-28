import torch
from torch.nn import functional as F
import torch.nn as nn

from aimaker.models.normalize_factory import NormalizeFactory
from aimaker.layers.pad_factory import PadFactory
from aimaker.models.base_model import BaseModel
import aimaker.models.resnet_model as res


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class MixingGeneratorModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        n_classes           = int(config['PSPNetGenerator settings']['n_classes'])
        sizes               = tuple([int(x) for x in config['PSPNetGenerator settings']['sizes'].split(',')])
        psp_size            = int(config['PSPNetGenerator settings']['psp_size'])
        deep_features_size  = int(config['PSPNetGenerator settings']['deep_feature_size'])
        final               = int(config['PSPNetGenerator settings']['final'])
        backend             = config['PSPNetGenerator settings']['backend']
        self.feats          = res.ResNetForClassificationModel(config)
        self.n_pre          = int(config['PSPNetGenerator settings']['numberOfInputImageChannels'])

        self.pre = nn.Conv2d(self.n_pre, 3, kernel_size=1)

        self.psp = PSPModule(psp_size, 1028, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1028, 64)
        self.up_2 = PSPUpsample(64, 32)
        self.up_3 = PSPUpsample(32, 16)
        self.up_4 = PSPUpsample(16, 16)
        self.up_5 = PSPUpsample(16, 8)
        self.up_6 = PSPUpsample(8, 8)


        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=final),
            #nn.LogSoftmax()
        )
        
        self.runaway = Generator()
        self.activate = nn.Tanh()
        

    def forward(self, x):
        if self.n_pre != 3:
            x = self.pre(x)
        f = self.feats(x) 
        #f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        #p = self.up_2(p)
        #p = self.drop_2(p)

        #p = self.up_3(p)
        #p = self.drop_2(p)

        #p = self.up_4(p)
        #p = self.drop_2(p)

        #p = self.up_5(p)
        #p = self.drop_2(p)
        #
        #p = self.up_6(p)
        #p = self.drop_2(p)
        #print(self.final(p).shape)
        #print(self.runaway(x).shape)

        return self.activate(0.1 * self.final(p) + self.runaway(x))



class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=32, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
#        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
