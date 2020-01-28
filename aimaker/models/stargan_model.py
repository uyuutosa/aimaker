import torch.nn as nn

import aimaker.utils.util as util
from aimaker.layers import PadFactory
from aimaker.models.base_model import BaseModel


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class StarGANGeneratorModel(BaseModel):
    """Generator network."""
    def __init__(self, config,):
        super(StarGANGeneratorModel, self).__init__(config)
        conv_dim=64
        repeat_num = int(config['starGANDiscriminator settings']['repeatNum'])
        c_dim = len(self.config['starGAN settings']['typeNames'].strip().split(","))

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class StarGANDiscriminatorModel(BaseModel):
    def __init__(self,
                 config):

        super(StarGANDiscriminatorModel, self).__init__(config)
        self.config = config
        ch = util.ConfigHandler(config)
        self.input_nc    = int(config['starGANDiscriminator settings']['numberOfInputImageChannels'])
        self.output_nc   = int(config['starGANDiscriminator settings']['numberOfOutputImageChannels'])
        image_size       = int(config['starGANDiscriminator settings']['imageSize'])
        self.c_dim       = len(self.config['starGAN settings']['typeNames'].strip().split(","))
        self.gpu_ids     = ch.getGPUID()
        self.pad_factory = PadFactory(config)
        #self.norm_layer  = NormalizeFactory(config).create(
        #                        config['ResNetForGenerator settings']['normalizeLayer'])
        conv_dim        = int(config['starGANDiscriminator settings']['convDim'])
        self.repeat_num = int(config['starGANDiscriminator settings']['repeatNum'])

        layers = []
        layers.append(nn.Conv2d(self.input_nc,  conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, self.repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2
        k_size = int(image_size / power(2, self.repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, self.c_dim, kernel_size=k_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.main(x)
        out_src = self.sigmoid(self.conv1(h))
        out_cls = self.sigmoid(self.conv2(h))
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
