import torch.nn as nn
from aimaker.utils import SettingHandler

from aimaker.models import NormalizeFactory
from aimaker.layers.pad_factory import PadFactory
from aimaker.models import BaseModel


class ResNetForClassificationModel(BaseModel):
    def __init__(self, setting):
        super(ResNetForClassificationModel, self).__init__(setting)
        ch = SettingHandler(setting)
        n_trim = int(setting['models']['resNet']['pretrain']['nTrim'])
        self.net = ch.getResNet()
        if setting['models']['resNet']['pretrain']['trimFullConnectedLayer']:
            self.net = nn.Sequential(*list(self.net.children())[:-n_trim])

    def forward(self, x):
        return self.net(x)


class ResNetForGeneratorModel(BaseModel):
    def __init__(self,
                 setting):
        super(ResNetForGeneratorModel, self).__init__(setting)

        self.setting = setting
        ch = SettingHandler(setting)
        input_nc         = int(setting['base']['numberOfInputImageChannels'])
        output_nc        = int(setting['base']['numberOfOutputImageChannels'])
        ngf              = int(setting['models']['resNet']['generator']['filterSize'])
        use_bias         = setting['models']['resNet']['generator']['useBias']
        self.gpu_ids     = ch.getGPUID()
        self.pad_factory = PadFactory(setting)
        self.norm_layer  = NormalizeFactory(setting).create(
                                setting['models']['resNet']['generator']['normalizeLayer'])

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 self.norm_layer(ngf),
                 nn.ReLU(True)
                ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult   = 2 ** i
            model += [nn.Conv2d(ngf * mult,
                                ngf * mult * 2,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=use_bias),
                      self.norm_layer(ngf * mult * 2),
                      nn.ReLU(True)
                     ]
        mult = 2**n_downsampling

        for i in range(int(setting['models']['resNet']['generator']['nBlocks'])):
            model += [ResnetBlock(ngf * mult, setting)] 

        for i in range(n_downsampling):
            mult   = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, 
                                         int(ngf * mult / 2),
                                         kernel_size=3, 
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=use_bias),
                      self.norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [self.pad_factory.create(setting['models']['resNet']['generator']['paddingType'])(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=(7, 7), padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.setting['base']['isDataParallel']:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class ResnetBlock(BaseModel):
    def __init__(self, dim, setting):
        super(ResnetBlock, self).__init__(setting)
        self.pad_factory = PadFactory(self.setting)
        self.norm_layer  = NormalizeFactory(setting).create(
                                setting['models']['resNet']['generator']['normalizeLayer'])

        self.conv_block  = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        use_bias = self.setting['models']['resNet']['generator']['useBias']
        conv_block = []
        conv_block += [self.pad_factory.create(self.setting['models']['resNet']['generator']['paddingType'])(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       self.norm_layer(dim),
                       nn.ReLU(True)]

        if self.setting['models']['resNet']['generator']['useDropout']:
            conv_block += [nn.Dropout(float(self.setting['models']['resNet']['generator']['dropoutRate']))]

        conv_block += [self.pad_factory.create(self.setting['models']['resNet']['generator']['paddingType'])(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       self.norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
        if self.setting['base']['isDataParallel']:
            return x + nn.parallel.data_parallel(self.conv_block, x, self.gpu_ids)
        else:
            return x + self.conv_block(x)
