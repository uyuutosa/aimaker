import torch.nn as nn
import aimaker.utils.util as util

from aimaker.models.normalize_factory import NormalizeFactory
from aimaker.layers.pad_factory import PadFactory
from aimaker.models.base_model import BaseModel


class ResNetForClassificationModel(nn.Module):
    def __init__(self, config):
        super(ResNetForClassificationModel, self).__init__()
        ch         = util.ConfigHandler(config)
        n_trim     = int(config['ResNet settings']['nTrim'])
        self.net   = ch.get_res_net()
        if config['ResNet settings'].getboolean('trimFullConnectedLayer'):
            self.net = nn.Sequential(*list(self.net.children())[:-n_trim])

    def forward(self, x):
        return self.net(x)


class BGModel(BaseModel):
    def __init__(self,
                 config):

        super(BGModel, self).__init__(config)
        self.config = config
        ch = util.ConfigHandler(config)
        input_nc         = int(config['global settings']['numberOfInputImageChannels'])
        ngf              = int(config['ResNetForGenerator settings']['filterSize'])
        use_bias         = config['ResNetForGenerator settings'].getboolean('useBias')
        self.gpu_ids     = ch.get_GPU_ID()
        self.pad_factory = PadFactory(config)
        self.norm_layer  = NormalizeFactory(config).create(
                                config['ResNetForGenerator settings']['normalizeLayer'])

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

        for i in range(int(config['ResNetForGenerator settings']['nBlocks'])):
            model += [ResnetBlock(ngf * mult, config)] 

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

        model += [self.pad_factory.create(config['ResNetForGenerator settings']['paddingType'])(7)]
        model += [nn.Conv2d(ngf,    ngf*2, kernel_size=(5, 1), padding=0, stride=(2,2))]
        model += [nn.Conv2d(ngf*2,  ngf*4, kernel_size=(13, 1), padding=0, stride=(2,2))]
        model += [nn.Conv2d(ngf*4,  ngf*8, kernel_size=(13, 7), padding=0, stride=(2,1))]
        model += [nn.Conv2d(ngf*8,  ngf*12, kernel_size=(13, 15), padding=0, stride=(2,1))]
        #model += [nn.Conv2d(ngf*12, ngf*12, kernel_size=(3, 3), padding=0, stride=(2,2))]
        model += [nn.Conv2d(ngf*12, 1, kernel_size=(23, 5), padding=0)]
        #model += [nn.Tanh()]
        #model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)
        self.act = nn.Sigmoid()

    def forward(self, input):
        if self.config['global settings'].getboolean('isDataParallel'):
            return self.act(nn.parallel.data_parallel(self.model, input, self.gpu_ids).view(-1,108)) * 200
        else:
            return self.act(self.model(input).view(-1,108)) * 200


class ResnetBlock(nn.Module):
    def __init__(self, dim, config):

        super(ResnetBlock, self).__init__()
        self.config = config
        self.pad_factory = PadFactory(self.config)
        self.norm_layer  = NormalizeFactory(config).create(
                                config['ResNetForGenerator settings']['normalizeLayer'])

        self.conv_block  = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        use_bias = self.config['ResNetForGenerator settings'].getboolean('useBias')
        conv_block = []
        conv_block += [self.pad_factory.create(self.config['ResNetForGenerator settings']['paddingType'])(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       self.norm_layer(dim),
                       nn.ReLU(True)]

        if self.config['ResNetForGenerator settings'].getboolean('useDropout'):
            conv_block += [nn.Dropout(float(self.config['ResNetForGenerator settings']['dropoutRate']))]

        conv_block += [self.pad_factory.create(self.config['ResNetForGenerator settings']['paddingType'])(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       self.norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BGModel2(BaseModel):
    def __init__(self,
                 config):

        super(BGModel2, self).__init__(config)
        self.config = config
        ch = util.ConfigHandler(config)
        input_nc         = int(config['global settings']['numberOfInputImageChannels'])
        output_nc        = int(config['global settings']['numberOfOutputImageChannels'])
        ngf              = int(config['ResNetForGenerator settings']['filterSize'])
        use_bias         = config['ResNetForGenerator settings'].getboolean('useBias')
        self.gpu_ids     = ch.get_GPU_ID()
        self.pad_factory = PadFactory(config)
        self.norm_layer  = NormalizeFactory(config).create(
                                config['ResNetForGenerator settings']['normalizeLayer'])

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

        for i in range(int(config['ResNetForGenerator settings']['nBlocks'])):
            model += [ResnetBlock(ngf * mult, config)] 

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

        model += [self.pad_factory.create(config['ResNetForGenerator settings']['paddingType'])(7)]
        model += [nn.Conv2d(ngf,    ngf*2, kernel_size=(5, 1), padding=0, stride=(2,2))]
        model += [nn.Conv2d(ngf*2,  ngf*4, kernel_size=(13, 1), padding=0, stride=(2,2))]
        model += [nn.Conv2d(ngf*4,  ngf*8, kernel_size=(13, 7), padding=0, stride=(2,1))]
        model += [nn.Conv2d(ngf*8,  ngf*12, kernel_size=(13, 15), padding=0, stride=(2,1))]
        #model += [nn.Conv2d(ngf*12, ngf*12, kernel_size=(3, 3), padding=0, stride=(2,2))]
        model += [nn.Conv2d(ngf*12, 1, kernel_size=(23, 97), padding=0)]
        #model += [nn.Tanh()]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.config['global settings'].getboolean('isDataParallel'):
            z = nn.parallel.data_parallel(self.model, input, self.gpu_ids).view(-1,16) * 200
            return z
        else:
            z = self.model(input).view(-1,16) * 200
            return z


