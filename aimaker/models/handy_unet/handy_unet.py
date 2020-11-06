import torch
import torch.nn as nn
from torch.nn import functional as F
from aimaker.models.base_model import BaseModel
from aimaker.models import ModelFactory
#from aimaker.models import TrainedEncoderFactory
from aimaker.layers import ConvBlock, ResizeConv, DownSample, ResBlock, ActivationFactory



class HandyDecoder(BaseModel):
    def __init__(self, settings, n_in_f_lst):
        super().__init__(settings)
        self.n_in_f_lst = n_in_f_lst[::-1]
        if settings is not None:
            s = settings.models.handy_unet.handy_decoder
            self.n_in_lst = s.n_in_lst
            self.n_out = s.n_out
            self.act = s.act
            self.act_last = s.act_last
            self.n_res_block_lst = s.n_res_block_lst
        elif kwargs is not None:
            s = kwargs
            self.n_in_lst = s.n_in_lst
            self.n_out = s.n_out
            self.act = s.act
            self.act_last = s.act_last
            self.n_res_block_lst = s.n_res_block_lst
        else:
            self.n_in_lst = [1028, 512, 256, 128, 64]
            self.n_out=3
            self.act = "ReLU"
            self.act_last = "Tanh"
            self.n_res_block_lst = 1

        setattr(self, f"layer_1", ResizeConv(self.n_in_f_lst[0], self.n_in_lst[0]))
        print(self.n_res_block_lst[1:-1])
        input(self.n_res_block_lst)
        for i, (n_in_f, n_in, n_out, n_res_block) in enumerate(zip(self.n_in_f_lst[1:], self.n_in_lst[:-1], self.n_in_lst[1:], self.n_res_block_lst[1:-1])):
            print(n_in_f, n_in, n_out, n_res_block)
            print(f"layer_{i+2}")
            setattr(self, f"layer_{i+2}", self._add_layer(n_in_f + n_in, n_in, n_out, n_res_block))
        self.output = self._add_last_layer(self.n_in_f_lst[-1] + self.n_in_lst[-1], self.n_out, self.n_res_block_lst[-1])

        if isinstance(self.n_res_block_lst, int):
            self.n_res_block_lst = [self.n_res_block_lst] * (len(self.n_in_lst) - 1)
    

    def _add_first_layer(self, n_in, n_out):
        pad = torch.nn.ReflectionPad2d(1)
        conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        #norm = torch.nn.BatchNorm2d(n_out)
        act = ActivationFactory(self.settings).create(self.act_last)
        return torch.nn.Sequential(pad, conv, act)
    def _add_layer(self, n_in, n_mid, n_out, n_res_block, scale_factor=2):
        conv = ConvBlock(n_in, n_mid)
        res_lst = []
        for i in range(n_res_block):
            res_lst += [ResBlock(n_mid)]
        up = ResizeConv(n_mid, n_out, scale_factor)
        return torch.nn.Sequential(conv, *res_lst, up)
    
    def _add_last_layer(self, n_in, n_out, n_res_block):
        res_lst = []
        for i in range(n_res_block):
            res_lst += [ResBlock(n_in)]
        pad = torch.nn.ReflectionPad2d(1)
        conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        #norm = torch.nn.BatchNorm2d(n_out)
        act = ActivationFactory(self.settings).create(self.act_last)
        return torch.nn.Sequential(*res_lst, pad, conv, act)
    
    def forward(self, feat_lst):
        x = self.layer_1(feat_lst[-1])
        for i, feat in enumerate(feat_lst[:-1][::-1][:-1]):
            x = eval(f"self.layer_{i+2}(torch.cat((feat, x), dim=1))")

        return self.output(torch.cat((feat_lst[0], x), dim=1))

class HandyBottleNeck(BaseModel):
    def __init__(self, settings, n_in, **kwargs):
        super().__init__(settings,  **kwargs)
        self.n_in = n_in
        if settings is not None:
            s = settings.models.handy_unet.handy_bottle_neck
            self.n_res_block = s.n_res_block
            self.is_skip = s.is_skip
        elif kwargs is not None:
            s = kwargs
            self.n_res_block = s.n_res_block
            self.is_skip = s.is_skip
        else:
            self.n_res_block = 1
            self.is_skip = True

        for i in range(self.n_res_block):
            setattr(self, f"layer_{i+1}", ResBlock(self.n_in, self.is_skip))
    

    def forward(self, x):
        for i in range(self.n_res_block):
            x = eval(f"self.layer_{i+1}(x)")
        return x


class HandyUNet(BaseModel):
    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        self.settings = settings
            
        print(settings.models.handy_unet.base.encoder_name)
        self.enc = ModelFactory(settings, is_base_sequential=False).create(settings.models.handy_unet.base.encoder_name)
        self.bottle_neck = HandyBottleNeck(settings, self.enc.n_in_f_lst[-1])
        self.dec = HandyDecoder(settings, self.enc.n_in_f_lst)
    
    def forward(self, x):
        x_lst = self.enc(x)
        x_lst[-1] = self.bottle_neck(x_lst[-1])
        x = self.dec(x_lst)
        return x
