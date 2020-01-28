from easydict import EasyDict
import json
import os

import torchvision.models as m

from aimaker.models.base_model import BaseModel
from aimaker.layers import *
from aimaker.utils import BaseFactory
from aimaker.layers import ConvBlock, ResizeConv, DownSample, ResBlock, ActivationFactory

#class BaseFactory:
#    def __init__(self, settings, module):
#        self.dic = dir(module)
#        self.settings = settings
#        self.module_name = None
#
#    def _importIfExists(self, name):
#        try:
#            self.mod = importlib.import_module("{}".format(self.module_name))
#            print('self.data_dic.update({"%s":self.mod.%s})' %(name, name))
#            exec('self.data_dic.update({"%s":self.mod.%s})' %(name, name))
#        except:
#            pass
#            #import traceback
#            #traceback.print_exc()
#            #raise NotImplementedError(('{} is wrong key word for ' + \
#            #                           '{}. choose {}')\
#            #                           .format(name, self.__class__.__name__, self.data_dic.keys()))
#        if self.module_name is not None:
#            raise ImportError(f"{self.module_name} is not None, but it couldn't be imported.")
#
#    def _create(self, name):
#        pass
#
#    def create(self, name):
#        self._importIfExists(name)
#        return self._create(name)
#

#class TrainedEncoderFactory(BaseFactory):
#    def __init__(self, settings):
#        super().__init__(settings)
#        self.module_name = self.settings.data.base.importModule
#
#    def _create(self, name):
#        names = name.split("_")
#        encoder = eval(f"{name}(settings=self.settings)")
#        InitializeFactory(self.settings).create(self.settings.base.initType)(encoder)
#        return encoder

class SimpleEncoder(BaseModel):
    def __init__(self, settings, **kwargs):
        super().__init__(settings)
        if settings is not None:
            s = settings.models.handy_unet.simple_encoder
            self.n_in = s.n_in
            self.n_in_lst = s.n_in_lst
            self.act = s.act
            self.n_res_block_lst = s.n_res_block_lst
        elif kwargs is not None:
            s = kwargs
            self.n_in = s.n_in
            self.n_in_lst = s.n_in_lst
            self.act = s.act
            self.n_res_block_lst = s.n_res_block_lst
        else:
            self.n_in = 3
            self.n_in_lst = [64, 128, 256, 512]
            self.act = "ReLU"
            self.n_res_block_lst = 1

        if isinstance(self.n_res_block_lst, int):
            self.n_res_block_lst = [self.n_res_block_lst] * (len(self.n_in_lst) - 1)

        self.input_layer = self._add_input_layer(self.n_in, self.n_in_lst[0])
        for i, (n_in, n_out, n_res_block) in enumerate(zip(self.n_in_lst[:-1], self.n_in_lst[1:], self.n_res_block_lst)):
            setattr(self, f"layer_{i+1}", self._add_layer(n_in, n_out, n_res_block))

        self.n_in_f_lst = self.n_in_lst
    
    def _add_layer(self, n_in, n_out, n_res_block, scale_factor=2):
        res_lst = []
        for i in range(n_res_block):
            res_lst += [ResBlock(n_in)]
        down = DownSample(n_in, n_out, scale_factor)
        return torch.nn.Sequential(*res_lst, down)
    
    def _add_input_layer(self, n_in, n_out):
        pad = torch.nn.ReflectionPad2d(1)
        conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        norm = torch.nn.BatchNorm2d(n_out)
        act = ActivationFactory(self.settings).create(self.act)
        return torch.nn.Sequential(pad, conv, norm, act)
    
    def forward(self, x):
        x_lst = []
        x = self.input_layer(x)
        x_lst += [x]
        for i in range(len(self.n_in_lst)-1):
            x = eval(f"self.layer_{i+1}(x)")
            x_lst += [x]
        return x_lst

class TrainedEncoderFactory():
    def __init__(self, settings):
        self.adaptor_dic = {
                            "vgg16": "m.{name}(pretrained=is_pretrianed)",
                            "vgg19": "m.{name}(pretrained=is_pretrianed)",
                            "mobilenet_v2": "torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=is_pretrained)"
                            }
        self.settings = settings

    def create(self, name, is_pretrianed=True, is_freeze=False):
        encoder = eval(self.adaptor_dic[name].format(name=name))
        InitializeFactory(self.settings).create(self.settings.base.initType)(encoder)
        return encoder

class MobileNetV2Encoder(BaseModel):
    def __init__(self, settings=None, **kwargs):
        super().__init__(settings)

        if settings is not None:
            s = settings.models.handy_unet.trained_encoder
            self.is_pretrained=s.is_pretrained
            self.is_freeze=s.is_freeze
            self.feat_idx = s.feat_idx
            self.n_in = s.mobilenetv2.n_in
        elif len(kwargs) != 0:
            kwargs = EasyDict(kwargs)
            self.is_pretrained=kwargs.is_pretrained
            self.is_freeze=kwargs.is_freeze
            self.feat_idx=kwargs.feat_idx
            self.n_in = kwargs.n_in
        else:
            self.is_pretrained=True
            self.is_freeze=True
            self.feat_idx=[3,5,8,15]
            self.n_in = 160
        
        self.block = nn.Sequential(ResBlock(self.n_in), ResBlock(self.n_in), ResBlock(self.n_in))
        model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=self.is_pretrained)
        self.model = self._build_model_list(model, self.feat_idx)
        if self.is_freeze:
            for param in model.parameters():
                param.requires_grad = False

        self.n_in_lst_f = [160, 64, 32, 24, 12] # filter size of each layer
                
    def _build_model_list(self, model, feat_idx):
        m_lst = list(model.children())[0]
        idx_lst = [0] + feat_idx
        for i in range(len(idx_lst)-1):
            setattr(self, "layer_{:03d}".format(i),torch.nn.Sequential(m_lst[idx_lst[i]: idx_lst[i+1]]))
        
    def forward(self, x):
        idx_lst = [0] + self.feat_idx
        ret_dic = EasyDict({"class_00": x})
        #x_lst = [x]
        for i in range(len(idx_lst)-1):
            x = getattr(self, "layer_{:03d}".format(i))(x)
            ret_dic[f'class_{i+1:02d}'] = x
#            x_lst += [x]
        return ret_dic
        #x =  self.block(x)
        #x_lst[-1] = self.block(x_lst[-1])
        #return x_lst


class HandyEncoder(BaseModel):
    def __init__(self, settings=None, **kwargs):
        super().__init__(settings)

        if settings is not None:
            s = settings.models.handy_unet.handy_encoder
            self.n_in = s.n_in
            self.model_name = s.model_name
            self.is_pretrained=s.is_pretrained
            self.is_freeze=s.is_freeze
            self.feat_idx = self._parseLayer(s.layer_names)
        elif len(kwargs) != 0:
            kwargs = EasyDict(kwargs)
            self.n_in = kwargs.n_in
            self.model_name = kwargs.model_name
            self.is_pretrained=kwargs.is_pretrained
            self.is_freeze=kwargs.is_freeze
            self.feat_idx=kwargs.feat_idx
        else:
            self.n_in = 3
            self.model_name = "vgg16"
            self.is_pretrained = True
            self.is_freeze = True
            self.feat_idx = ["relu1_1"]
        
        model = TrainedEncoderFactory(settings).create(self.model_name)
        self.model = self._build_model_list(model, self.feat_idx)
        if self.is_freeze:
            for param in model.parameters():
                param.requires_grad = False

        self.n_in_f_lst = [x.shape[1] for x in self(torch.Tensor(1, self.n_in, 128, 128))]#[160, 64, 32, 24, 12] # filter size of each layer

    def _parseLayer(self, layer_names):
        name_dic = json.load(open(os.path.join('/'.join(__file__.split('/')[:-1]), "name_definition_" + self.model_name + '.json')))
        name_lst = [x.strip() for x in layer_names.split(",")]
        ret_lst = []
        for i in name_lst:
            try:
                ret_lst += [int(i)]
            except:
                ret_lst += [name_dic[i]]
        return ret_lst


                
    def _build_model_list(self, model, feat_idx):
        m_lst = list(model.children())[0]
        idx_lst = [0] + feat_idx
        for i in range(len(idx_lst)-1):
            setattr(self, "layer_{:03d}".format(i),torch.nn.Sequential(m_lst[idx_lst[i]: idx_lst[i+1]]))
        
    def forward(self, x):
        idx_lst = [0] + self.feat_idx
        #ret_dic = EasyDict({"class_00": x})
        x_lst = []
        for i in range(len(idx_lst)-1):
            x = getattr(self, "layer_{:03d}".format(i))(x)
            #ret_dic[f'class_{i+1:03d}'] = x
            x_lst += [x]
        return x_lst
        #return ret_dic
