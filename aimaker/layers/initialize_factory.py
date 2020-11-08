import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.init as init
import functools
import torchvision
import os
import random
import itertools
from aimaker.utils.util import fcnt, fcnt_load
    
class InitializeFactory:
    def __init__(self, config):
        self.config = config

        self.init_dic = {
                         "normal"     : weights_init_normal,
                         "xavier"    : weights_init_xavier,
                         "kaiming"   : weights_init_kaiming,
                         "orthogonal": weights_init_orthogonal
                        }

    def create(self, name):
        if not name in self.init_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                      '{}. choose {}')\
                                      .format(name, self.__class__.__name__, self.init_dic.keys()))
        return self.init_dic[name]#(self.config)

    #def create(self):
    #    self.is_exist()
    #    return init_dir[name] 

    #def is_exist(self, name):
    #    if self.opt.initializer:
    #        raise NotImplementedError('initialization method [%s] is not implemented' % self.config["global"].initializer)

    
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__ # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Lienar') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

