import torch
import torch.nn as nn

class ActivationFactory:
    def __init__(self, config):
        self.act_dic = {'relu'      : nn.ReLU,
                        'relu6'     : nn.ReLU6,
                        'elu'       : nn.ELU,
                        'selu'      : nn.SELU,
                        'prelu'     : nn.PReLU,
                        'tanh'      : nn.Tanh,
                        'LeakyReLU' : nn.LeakyReLU,
                        'sigmoid'   : nn.Sigmoid,
                          }
        self.config = config

    def create(self, name):
        if not name in self.act_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                       .format(name, self.__class__.__name__, self.act_dic.keys()))

        return self.act_dic[name]() 


        

#def get_norm_layer(norm_type='instance'):
#    if norm_type == 'batch':
#        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
#    elif norm_type == 'instance':
#        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
#    elif norm_type == 'none':
#        norm_layer = None
#    else:
#        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#    return norm_layer
    
