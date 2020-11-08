import functools
import torch.nn as nn


class NormalizeFactory:
    def __init__(self, settings):
        self.model_dic = {'batch': BatchNorm2dModel,
                          'instance': InstanceNorm2dModel,
                          }
        self.settings = settings

    def create(self, name):
        if not name in self.model_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                       .format(name, self.__class__.__name__, self.model_dic.keys()))

        return self.model_dic[name](self.settings)


def BatchNorm2dModel(settings):
    eps = settings['models']['normalize']['batch']['eps'] if settings is not None else 1e-05
    momentum = settings['models']['normalize']['batch']['momentum'] if settings is not None else 0.1
    affine = settings['models']['normalize']['batch']['affine'] if settings is not None else True
    return functools.partial(nn.BatchNorm2d, 
                      eps=float(eps),
                      momentum=float(momentum),
                      affine=affine)
        
def InstanceNorm2dModel(settings):
    eps = settings['models']['normalize']['instance']['eps'] if settings is not None else 1e-05
    momentum = settings['models']['normalize']['instance']['momentum'] if settings is not None else 0.1
    affine = settings['models']['normalize']['instance']['affine'] if settings is not None else True
    return functools.partial(nn.InstanceNorm2d, 
                      eps=float(eps),
                      momentum=float(momentum),
                      affine=affine)

#    def __init__(self, settings):
#        super(BatchNorm2dModel, num_features, self).__init__(
#            num_features,
#            eps=float(settings['models']['normalize']['batch']['eps']),
#            momentum=float(settings['models']['normalize']['batch']['momentum']),
#            affine=settings['models']['normalize']['batch'].getboolean('affine'),
#        )
#        
#class InstanceNorm2dModel(nn.InstanceNorm2d):
#    def __init__(self, settings):
#        super(BatchNorm2dModel, num_features, self).__init__(
#            num_features,
#            eps=float(settings['models']['normalize']['instance']['eps']),
#            momentum=float(settings['models']['normalize']['instance']['momentum']),
#            affine=settings['models']['normalize']['instance'].getboolean('affine'),
#        )
#        
        

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
    
