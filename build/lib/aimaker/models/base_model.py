import torch.nn as nn
from torch.autograd import Variable

from aimaker.utils  import SettingHandler

class BaseModel(nn.Module):
    def __init__(self, settings):
        super(BaseModel, self).__init__()
        if settings is not None:
            self.setSetting(settings)
            ch = self.ch = SettingHandler(settings)
            self.gpu_ids = ch.get_GPU_ID()
        else:
            self.gpu_ids = [-1]

            
        self.feature_image_dic = {}

    def setSetting(self, settings):
        self.settings = settings

    def getFeature(self):
        return self.feature_image_dic

    def _setFeatureForView(self, name, x):
        name = self.__class__.__name__ + "_" + name
        if name in self.settings['ui']['base']['visdomImageTags']:
            self.feature_image_dic.update({name: x.data[0:1].view(-1, 1, *x.shape[-2:])})

    def _setParameterForView(self, name, layer):
        name = self.__class__.__name__ + "_" + name + "_param"
        if name in self.settings['ui']['base']['visdomImageTags']:
            param = list(layer.parameters())[0]
            param = (param - param.min()) / (param.max() - param.min()) 
            self.feature_image_dic.update({name: param.data.view(-1, 1, *param.shape[-2:])})

    def foward(self):
        pass


    def cuda(self, device=None):
        if device == -1:
            return super().cpu()
        else:
            return super().cuda(device)

