import torch
import torch.nn as nn
import torch.nn.functional as F
import aimaker.utils.util as util
from aimaker.loss.GAN_loss     import GANLoss
from aimaker.loss.feature_loss import FeatureLoss, TFLoss
from aimaker.loss.mix_loss import MixLoss
from aimaker.loss.distance_loss import DistanceLoss
import aimaker.loss.base_loss as bs

class LossFactory:
    def __init__(self, setting):
        self.setting   = setting
        self.loss_dic = {"GANLoss"       : GANLoss,
                         "feature"       : FeatureLoss,
                         "L1"            : L1Loss,
                         "MSE"           : MSELoss,
                         "BCE"           : BCELoss,
                         "BCEwithLogits" : BCEwithLogitsLoss,
                         "mix"           : MixLoss,
                         "abs"           : AbsLoss,
                         "distance"      : DistanceLoss,
                         "TF"            : TFLoss
                        }
    def create(self, name):
        self._is_exist(name)
        return self.loss_dic[name](self.setting)

    def _is_exist(self, name):
        if not name in self.loss_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                   .format(name, self.__class__.__name__, self.loss_dic.keys()))


class AbsLoss(nn.Module):
    def __init__(self, setting):
        super(AbsLoss, self).__init__()

    def forward(self, x, y):
        z = ((x - y)**2)
        z = torch.sum(z, dim=1)
        return z.sum()
        

    def cuda(self, device=None):
        if device == -1:
            return super().cpu()
        else:
            return super().cuda(device)

class MSELoss(nn.MSELoss):
    def __init__(self, setting):
        super(MSELoss, self).__init__(
            size_average=setting['loss']['base']['sizeAverage'],
            reduce=setting['loss']['base']['reduce']
        )

    def cuda(self, device=None):
        if device == -1:
            return super().cpu()
        else:
            return super().cuda(device)

class BCELoss(nn.BCELoss):
    def __init__(self, setting):
        super(BCELoss, self).__init__(
            weight=None,#setting['loss settings'].getboolean('weight'),
            size_average=setting['loss']['base']['sizeAverage'],
#            reduce=setting['loss settings'].getboolean('reduce')
        )

    def cuda(self, device=None):
        if device == -1:
            return super.cpu()
        else:
            return super.cuda(device)

class BCEwithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, setting):
        super(BCEwithLogitsLoss, self).__init__(
            weight=None, #setting['loss settings'].getboolean('weight'),
            size_average=setting['loss']['base']['sizeAverage'],
            #reduce=setting['loss settings'].getboolean('reduce')
        )

    def cuda(self, device=None):
        if device == -1:
            return super.cpu()
        else:
            return super.cuda(device)

class L1Loss(nn.L1Loss):
    def __init__(self, setting):
        super(L1Loss, self).__init__(
            size_average=setting['loss']['base']['sizeAverage'],
            reduce=setting['loss']['base']['reduce']
        )

    def cuda(self, device=None):
        if device == -1:
            return super().cpu()
        else:
            return super().cuda(device)

