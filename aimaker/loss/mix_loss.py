import aimaker.utils.util as util
import torch.nn as nn
import torch.autograd as autograd
import aimaker.loss.base_loss as bs


class MixLoss(bs.BaseLoss):
    def __init__(self, config):
        super(MixLoss, self).__init__(config)

        self.loss_lst = self._parseLosses()
        self.alpha_lst = [float(x) for x in config['mix loss settings']['alphas'].split(",")]

    def _parseLosses(self):
        from aimaker.loss.loss_factory import LossFactory
        loss_lst = []

        for name in self.config['mix loss settings']['lossNames'].split(","):
            loss_lst += [LossFactory(self.config).create(name)]

        return loss_lst


    def __call__(self, input, target):
        losses = 0
        for alpha, loss in zip(self.alpha_lst, self.loss_lst):
            losses = losses + alpha * loss(input, target)

        return losses

    def backward(self, retain_graph=True):
        for loss in self.loss_lst:
            loss.backward(retain_graph=True)
                 
