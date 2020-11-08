import torch
import torch.nn as nn
import torch.autograd as autograd

import aimaker.utils.util as util
import aimaker.loss.base_loss as bs


class GANLoss(bs.BaseLoss):
    def __init__(self, config):

        from aimaker.loss.loss_factory import LossFactory
        super(GANLoss, self).__init__(config)
        self.real_label = float(config['loss']['GANLoss']['realLabel'])
        self.fake_label = float(config['loss']['GANLoss']['fakeLabel'])

        self.loss = LossFactory(config).create(config['loss']['GANLoss']['lossName'])
        #if config['cycleGAN settings'].getboolean('useLsGAN'):
        #    self.loss = nn.MSELoss()
        #else:
        #    self.loss = nn.BCELoss()



    def __call__(self, input, target_is_real):
        label = self.real_label if target_is_real else self.fake_label
        return self.loss(input, autograd.Variable(torch.Tensor(input.size()).to(self.config['base']["gpu_ids"][0]).fill_(label)))
        #return self.loss(input, autograd.Variable(torch.Tensor(input.size(), device=self.config['base']["gpu_ids"][0]).fill_(label)))
        #return self.loss(input, autograd.Variable(self.Tensor(input.size(), device=self.config['base']["gpu_ids"][0])\
        #                            .fill_(label)).cuda(self.ch.getGPUID()[0]))

