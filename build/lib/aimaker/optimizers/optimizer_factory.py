import torch
import torch.nn as nn
import torch.optim as optim

import aimaker.utils.util as util


class OptimizerFactory:
    def __init__(self, config):
        self.config   = config
        self.optimizer_dic = {"adam": AdamOptimizer,
                             }
    def create(self, name):
        self._is_exist(name)
        return self.optimizer_dic[name]#(self.config)

    def _is_exist(self, name):
        if not name in self.optimizer_dic:
            raise NotImplementedError('optimizer [%s] is not implemented' % name)
    
class AdamOptimizer(optim.Adam):
    def __init__(self, params, config):
        super(AdamOptimizer, self).__init__(params, lr=float(config['optimizer']['base']['lr']),
                                            betas=tuple(float(x) for x in config['optimizer']['base']['betas'].split(",")),
                                            eps=float(config['optimizer']['base']['eps']),
                                            weight_decay=float(config['optimizer']['base']['weightDecay']),
                                           )
