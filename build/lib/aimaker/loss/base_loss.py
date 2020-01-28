import torch.nn as nn

import aimaker.utils as util


class BaseLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = util.SettingHandler(config)

    def __call__(self, input, target):
        pass

    def cuda(self, device=None):
        if device == -1:
            return super().cpu()
        else:
            return super().cuda(device)
