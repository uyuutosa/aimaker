import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x):
        return x


