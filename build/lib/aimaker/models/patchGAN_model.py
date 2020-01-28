import torch
import torch.nn as nn

from aimaker.utils import SettingHandler
from aimaker.models.base_model import BaseModel



class MinibatchDiscrimination(torch.nn.Module):
   def __init__(self, n_in, n_B, n_C):
       super().__init__()
       self.linear = torch.nn.Linear(in_features=n_in, out_features=n_B * n_C)
       torch.nn.init.xavier_uniform_(self.linear.weight)
       self.n_B = n_B
       self.n_C = n_C

   def forward(self, x):
       nb = x.shape[0]
       m = self.linear(x).reshape(-1,  self.n_B, self.n_C)
       o = torch.exp(-abs(m[None]  -  m[:, None] ).sum(-1)).sum(0)
       return torch.cat((x, o), dim=1)

class PatchGANModel(BaseModel):
    def __init__(self, 
                 settings,
                 gpu_ids=[]):

        super(PatchGANModel, self).__init__(settings)


        # get global
        self.settings  = settings
        ch           = SettingHandler(settings)
        self.gpu_ids = ch.getGPUID()
        norm_layer   = ch.getNormLayer()

        n = 1
        if settings['base']['controller'] == 'pix2pix' or settings['base']['controller'] == 'pix2pixMulti':
            n = 2 # for image pooling
        input_nc = settings['models']['patchGANDiscriminator']['input_n'] * n

        use_bias = True
#        if type(norm_layer) == functools.partial:
#            use_bias = norm_layer.func == nn.InstanceNorm2d
#        else:
#            use_bias = norm_layer == nn.InstanceNorm2d

        
        print(settings['models'])
        kw         = int(settings['models']['patchGANDiscriminator']['kernelSize'])
        padw       = int(settings['models']['patchGANDiscriminator']['paddingSize'])
        ndf        = int(settings['models']['patchGANDiscriminator']['numberOfDiscriminatorFilters'])
        n_layers   = int(settings['models']['patchGANDiscriminator']['nLayers'])
        is_sigmoid = settings['models']['patchGANDiscriminator']['useSigmoid']


        sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult      = min(2**n, 8)
            sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult      = min(2**n_layers, 8)
        sequence += [
                nn.Conv2d(ndf * nf_mult_prev, 
                          ndf * nf_mult,
                          kernel_size=kw, 
                          stride=1,
                          padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, 
                     1, 
                     kernel_size=kw, 
                     stride=1, 
                     padding=padw)]

        if is_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
        #if self.settings['global'].getboolean('isDataParallel'):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        #    return self.model(input)

