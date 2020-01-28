import torch
import torch.nn as nn

import aimaker.utils as util
from aimaker.models import BaseModel


class MinibatchDiscrimination(torch.nn.Module):
   def __init__(self, n_in, n_B, n_C):
       super().__init__()
       self.linear = torch.nn.Linear(in_features=n_in, out_features=n_B * n_C)
       torch.nn.init.xavier_uniform_(self.linear.weight)
       self.n_B = n_B 
       self.n_C = n_C 

   def forward(self, x): 
       x = x.reshape(x.shape[0], -1)
       nb = x.shape[0]
       m = self.linear(x).reshape(-1,  self.n_B, self.n_C)
       o = torch.exp(-abs(m[None]  -  m[:, None] ).sum(-1)).sum(0)
       return torch.cat((x, o), dim=1)


class PatchGANMDPlusModelForMD(BaseModel):
    def __init__(self, 
                 settings=None,
                 gpu_ids=[]):
        super().__init__(settings)
        kw         = int(4)
        padw       = int(1)
        #padw       = int(settings['models']['patchGANDiscriminator']['paddingSize'])
        ndf        = int(64)
        n_layers   = int(3)
        is_sigmoid = True
        use_bias = True
        ch            = util.SettingHandler(settings)
        norm_layer    = ch.getNormLayer()
        
        self.conv1 = nn.Conv2d(10, 64, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm1 = norm_layer(64)
        self.act1 = nn.LeakyReLU(0.2, True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm2 = norm_layer(128)
        self.act2 = nn.LeakyReLU(0.2, True)
        
#        self.md = MinibatchDiscrimination(524288, 64, 64)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm3 = norm_layer(256)
        self.act3 = nn.LeakyReLU(0.2, True)
        
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm4 = norm_layer(512)
        self.act4 = nn.LeakyReLU(0.2, True)
        
        
        
        self.conv5 = nn.Conv2d(512, 3, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
#        self.norm5 = norm_layer(3)
        self.act5 = nn.LeakyReLU(0.2, True)
        self.md = MinibatchDiscrimination(3*8*8, 8**2, 4)
        
        #self.conv6 = nn.Conv2d(3, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        self.conv6 = nn.Conv2d(3 + 1, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        #self.norm6 = norm_layer(1)
        self.act6 = nn.Sigmoid()

        self.conv5_2 = nn.Conv2d(512, 256, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
#        self.norm5 = norm_layer(3)
        self.act5_2 = nn.LeakyReLU(0.2, True)

        self.pool = torch.nn.functional.adaptive_avg_pool2d

        self.fc = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.LeakyReLU(0.2, True), torch.nn.Sequential(torch.nn.Linear(128, 5)))

    def forward(self, x):
        x = self.conv1(x)
#        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x_b = self.act4(x)
        
        x = self.conv5(x_b)
#        x = self.norm5(x)
        x = self.act5(x)
        x = self.md(x).reshape(x.shape[0], 3 + 1, 8, 8)
        
        x = self.conv6(x)
        #x = self.norm6(x)
        x = self.act6(x)

        x_2 = self.conv5_2(x_b)
        x_2 = self.act5_2(x_2)
        x_2 = self.pool(x_2, (1,1)).reshape(x.shape[0], -1)
        x_2 = self.fc(x_2)
        
    

        return x, x_2
