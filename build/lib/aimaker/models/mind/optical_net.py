import torch
import torch.nn as nn
from torch.nn import functional as F

from aimaker.models.base_model import BaseModel

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1)) 
        return self.relu(bottle)
    
class ResizeConv(torch.nn.Module):
    def __init__(self, n_in, n_out, scale_factor=2):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        self.norm = torch.nn.BatchNorm2d(n_out)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class DownSample(torch.nn.Module):
    def __init__(self, n_in, n_out, scale_factor=2):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=scale_factor, padding=0)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.act(x)
        return x
    
class ResBlock(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(n_in, n_in, kernel_size=3, stride=1, padding=0)
        self.norm = torch.nn.BatchNorm2d(n_in)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        x_ini = x
        x = self.pad(x) 
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x + x_ini
    
class AdaIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        epsilon=1e-5
        c_mean, c_var = x.mean([2,3], keepdims=True), x.var([2,3], keepdims=True)
        s_mean, s_var = y.mean([1], keepdims=False)[:,None,None,None], y.var([1], keepdims=False)[:,None,None,None]
        c_std, s_std = torch.sqrt(c_var + epsilon), torch.sqrt(s_var + epsilon)
        return s_std * (x - c_mean) / c_std + s_mean
    
class DecoderWithContractionPath(torch.nn.Module):
    def __init__(self, n_in_lst=[12, 24, 32, 64, 160][::-1], n_out=3):
        super().__init__()
        self.layer_1 = self._add_layer(n_in_lst[0], n_in_lst[1])
        self.layer_2 = self._add_layer(2*n_in_lst[1], n_in_lst[2])
        self.layer_3 = self._add_layer(2*n_in_lst[2], n_in_lst[3])
        self.layer_4 = self._add_layer(2*n_in_lst[3], n_in_lst[4], scale_factor=4)
        self.layer_5 = self._add_layer(n_in_lst[4]+3, n_in_lst[4], scale_factor=1)
        self.output = self._add_last_layer(n_in_lst[4], n_out)
        self.adain = AdaIN()
    
    def _add_layer(self, n_in, n_out, scale_factor=2):
        up = ResizeConv(n_in, n_out, scale_factor)
        block = ResBlock(n_out)
        block_2 = ResBlock(n_out)
        return torch.nn.Sequential(up, block, block_2)
    
    
    def _add_last_layer(self, n_in, n_out):
        pad = torch.nn.ReflectionPad2d(1)
        conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        #norm = torch.nn.BatchNorm2d(n_out)
        #act = torch.nn.Tanh()
        return torch.nn.Sequential(pad, conv)#, act)
    
    def forward(self, feat_lst, feat_cls):
        x = self.layer_1(feat_lst[-1])
        x = self.adain(x, feat_cls)
        x = self.layer_2(torch.cat((feat_lst[-2], x), dim=1))
        x = self.adain(x, feat_cls)
        f = x
        x = self.layer_3(torch.cat((feat_lst[-3], x), dim=1))
        x = self.adain(x, feat_cls)
        x = self.layer_4(torch.cat((feat_lst[-4], x), dim=1))
        x = self.adain(x, feat_cls)
        x = self.layer_5(torch.cat((feat_lst[-5], x), dim=1))
        return self.output(x), f
    
class DecoderWithoutContractionPath(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        scale_factor = 2
        self.psp = PSPModule(n_in, 256)
        self.up1 = ResizeConv(256, 256, scale_factor)
        self.up2 = ResizeConv(256, 256, scale_factor)
        self.up3 = ResizeConv(256, 256, scale_factor)
        self.up4 = ResizeConv(256, 256, scale_factor)
        self.up5 = ResizeConv(256, 512, scale_factor)
        self.out = nn.Sequential(nn.Conv2d(512, n_out, kernel_size=3, stride=1, padding=1), nn.Tanh())
    
    def forward(self, x):
        x = self.psp(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.out(x)
    
class FeatureNet(torch.nn.Module):
    def __init__(self, is_pretrained=True, is_freeze=True, feat_idx=[3,5,8,15]):
        super().__init__()
        model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=is_pretrained)
        self.model = self._build_model_list(model, feat_idx)
        self.feat_idx= feat_idx
        self.block = nn.Sequential(ResBlock(160), ResBlock(160), ResBlock(160))
        
        if is_freeze:
            for param in model.parameters():
                param.requires_grad = False
                
    def _build_model_list(self, model, feat_idx):
        m_lst = list(model.children())[0]
        idx_lst = [0] + feat_idx
        for i in range(len(idx_lst)-1):
            setattr(self, "layer_{:03d}".format(i),torch.nn.Sequential(m_lst[idx_lst[i]:idx_lst[i+1]]))
            
        
    def forward(self, x):
        idx_lst = [0] + self.feat_idx
        x_lst = [x]
        for i in range(len(idx_lst)-1):
            x = getattr(self, "layer_{:03d}".format(i))(x)
            x_lst += [x]
        x_lst[-1] = self.block(x_lst[-1])
        return x_lst
    
class ClassEncoder(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.conv_64 = DownSample(n_in, 64)
        self.conv_128 = DownSample(64, 128)
        self.conv_256 = DownSample(128, 256)
        self.conv_512 = DownSample(256, 512)
        self.conv_1024 = DownSample(512, 1024)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_1 = torch.nn.Sequential(torch.nn.Linear(1024, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.fc_2 = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        self.fc_3 = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU())
        
    def forward(self, x):
        x = self.conv_64(x)
        x = self.conv_128(x)
        x = self.conv_256(x)
        x = self.conv_512(x)
        x = self.conv_1024(x)
        x = self.pool(x).reshape(-1, 1024)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x
    
class OpticalNet(BaseModel):
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        if settings is not None:
            self.n_in = 3
            self.n_out = 9
            
        self.feat = FeatureNet()
        self.class_enc = ClassEncoder(self.n_in)
        self.dec_with_cont = DecoderWithContractionPath(n_out=self.n_out)
        self.dec_without_cont = DecoderWithoutContractionPath(n_in=160, n_out=1)
    
    def forward(self, input):
        x, x_cls = input
        feat_lst = self.feat(x)
        feat_cls = self.class_enc(x_cls)
        
        out, f = self.dec_with_cont(feat_lst, feat_cls)
        tex, diff, depth, model_light = out[:, 0:3], out[:, 3:6], out[:, 6:7], out[:, 7:8]#, out[:, 8:]
        light_source = self.dec_without_cont(feat_lst[-1])
        return tex, diff, depth, model_light, light_source, f # params
        

class DecoderWithContractionPath2(torch.nn.Module):
    def __init__(self, n_in_lst=[12, 24, 32, 64, 160][::-1], n_out=3):
        super().__init__()
        self.layer_1 = self._add_layer(n_in_lst[0], 1024)
        self.layer_2 = self._add_layer(n_in_lst[1] + 1024, 512)
        self.layer_3 = self._add_layer(n_in_lst[2] + 512,  256)
        self.layer_4 = self._add_layer(n_in_lst[3] + 256,  128, scale_factor=4)
        self.layer_5 = self._add_layer(n_in_lst[4] + 128 - 9,  64, scale_factor=1)
        self.output = self._add_last_layer(64, n_out)
    
    def _add_layer(self, n_in, n_out, scale_factor=2):
        up = ResizeConv(n_in, n_out, scale_factor)
        block = ResBlock(n_out)
        block_2 = ResBlock(n_out)
        return torch.nn.Sequential(up, block, block_2)
    
    
    def _add_last_layer(self, n_in, n_out):
        pad = torch.nn.ReflectionPad2d(1)
        conv = torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=0)
        #norm = torch.nn.BatchNorm2d(n_out)
        #act = torch.nn.Tanh()
        return torch.nn.Sequential(pad, conv)#, act)
    
    def forward(self, feat_lst):
        x = self.layer_1(feat_lst[-1])
        x = self.layer_2(torch.cat((feat_lst[-2], x), dim=1))
        x = self.layer_3(torch.cat((feat_lst[-3], x), dim=1))
        x = self.layer_4(torch.cat((feat_lst[-4], x), dim=1))
        x = self.layer_5(torch.cat((feat_lst[-5], x), dim=1))
        return self.output(x)


class OpticalNet2(BaseModel):
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        if settings is not None:
            self.n_in = 3
            self.n_out = 6
            
        self.feat = FeatureNet()
        self.dec_with_cont = DecoderWithContractionPath2(n_out=self.n_out)
    
    def forward(self, input):
        x = input
        feat_lst = self.feat(x)
        
        out = self.dec_with_cont(feat_lst)
        return out[:, 0:3], out[:, 3:]
