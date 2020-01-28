import torch
from torch.nn import functional as F
import torch.nn as nn
#import aimaker.utils.util as util

from aimaker.models.normalize_factory import NormalizeFactory
from aimaker.layers.pad_factory import PadFactory
from aimaker.models.base_model import BaseModel
from aimaker.models import ResNetForClassificationModel


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


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, use_deconv=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            #nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.n = n
        self.use_deconv = False

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, padding=0),
            nn.InstanceNorm2d(out_channels),
            #nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def _bilinearUpsample(self, x):
        h, w = self.n * x.size(2), self.n * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p) 

    def _deconvUpsample(self, x):
        return self.deconv(x) 

    def forward(self, x):
        return self._bilinearUpsample(x)
        if self.use_deconv:
            return self._deconvUpsample(x)
        else:
            return self._bilinearUpsample(x)


class PSPNetGeneratorModel(BaseModel):
    def __init__(self, settings):
        super(PSPNetGeneratorModel, self).__init__(settings)
        n_classes           = int(settings['models']['PSPNet']['generator']['n_classes'])
        sizes               = tuple([int(x) for x in settings['models']['PSPNet']['generator']['sizes'].split(',')])
        psp_size            = int(settings['models']['PSPNet']['generator']['psp_size'])
        deep_features_size  = int(settings['models']['PSPNet']['generator']['deep_feature_size'])
        final_stride        = int(settings['models']['PSPNet']['generator']['finalStride'])
        backend             = settings['models']['PSPNet']['generator']['backend']
        self.feats          = ResNetForClassificationModel(settings)
        self.n_pre          = int(settings['models']['PSPNet']['generator']['numberOfInputImageChannels'])
        self.use_deconv     = settings['models']['PSPNet']['generator']['useDeconv']

        self.pre = nn.Conv2d(self.n_pre, 3, kernel_size=1)

        self.psp = PSPModule(psp_size, 256, sizes)
        self.drop_0 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
        self.up_2 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
        self.up_3 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
        self.up_4 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
        self.up_5 = PSPUpsample(256, 512, use_deconv=self.use_deconv)

        self.drop_1 = nn.Dropout2d(p=0.15)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.drop_3 = nn.Dropout2d(p=0.15)
        self.drop_4 = nn.Dropout2d(p=0.15)
        self.drop_5 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1, stride=final_stride),
            #nn.LogSoftmax()
            nn.Tanh()
            #nn.Sigmoid()
        )

        #self.classifier = nn.Sequential(
        #    nn.Linear(deep_features_size, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, n_classes)
        #)

    def forward(self, x):
        if self.n_pre != 3:
            x = self.pre(x)
        f = self.feats(x) 
        #f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_0(p)

        p = self.up_1(p)
        p = self.drop_1(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_3(p)

        p = self.up_4(p)
        p = self.drop_4(p)

        p = self.up_5(p)
        p = self.drop_5(p)

        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p) #, self.classifier(auxiliary)

class PSPNetGeneratorModelDownsampled(BaseModel):
    def __init__(self, settings):
        super(PSPNetGeneratorModelDownsampled, self).__init__(settings)
        n_classes           = int(settings['models']['PSPNet']['generator']['n_classes'])
        sizes               = tuple([int(x) for x in settings['models']['PSPNet']['generator']['sizes'].split(',')])
        psp_size            = int(settings['models']['PSPNet']['generator']['psp_size'])
        deep_features_size  = int(settings['models']['PSPNet']['generator']['deep_feature_size'])
        final_stride        = int(settings['models']['PSPNet']['generator']['finalStride'])
        backend             = settings['models']['PSPNet']['generator']['backend']
        self.feats          = ResNetForClassificationModel(settings)
        self.n_pre          = int(settings['models']['PSPNet']['generator']['numberOfInputImageChannels'])
        self.use_deconv     = settings['models']['PSPNet']['generator']['useDeconv']

        self.pre = nn.Conv2d(self.n_pre, 3, kernel_size=1)

        self.psp = PSPModule(psp_size, 256, sizes)
        self.drop_0 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
        self.up_2 = PSPUpsample(256, 512, use_deconv=self.use_deconv, n=16)

        self.drop_1 = nn.Dropout2d(p=0.15)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1, stride=final_stride),
            #nn.LogSoftmax()
            nn.Tanh()
            #nn.Sigmoid()
        )


    def forward(self, x):
        if self.n_pre != 3:
            x = self.pre(x)
        f = self.feats(x) 
        #f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_0(p)

        p = self.up_1(p)
        p = self.drop_1(p)

        p = self.up_2(p)
        p = self.drop_2(p)


        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p) #, self.classifier(auxiliary)


##Downsized Version
#class PSPNetGeneratorModel(BaseModel):
#    def __init__(self, settings):
#        super(PSPNetGeneratorModel, self).__init__(settings)
#        n_classes           = int(settings['models']['PSPNet']['generator']['n_classes'])
#        sizes               = tuple([int(x) for x in settings['models']['PSPNet']['generator']['sizes'].split(',')])
#        psp_size            = int(settings['models']['PSPNet']['generator']['psp_size'])
#        deep_features_size  = int(settings['models']['PSPNet']['generator']['deep_feature_size'])
#        final_stride        = int(settings['models']['PSPNet']['generator']['finalStride'])
#        backend             = settings['models']['PSPNet']['generator']['backend']
#        self.feats          = res.ResNetForClassificationModel(settings)
#        self.n_pre          = int(settings['models']['PSPNet']['generator']['numberOfInputImageChannels'])
#        self.use_deconv     = settings['models']['PSPNet']['generator']['useDeconv']
#
#        self.pre = nn.Conv2d(self.n_pre, 3, kernel_size=1)
#
#        self.psp = PSPModule(psp_size, 256, sizes)
#        self.drop_0 = nn.Dropout2d(p=0.3)
#
#        self.up_1 = PSPUpsample(256, 56, n=4, use_deconv=self.use_deconv)
#        #self.up_2 = PSPUpsample(56, 56, use_deconv=self.use_deconv)
#        self.up_3 = PSPUpsample(56, 56, n=4, use_deconv=self.use_deconv)
#        #self.up_4 = PSPUpsample(56, 56, use_deconv=self.use_deconv)
#        self.up_5 = PSPUpsample(56, 512, n=2,use_deconv=self.use_deconv)
#        #self.up_1 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
#        #self.up_2 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
#        #self.up_3 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
#        #self.up_4 = PSPUpsample(256, 256, use_deconv=self.use_deconv)
#        #self.up_5 = PSPUpsample(256, 512, use_deconv=self.use_deconv)
#
#        self.drop_1 = nn.Dropout2d(p=0.15)
#        #self.drop_2 = nn.Dropout2d(p=0.15)
#        self.drop_3 = nn.Dropout2d(p=0.15)
#        #self.drop_4 = nn.Dropout2d(p=0.15)
#        self.drop_5 = nn.Dropout2d(p=0.15)
#        self.final = nn.Sequential(
#            nn.Conv2d(512, n_classes, kernel_size=1, stride=final_stride),
#            #nn.LogSoftmax()
#            nn.Tanh()
#            #nn.Sigmoid()
#        )
#
#        #self.classifier = nn.Sequential(
#        #    nn.Linear(deep_features_size, 256),
#        #    nn.ReLU(),
#        #    nn.Linear(256, n_classes)
#        #)
#
#    def forward(self, x):
#        if self.n_pre != 3:
#            x = self.pre(x)
#        f = self.feats(x) 
#        #f, class_f = self.feats(x) 
#        p = self.psp(f)
#        p = self.drop_0(p)
#
#        p = self.up_1(p)
#        p = self.drop_1(p)
#
#        #p = self.up_2(p)
#        #p = self.drop_2(p)
#
#        p = self.up_3(p)
#        p = self.drop_3(p)
#
#        #p = self.up_4(p)
#        #p = self.drop_4(p)
#
#        p = self.up_5(p)
#        p = self.drop_5(p)
#
#        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
#
#        return self.final(p) #, self.classifier(auxiliary)
##    def forward(self, x):
##        if self.n_pre != 3:
##            x = self.pre(x)
##        f = self.feats(x) 
##        #f, class_f = self.feats(x) 
##        p = self.psp(f)
##        p = self.drop_0(p)
##
##        p = self.up_1(p)
##        p = self.drop_1(p)
##
##        p = self.up_2(p)
##        p = self.drop_2(p)
##
##        p = self.up_3(p)
##        p = self.drop_3(p)
##
##        p = self.up_4(p)
##        p = self.drop_4(p)
##
##        p = self.up_5(p)
##        p = self.drop_5(p)
##
##        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
##
##        return self.final(p) #, self.classifier(auxiliary)
