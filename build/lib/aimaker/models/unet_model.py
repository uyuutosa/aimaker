import torch.nn as nn
import torch

from aimaker.layers import NormalizeFactory
from aimaker.layers import ActivationFactory
from aimaker.models import BaseModel
from aimaker.utils import SettingHandler

class UnetBlock(nn.Module):
    def __init__(self, i, settings, medium_layers=[]):
        super(UnetBlock, self).__init__()
        self.ch = SettingHandler(settings)
        self.n_input           = int(settings['models']['unet']['generator']['numberOfInputImageChannels'])
        self.n_output          = int(settings['models']['unet']['generator']['numberOfOutputImageChannels'])
        self.n_conv            = int(settings['models']['unet']['generator']['numComvolutionEachHierarchy'])
        num_hierarchy          = int(settings['models']['unet']['generator']['numHierarchy'])
        self.scale_ratio       = int(settings['models']['unet']['generator']['scaleRatio'])
        self.feature_size      = int(settings['models']['unet']['generator']['featureSize'])
        self.input_nc          = int(2 ** (num_hierarchy - i - 2)  * self.feature_size * self.scale_ratio)
#        use_bias               = settings['models']['unet']['generator']['useBias']
        self.norm_layer        = NormalizeFactory(settings).create(settings['models']['unet']['generator']['normalizeLayer'])
        self.inner_activation  = ActivationFactory(settings).create(
                                     settings['models']['unet']['generator']['innerActivation'])
        self.output_activation = ActivationFactory(settings).create(
                                     settings['models']['unet']['generator']['outputActivation'])

        self.is_outermost = self.is_innermost = False
        if i == (num_hierarchy - 1):
            self.is_outermost = True
        elif i == 0:
            self.is_innermost = True

        if self.is_innermost:
            self.model = nn.Sequential(*self._genMediumSideLayers())
        elif self.is_outermost:
            input_outermost_layers, output_outermost_layers = self._genOutermostLayers()
            self.model = nn.Sequential(*(input_outermost_layers +
                                         medium_layers +
                                         output_outermost_layers))
        else:
            input_layers = self._genInputSideLayers()
            output_layers     = self._genOutputSideLayers()
            self.input_model  = nn.Sequential(*input_layers)  
            self.input_medium_model = nn.Sequential(*(input_layers + medium_layers))
            self.output_model = nn.Sequential(*output_layers)
            
        #self.model = nn.Sequential(*layers)

    def _genMediumSideLayers(self):
        medium_layers = [nn.Conv2d(self.input_nc // (self.scale_ratio), 
                                    self.input_nc ,
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1), 
                         self.norm_layer(self.input_nc),
                         self.inner_activation] 

        for n in range(self.n_conv - 1):
            medium_layers += [nn.Conv2d(self.input_nc, 
                                        self.input_nc,
                                        kernel_size=3, 
                                        stride=1, 
                                        padding=1),
                              self.norm_layer(self.input_nc),
                              self.inner_activation] 

        medium_layers += [nn.Conv2d(self.input_nc, 
                                    self.input_nc // (self.scale_ratio),
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1),
                          self.norm_layer(self.input_nc // (self.scale_ratio)),
                          self.inner_activation] 

        return medium_layers

    def _genInputSideLayers(self):
        input_layers = [nn.Conv2d(self.input_nc // (self.scale_ratio), 
                                  self.input_nc, 
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1),
                        self.norm_layer(self.input_nc),
                        self.inner_activation] 

        for n in range(1, self.n_conv):
            input_layers += [nn.Conv2d(self.input_nc, 
                                       self.input_nc,
                                       kernel_size=3, 
                                       stride=1,
                                       padding=1),
                             self.norm_layer(self.input_nc),
                             self.inner_activation] 

        return input_layers

    def _genOutputSideLayers(self):

        output_layers = [nn.ConvTranspose2d(self.input_nc * (self.scale_ratio),
                                            self.input_nc,
                                            kernel_size=3, 
                                            stride=1, 
                                            padding=1),
                         self.norm_layer(self.input_nc),
                         self.inner_activation] 

        for n in range(1, self.n_conv - 1):
            output_layers += [nn.ConvTranspose2d(self.input_nc, 
                                      self.input_nc,
                                      kernel_size=3, 
                                      stride=1, 
                                      padding=1),
                              self.norm_layer(self.input_nc),
                              self.inner_activation] 

        output_layers += [nn.ConvTranspose2d(self.input_nc,
                                          self.input_nc // (self.scale_ratio),
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1),
                          self.norm_layer(self.input_nc // (self.scale_ratio)),
                          self.inner_activation] 
        return output_layers

    def _genOutermostLayers(self):

        n_channel    = self.n_input
        #n_channel    = self.ch.getNumberOfInputImageChannels()
        input_layers = [nn.ConvTranspose2d(n_channel, 
                                            self.input_nc,
                                            kernel_size=3, 
                                            stride=1, 
                                            padding=1),
                        self.norm_layer(self.input_nc),
                        self.inner_activation] 

        output_layers   = [nn.ConvTranspose2d(self.input_nc, 
                                              self.n_output,
                                              kernel_size=3, 
                                              stride=1, 
                                              padding=1),
                           self.output_activation] 

        return input_layers, output_layers

    def forward(self, x):
        if self.is_innermost or self.is_outermost:
            return  self.model(x)
        else:
            return self.output_model(
                       torch.cat([self.input_model(x), 
                                  self.input_medium_model(x)], 1))
        

class UnetForGeneratorModel(BaseModel):
    def __init__(self,
                 settings):

        super(UnetForGeneratorModel, self).__init__(settings)
        self.settings = settings
        ch = SettingHandler(settings)
        self.gpu_ids     = ch.getGPUID()
        #self.pad_factory = PadFactory(settings)

        self.model = UnetBlock(0, settings)
        for i in range(1, int(settings['models']['unet']['generator']['numHierarchy'])):
            self.model = UnetBlock(i, settings, [self.model])
        
            
            
    def forward(self, input):
        return self.model(input)

       # if self.settings['base']['isDataParallel']:
       #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
       # else:
       #     return self.model(input)
