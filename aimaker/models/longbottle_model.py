import torch
import torch.nn as nn
import aimaker.utils.util as util

class Bottleneck(nn.Module):
    def __init__(self, config, num_layer):
        super(Bottleneck, self).__init__()
        

            
        self.bottleneck = nn.Sequential(torch.nn.ReflectionPad2d(1),    
                                        torch.nn.Conv2d(num_layer, num_layer//2, 3, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.ReflectionPad2d(1),
                                        torch.nn.Conv2d(num_layer//2, num_layer, 3, 1),
                                        torch.nn.ReLU())
                         
    def forward(self, x):
        return x + self.bottleneck(x)


class LongBottleModel(nn.Module):
    def __init__(self, config):
        super(LongBottleModel, self).__init__()

        num_input_feature = int(config['longBottle settings']['numInputFeature'])
        num_feature       = int(config['longBottle settings']['numFeature'])
        num_bottleneck    = int(config['longBottle settings']['numBottleneck'])
        self.gpu_ids           = util.ConfigHandler(config).getGPUID()

        
        num_downsampled_feature = num_feature * 2
        self.downsample =  nn.Sequential(torch.nn.ReflectionPad2d(3),
                      torch.nn.Conv2d(num_input_feature, num_feature, 7, 1),
                      torch.nn.ReLU(),
                      torch.nn.BatchNorm2d(num_feature),
                      torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(num_feature, num_feature*2,3,2),
                      torch.nn.ReLU(),              # div2
                      torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(num_feature*2, num_downsampled_feature,3,2),
                      torch.nn.ReLU())# div 2
        
        
        bottlenecks = []
        num_layer = num_downsampled_feature
        for i in range(num_bottleneck):
                        
            bottlenecks += [Bottleneck(config, num_layer)]
            
                
        self.bottlenecks = nn.Sequential(*bottlenecks)
        

        self.upsample =  nn.Sequential(
                      #torch.nn.ReflectionPad2d(1),
                      torch.nn.ConvTranspose2d(num_downsampled_feature, num_feature*2, 3,2,padding=1,output_padding=1),
                      torch.nn.ReLU(),              # mult 2
                      torch.nn.BatchNorm2d(num_feature*2),
                      #torch.nn.ReflectionPad2d(1),
                      torch.nn.ConvTranspose2d(num_feature*2, num_feature, 3,2,padding=1,output_padding=1),
                      torch.nn.ReflectionPad2d(3),          
                      torch.nn.Conv2d(num_feature, num_input_feature, 7, 1),  
                      torch.nn.Tanh())# div 2
        
    def forward(self, x):
        downsampled = self.downsample(x)
        downsampled = self.bottlenecks(downsampled) 
            
        return self.upsample(downsampled)# + x 
        

