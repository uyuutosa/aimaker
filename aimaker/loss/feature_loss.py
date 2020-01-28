import aimaker.utils.util as util
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models
from collections import OrderedDict
import aimaker.loss.base_loss as bs

class TFLoss(bs.BaseLoss):
    def __init__(self, settings):
        from aimaker.loss.loss_factory    import LossFactory
        super(TFLoss, self).__init__(settings)
        loss_factory = LossFactory(settings)
        self.criterion = loss_factory.create('MSE')

    def __call__(self, pointing_map, feature_map_GT):
        arr = torch.cat([pointing_map.max(2)[0].argmax(2)[:, None, ...], pointing_map.max(3)[0].argmax(2)[:, None, ...]], dim=1).transpose(1,2)

        feature_map = (arr[:,:,None,:] >=  arr[:, None, ...]).view(arr.shape[0], arr.shape[1], -1)[:,None,...].float()
        feature_map = torch.where(feature_map>0, torch.ones_like(feature_map), torch.ones_like(feature_map)*-1)
        return self.criterion(feature_map, feature_map_GT), feature_map

        


class FeatureLoss(bs.BaseLoss):
    def __init__(self, settings):
        from aimaker.loss.loss_factory    import LossFactory
        from aimaker.models.model_factory import ModelFactory
        super(FeatureLoss, self).__init__(settings)

        self.gpu_ids = self.ch.getGPUID()

        loss_factory = LossFactory(settings)
        self.content_criterion = loss_factory.create(settings['feature loss settings']\
                                                           ['contentCriterion'])
        self.style_criterion   = loss_factory.create(settings['feature loss settings']\
                                                           ['styleCriterion'])

        self.alpha = float(settings['feature loss settings']['alpha'])
        self.beta  = float(settings['feature loss settings']['beta'])

        feature_dic = {'vgg19':self._constructVgg19}
        
        feature_dic[settings['feature loss settings']['featureNet']]()
        
        self.content_layers = settings['feature loss settings']\
                                    ['contentLossLayers'].split(',')
        self.style_layers   = settings['feature loss settings']\
                                    ['styleLossLayers'].split(',')

    def _constructVgg19(self):
        model = models.vgg19(pretrained=True)
        gpu_ids =  self.ch.getGPUID()
        if len(gpu_ids) and gpu_ids[0] != -1:
            model = model.cuda(gpu_ids[0])
        layers = model.features
        self.layer_dic = OrderedDict()

       
        conv_cnt = 0
        cluster_cnt = 1
        content_losses = []
        style_losses = []
        for i, layer in enumerate(layers):
            layer.requires_grad = False

            if isinstance(layer, nn.Conv2d):
                layer_name = 'conv'
                conv_cnt += 1

            elif isinstance(layer, nn.ReLU):
                layer_name = 'relu'
            elif isinstance(layer, nn.MaxPool2d):
                layer_name = 'pool'
                cluster_cnt += 1
                conv_cnt = 0
            name = "{}_{}_{}".format(layer_name, cluster_cnt, conv_cnt)
            self.layer_dic[name] = layer
            
            
    def __call__(self, input, content_target, style_target=None):


        is_style = True
        if style_target is None:
            style_target = content_target.clone()
            

#        is_first_time_for_content = True
        content_loss_lst = []
        style_loss_lst = []
        for name, layer in self.layer_dic.items():

            input          = layer(input)
            content_target = layer(content_target)
            style_target   = layer(style_target)


            if name in self.content_layers:
                content_loss_lst += [ContentLoss(input, content_target, self.content_criterion)]
                #content_loss_lst += [ContentLoss(input / torch.Tensor([input.shape[1:]]).prod(), content_target / torch.Tensor([content_target.shape[1:]]).prod(), self.content_criterion)]
                
            if name in self.style_layers:
                style_loss_lst += [StyleLoss(input, style_target, self.style_criterion)]

        self.dummy_content_model = nn.Sequential(*content_loss_lst)
        self.dummy_style_model   = nn.Sequential(*style_loss_lst)
        #return self.beta * dummy_style_model(input)
        return self.alpha * self.dummy_content_model(0) + self.beta * self.dummy_style_model(0)
    def backward(self, retain_grah=True):
        self.dummy_content_model.backward(retain_graph=retain_graph)
        self.dummy_style_model.backward(retain_graph=retain_graph)
        

class GramMatrix(bs.BaseLoss):
    def forward(self, input):
        a, b, c, d = input.size()

        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)
            
class ContentLoss(bs.BaseLoss):
    def __init__(self, input, target, criterion):
        super(ContentLoss, self).__init__()
        self.criterion = criterion
        self.input = input
        self.target = target

    def forward(self, x):
        self.loss = self.criterion(self.input, self.target.detach())  + x
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class StyleLoss(bs.BaseLoss):
    def __init__(self, input, target, criterion):
        super(StyleLoss, self).__init__()
        self.criterion = criterion
        self.input = input
        self.target = target
        a,b,c,d = input.shape
        self.gram = GramMatrix()

    def forward(self, x):
        self.loss = self.criterion(self.gram(self.input), 
                                   self.gram(self.target).detach()) + x
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

