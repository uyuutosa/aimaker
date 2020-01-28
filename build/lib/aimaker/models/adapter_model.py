from aimaker.models.base_model import BaseModel
import torch.nn as nn

class AdapterModel(BaseModel):
    def __init__(self, config, pre_model, post_model):
        from models.activation_factory import ActivationFactory
        super(AdapterModel, self).__init__()
        pre_model = list(pre_model.children()[-1])
        post_model = list(post_model.children()[0])
        if isinstance(pre_model, nn.Linear):
            self.adapter = nn.Linear(pre_model.out_feature, post_model.in_feature)
        elif isinstance(pre_model, nn.Conv2d):
            self.adapter = nn.Conv2d(pre_model.out_chennels, 
                                     post_model.in_feature, 
                                     int(config['adapter settings']['stride']),
                                     int(config['adapter settings']['padding']))

        self.activation = ActivationFactory(config)\
                              .create(config['adapter settings']['activation'])
        
    def forward(self, x):
        x = self.adapter(x)
        self._setFeatureForView("adapter", x)
        x = self.activation()
        self._setFeatureForView("activation", x)
        return x


class LinearModel(BaseModel):
    def __init__(self, config, pre_model, n_out):
        from models.activation_factory import ActivationFactory
        super(AdapterModel, self).__init__()
        pre_model = list(pre_model.children()[-1])
        
        self.adapter = nn.Linear(pre_model.out_feature, n_out)

        self.activation = ActivationFactory(config)\
                              .create(config['adapter settings']['activation'])
        
    def forward(self, x):
        x = self.adapter(x)
        self._setFeatureForView("adapter", x)
        x = self.activation(x)
        self._setFeatureForView("activation", x)
        return x
