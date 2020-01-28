from torch.optim import Adam
from aimaker.utils import BaseFactory


class AdamOptimizer:
    def __init__(self, settings):
        self.settings = settings
                 
    def __call__(self, params, settings):
        return Adam(
            params, lr=float(settings['optimizer']['base']['lr']),
            betas=tuple(float(x) for x in settings['optimizer']['base']['betas'].split(",")),
            eps=float(settings['optimizer']['base']['eps']),
            weight_decay=float(settings['optimizer']['base']['weightDecay']),
        )


class OptimizerFactory(BaseFactory):
    def __init__(self, settings):
        super().__init__(settings)
        self.adam = AdamOptimizer

    def _create(self, name):
        return eval(self.suffix+f"{name}(settings=self.settings)")
