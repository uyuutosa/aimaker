from torch.nn import ReLU, RReLU, Hardtanh, Tanh, ReLU6,\
                     Sigmoid, ELU, CELU, SELU, GLU, Hardshrink,\
                     LeakyReLU, LogSigmoid, Softplus, Softshrink
from aimaker.utils import BaseFactory

class ActivationFactory(BaseFactory):
    def __init__(self, settings):
        super().__init__(settings)

    def _create(self, name):
        return eval(self.suffix+f"{name}()")
