import torch.nn as nn

from aimaker.layers.initialize_factory import InitializeFactory


class PadFactory:
    def __init__(self, settings):
        self.pad_dic = {'reflection': nn.ReflectionPad2d
                         }
        self.settings = settings

    def create(self, name):
        if not name in self.pad_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                       .format(name, self.__class__.__name__, self.pad_dic.keys()))

        return self.pad_dic[name]
