import importlib

import aimaker.data.datasets as datasets
from aimaker.data.datasets.super_resolution import DatasetOfSuperRes
from aimaker.data.datasets.edge    import DatasetOfEdges
from aimaker.data.datasets.interpolation import DatasetOfInterpolation
from aimaker.data.datasets.pair import DatasetOfPair
from aimaker.data.datasets.starGAN import DatasetOfStarGAN
from aimaker.data.datasets.mind.mind_dataset import Dataset, Dataset2, Dataset4, MultiDataset

class DatasetFactory():
    def __init__(self, setting):
        self.data_dic = {'edge'    : DatasetOfEdges,
                         'interp'  : DatasetOfInterpolation,
                         'pair'    : DatasetOfPair,
                         'sr'      : DatasetOfSuperRes,
                         'stargan' : DatasetOfStarGAN,
                         'mind'    : Dataset, 
                         'mind2'    : Dataset2, 
                         'mind3'    : MultiDataset, 
                         'mind4'    : Dataset4, 
                         }
        self.setting = setting

    def _importIfAdditionalDatasetExists(self, name):
        module_name = self.setting['data']['base']['importModule']

        try:
            self.mod = importlib.import_module("{}".format(module_name))
            print('self.data_dic.update({"%s":self.mod.%s})' %(name, name))
            exec('self.data_dic.update({"%s":self.mod.%s})' %(name, name))
            return True
        except:
            import traceback
            traceback.print_exc()
            return False

    def create(self, name):
        if not name in self.data_dic:
            if not self._importIfAdditionalDatasetExists(name):
                raise NotImplementedError(('{} is wrong key word for ' + \
                                           '{}. choose {}')\
                                          .format(name, self.__class__.__name__, self.data_dic.keys()))

        return self.data_dic[name](self.setting)

