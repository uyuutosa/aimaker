import random

import PIL.Image as I
import torch
import torch.nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
import torch.utils.data.dataloader as dataloader

import importlib
import data_source.data_source_factory as dsf
from   data_source.data_source import DataSourceFromCSV
from aimaker.utils import SettingHandler

class BaseDataset(data.Dataset):
    def __init__(self, setting=None):
        super(BaseDataset, self).__init__()

        self.setting = setting
        self.ch = SettingHandler(setting)
        self.mode = 'train'
        self.nom_transform    = self.ch.get_normalize_transform()
        self.to_tensor        = transforms.ToTensor()
        self.is_shuffle = self.setting['data']['base']['isShuffle'] if setting is not None else True

        #self.port = setting['utils']['connection']['port']
        self.train_test_ratio = float(setting['data']['base']['trainTestRatio']) if setting is not None else 1

        self.dsf = dsf.DataSourceFactory(train_test_ratio=self.train_test_ratio)
        self._setDataSource()

    def getDataLoader(self):
        return dataloader.DataLoader(self,
                                     batch_size=self.ch.get_batch_size(self.mode),
                                     shuffle=self.is_shuffle,
                                     )

    def getTransforms(self):
        self.input_transform = self.ch.get_input_transform(self.mode)
        self.target_transform = self.ch.get_target_transform(self.mode)
        self.common_transform = self.ch.get_common_transform(self.mode)

    def _setDataSource(self):
        self.ds = "dammy data Source"
        pass

    def _getInput(self, index):
        pass

    def _getTarget(self, index):
        pass

    def _inputTransform(self, input):
        if isinstance(input, list):
            input = torch.cat([self.input_transform(x) for x in input])
        else:
            input = self.input_transform(input)
        return input

    def _targetTransform(self, target):
        if isinstance(target, list):
            target = torch.cat([self.target_transform(x) for x in target])
        else:
            target = self.target_transform(target)
        return target

    def _commonTransform(self, input, target):
        seed = random.randint(0, 2147483647)

        if isinstance(input, list):
            lst = []
            for i in input:
                random.seed(seed)
                lst += [self.common_transform(i)]
            input = lst
        else:
            random.seed(seed)
            input = self.common_transform(input)

        if isinstance(target, list):
            lst = []
            for i in target:
                random.seed(seed)
                lst += [self.common_transform(i)]
            target = lst
        else:
            random.seed(seed)
            target = self.common_transform(target)

        return input, target

    def setMode(self, mode):
        if mode == 'valid':
            mode == 'train'
        self.ds.setMode(mode)

    def __getitem__(self, index):
        input   = self._getInput(index)
        target  = self._getTarget(index)

        if self.common_transform:
            input, target = self._commonTransform(input, target)

        if self.input_transform:
            input  = self._inputTransform(input)
        if self.target_transform:
            target = self._targetTransform(target)

        return input, target

    def __len__(self):
        return len(self.ds)
