#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import importlib

import torch
import torch.nn as nn

from aimaker.utils import SettingHandler, fcnt
from aimaker.models import ModelFactory
import aimaker.loss.loss_factory as lf# import LossFacotry
from aimaker.optimizers import OptimizerFactory


class BaseController:
    def __init__(self, settings=None, **kwargs):

        if settings is not None:
            self.is_data_parallel = settings['base']['isDataParallel']
            self.is_showmode_info = settings['ui']['base']['isShowModelInfo']
        else:
            self.is_data_parallel = False
            self.is_showmode_info = False

        self.settings = settings
        ch = SettingHandler(settings)
        self.gpu_ids = ch.getGPUID()
        self.checkpoints_dir = ch.getCheckPointsDir()

        self.loss_factory = lf.LossFactory(settings)
        self.optimizer_factory = OptimizerFactory(settings)
        self.show_model_lst = []

        self.model_factory = ModelFactory(settings)

    def _data_paralel_if_use(self, input, model):
        if self.is_data_parallel:
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def setInput(self, inputs, is_volatile=False):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def getModel(self):
        pass 

    def setMode(self, mode):
        if mode == 'train':
            pass
        elif mode == 'test' or mode == 'valid':
            pass

    def _setPredictModel(self, model):
        self.pred_model = model

    def _showModel(self):
        if self.is_showmode_info:
            print('---------- Networks initialized ----------')
            for model in self.show_model_lst:
                print(str(model.__class__).split("'")[1])
                self.printNetwork(model)
            print('------------------------------------------')

    def _saveModel(self, model, file_name, is_fcnt=True):
        model_name = str(model.__class__).split("'")[1]
        if model_name.split(".")[-1] == "BaseSequential":
            self._saveModel(list(model)[0], file_name)
        else:
            if is_fcnt:
                path = fcnt(self.checkpoints_dir, file_name, "pth")
            else:
                path = file_name
            state_dict = model.state_dict()
            state_dict['model_name'] = model_name
            state_dict['settings'] = self.settings
            torch.save(state_dict, path)
            print("{} is saved".format(path))

    def saveModels(self):
        pass

    def _loadModel(self, path, gpu_id):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("{} could not be found. ".format(path))
            state_dict = torch.load(path)
            m_lst = state_dict['model_name'].split(".")
            state_dict.pop('model_name')
            model_class = eval('importlib.import_module(".".join(m_lst[:-1])).{}'.format(m_lst[-1]))
            model = model_class(state_dict['settings']).to(gpu_id)
            state_dict.pop('settings')
            model.load_state_dict(state_dict)

        except:
            import traceback
            traceback.print_exc()
            model = torch.load(path, map_location=torch.device(gpu_id))
        print("{} has been loaded successfully.".format(path))
        return model

    def loadModels(self,):
        return True

    def printNetwork(self, model):
        import pprint as pp
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        pp.pprint(model)
        print('Total number of parameters: %d' % num_params)

    def getLosses(self):
        return {}

    def getImages(self):
        return {}

    def predict(self, tensor, isreturn=False):
        with torch.no_grad():
            return self.pred_model(tensor)#.data.cpu()
