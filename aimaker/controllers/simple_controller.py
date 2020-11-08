#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from aimaker.controllers import BaseController
from aimaker.utils import fcnt_load


class SimpleController(BaseController):
    def __init__(self, settings=None, **kwargs):
        super().__init__(settings)

        if settings is not None:
            self.model_name = settings['controllers']['simple']['model']
            self.criterion_name = settings['controllers']['simple']['criterion']
            self.optimizer_name = settings['controllers']['simple']['optimizer']
        else:
            self.model_name = kwargs['model']
            self.criterion_name = kwargs['criterion']
            self.optimizer_name = kwargs['optimizer']

        self.load_models()
        self.criterion = self.loss_factory.create(self.criterion_name).to(self.gpu_ids[0])
        self.optimizer = self.optimizer_factory.create(self.optimizer_name)(self.model.parameters(), settings)
        self._show_model()

    def __call__(self, inputs, is_volatile=False):
        self.set_input(inputs, is_volatile=is_volatile)

    def set_input(self, inputs, is_volatile=False):
        self.real, self.target = inputs
        if len(self.gpu_ids):
            self.real = self.real.to(self.gpu_ids[0])
            self.target = self.target.to(self.gpu_ids[0])

    def forward(self):
        if self.mode != "train":
            with torch.no_grad():
                self.fake = self._data_paralel_if_use(self.real, self.model)
                self.loss = self.criterion(self.fake, self.target)
        else:
            self.fake = self._data_paralel_if_use(self.real, self.model)
            self.loss = self.criterion(self.fake, self.target)

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.model = self.model.train(True)
        elif mode == 'test' or mode == 'valid':
            self.model = self.model

    def get_model(self):
        return self.model

    def save_models(self):
        self._save_model(self.model, "simple")

    def load_models(self, ):
        import aimaker.models.model_factory as mf
        model_factory = mf.ModelFactory(self.settings)

        try:
            model_path = fcnt_load(self.checkpoints_dir, "simple", "pth")
            self.model = self._load_model(model_path, self.gpu_ids[0])
        except:
            import traceback
            traceback.print_exc()
            self.model = model_factory.create(self.model_name).to(self.gpu_ids[0]) 
        self._set_predict_model(self.model)
        self.show_model_lst += [self.model]
        return True

    def get_losses(self):
        return {"loss": self.loss.item()}

    def get_images(self):
        ret_dic = {"real": self.real[0][0:3],
                   "fake": self.fake[0][0:3],
                   "target": self.target[0][0:3],
                   }
        ret_dic.update(self.model.getFeature())
        return ret_dic
