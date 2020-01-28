#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
import PIL.Image as I
import torch
import torch.nn as nn

from aimaker.controllers import BaseController
import aimaker.utils as util


class OptNetController(BaseController):
    def __init__(self, settings=None, **kwargs):
        super().__init__(settings)

        if settings is not None:
            self.model_name = "optNet"
            self.criterion_name = "L1"
            self.optimizer_name = "adam"
        else:
            self.model_name = kwargs['model']
            self.criterion_name = kwargs['criterion']
            self.optimizer_name = kwargs['optimizer']

        self.loadModels()
        self.criterion = self.loss_factory.create(self.criterion_name).to(self.gpu_ids[0])
        self.optimizer = self.optimizer_factory.create(self.optimizer_name)(self.model.parameters(), settings)
        self._showModel()

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def setInput(self, inputs, is_volatile=False):
        self.real_d1, self.target_d1 = inputs[0]
        self.real_d2, self.target_d2 = inputs[1]
        if len(self.gpu_ids):
            self.real_d1 = self.real_d1.to(self.gpu_ids[0])
            self.real_d2 = self.real_d2.to(self.gpu_ids[0])
            self.target_d1 = self.target_d1.to(self.gpu_ids[0])
            self.target_d2 = self.target_d2.to(self.gpu_ids[0])
        self.tex_target, self.depth_target, self.ml_target, self.ls_target = self.target_d1[:, 0:3], self.target_d1[:, 3:4], self.target_d1[:, 4:5], self.target_d1[:, 5:6]
        self.tex_target_d2, self.depth_target_d2, self.ml_target_d2, self.ls_target_d2 = self.target_d2[:, 0:3], self.target_d2[:, 3:4], self.target_d2[:, 4:5], self.target_d2[:, 5:6]

    def _data_paralel_if_use(self, input, model):
        if self.is_data_parallel:
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def losses(self, real, tex, diff, depth, ml, ls, target):
        tex_target, depth_target, ml_target, ls_target = target[:, 0:3], target[:, 3:4], target[:, 4:5], target[:, 5:6]
        diff_target = real - tex_target
        diff_loss_v = self.criterion(diff, diff_target)
        tex_loss_v = self.criterion(tex, tex_target)
        
        depth_loss_v = self.criterion(depth, depth_target)
        ml_loss_v = self.criterion(ml, ml_target)
        ls_loss_v = self.criterion(ls, ls_target)
        return tex_loss_v, depth_loss_v, ml_loss_v, ls_loss_v, diff_loss_v, diff_target

    def forward(self):
        self.tex, self.diff, self.depth, self.ml, self.ls, self.f = self._data_paralel_if_use((self.real_d1, self.real_d2), self.model)
        self.fake = self.tex + self.diff
        self.tex2, self.diff2, self.depth2, self.ml2, self.ls2, self.f2 = self._data_paralel_if_use((self.fake, self.real_d2), self.model)
        self.fake2 = self.tex2 + self.diff2
        self.tex_loss_v, self.depth_loss_v, self.ml_loss_v, self.ls_loss_v, self.diff_loss_v, self.diff_target = self.losses(self.real_d1, self.tex, self.diff, self.depth, self.ml, self.ls, self.target_d1)
        self.tex2_loss_v, self.depth2_loss_v, self.ml2_loss_v, self.ls2_loss_v, self.diff2_loss_v, self.diff2_target = self.losses(self.real_d1, self.tex2,  self.diff2, self.depth2, self.ml2, self.ls2, self.target_d1)

        self.tex_d2, self.diff_d2, self.depth_d2, self.ml_d2, self.ls_d2, self.f_d2 = self._data_paralel_if_use((self.real_d2, self.real_d2), self.model)
        self.fake_d2 = self.tex_d2 + self.diff_d2
        self.tex_d2_loss_v, self.depth_d2_loss_v, self.ml_d2_loss_v, self.ls_d2_loss_v, self.diff_d2_loss_v, self.diff_d2_target = self.losses(self.real_d2, self.tex_d2, self.diff_d2, self.depth_d2, self.ml_d2, self.ls_d2, self.target_d2)
        self.cycle_consistency_loss_v = self.criterion(self.real_d2, self.diff_d2 + self.tex_d2)

        self.feet_loss_v = (self.criterion(self.f, self.f_d2) + self.criterion(self.f2, self.f_d2)) / 2

    def backward(self):
        self.optimizer.zero_grad()

        self.tex_loss_v.backward(retain_graph=True)
        self.depth_loss_v.backward(retain_graph=True)
        self.ml_loss_v.backward(retain_graph=True)
        self.ls_loss_v.backward(retain_graph=True)
        #self.diff_loss_v.backward(retain_graph=True)

        self.tex2_loss_v.backward(retain_graph=True)
        self.depth2_loss_v.backward(retain_graph=True)
        self.ml2_loss_v.backward(retain_graph=True)
        self.ls2_loss_v.backward(retain_graph=True)
        #self.diff2_loss_v.backward(retain_graph=True)

        self.tex_d2_loss_v.backward(retain_graph=True)
        self.depth_d2_loss_v.backward(retain_graph=True)
        self.ml_d2_loss_v.backward(retain_graph=True)
        self.ls_d2_loss_v.backward(retain_graph=True)
        self.diff_d2_loss_v.backward(retain_graph=True)

        self.feet_loss_v.backward(retain_graph=True)
        self.cycle_consistency_loss_v.backward(retain_graph=True)

        self.optimizer.step()

    def getModel(self):
        return self.model

    def setMode(self, mode):
        if mode == 'train':
            self.model = self.model.train(True)
        elif mode == 'test' or mode == 'valid':
            pass
            #self.model = self.model.eval()

    def saveModels(self):
        self._saveModel(self.model, "simple")

    def loadModels(self,):
        from aimaker.models import ModelFactory
        model_factory = ModelFactory(self.settings)

        try:
            model_path = util.fcnt_load(self.checkpoints_dir, "simple", "pth")
            self.model = self._loadModel(model_path, self.gpu_ids[0])
        except:
            import traceback
            traceback.print_exc()
            self.model = model_factory.create(self.model_name).to(self.gpu_ids[0]) 
        self._setPredictModel(self.model)
        self.show_model_lst += [self.model]
        return True

    def getLosses(self):
        return {"tex_loss": self.tex_loss_v.item(),
                "depth_loss": self.depth_loss_v.item(),
                "ml_loss": self.ml_loss_v.item(),
                "ls_loss": self.ls_loss_v.item(),
                "diff_loss": self.diff_loss_v.item(),
                "tex2_loss": self.tex2_loss_v.item(),
                "depth2_loss": self.depth2_loss_v.item(),
                "ml2_loss": self.ml2_loss_v.item(),
                "ls2_loss": self.ls2_loss_v.item(),
                "diff2_loss": self.diff2_loss_v.item(),
                "tex_d2_loss": self.tex_d2_loss_v.item(),
                "depth_d2_loss": self.depth_d2_loss_v.item(),
                "ml_d2_loss": self.ml_d2_loss_v.item(),
                "ls_d2_loss": self.ls_d2_loss_v.item(),
                "diff_d2_loss": self.diff_d2_loss_v.item(),
                "feet_loss": self.feet_loss_v.item(),
                "cycle_consistency_loss": self.cycle_consistency_loss_v.item()
                }

    def getImages(self):
        ret_dic = {"real_d1": self.real_d1,
                   "real_d2": self.real_d2,
                   "tex": self.tex,
                   "tex2": self.tex2,
                   "tex_d2": self.tex_d2,
                   "tex_target": self.tex_target,
                   "tex_target_d2": self.tex_target_d2,
                   "depth": self.depth,
                   "depth2": self.depth2,
                   "depth_d2": self.depth_d2,
                   "depth_target": self.depth_target,
                   "depth_target_d2": self.depth_target_d2,
                   "ml": self.ml,
                   "ml2": self.ml2,
                   "ml_d2": self.ml_d2,
                   "ml_target": self.ml_target,
                   "ml_target_d2": self.ml_target_d2,
                   "ls": self.ls,
                   "ls2": self.ls2,
                   "ls_d2": self.ls_d2,
                   "ls_target": self.ls_target,
                   "ls_target_d2": self.ls_target_d2,
                   "stylized_image_d1": self.fake,
                   "stylized_image_2": self.fake2,
                   "stylized_image_d2": self.fake_d2,
                   "diff": self.diff,
                   "diff2": self.diff2,
                   "diff_d2": self.diff_d2,
                   "diff_target": self.diff_target,
                   "diff2_target": self.diff2_target,
                   "diff_d2_target": self.diff_d2_target,
                   }
        ret_dic.update(self.model.getFeature())
        return ret_dic


class OptNetController2(BaseController):
    def __init__(self, settings=None, **kwargs):
        super().__init__(settings)

        if settings is not None:
            self.model_name = "OpticalNet2"
            self.criterion_name = "L1"
            self.optimizer_name = "adam"
        else:
            self.model_name = kwargs['model']
            self.criterion_name = kwargs['criterion']
            self.optimizer_name = kwargs['optimizer']

        self.alpha = 0.01

        self.loadModels()
        self.criterion = self.loss_factory.create(self.criterion_name).to(self.gpu_ids[0])
        self.optimizer = self.optimizer_factory.create(self.optimizer_name)(self.model.parameters(), settings)
        self._showModel()

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def setInput(self, inputs, is_volatile=False):
        self.real, self.tex_tar = inputs
        self.diff_tar = self.real - self.tex_tar
        if len(self.gpu_ids):
            self.real = self.real.to(self.gpu_ids[0])
            self.diff_tar = self.diff_tar.to(self.gpu_ids[0])
            self.tex_tar = self.tex_tar.to(self.gpu_ids[0])

    def _data_paralel_if_use(self, input, model):
        if self.is_data_parallel:
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self):
        self.tex, self.diff = self._data_paralel_if_use(self.real, self.model)
        self.rec_loss = self.criterion(self.real, self.tex + self.diff)
        self.tex_loss = self.criterion(self.tex, self.tex_tar)
        self.diff_loss = self.criterion(self.diff, self.diff_tar)

        self.total_loss = self.rec_loss * self.alpha + self.tex_loss + self.diff_loss

        #self.feet_loss_v = (self.criterion(self.f, self.f_d2) + self.criterion(self.f2, self.f_d2)) / 2

    def backward(self):
        self.optimizer.zero_grad()

        self.total_loss.backward()

        self.optimizer.step()

    def getModel(self):
        return self.model

    def setMode(self, mode):
        if mode == 'train':
            self.model = self.model.train(True)
        elif mode == 'test' or mode == 'valid':
            pass
            #self.model = self.model.eval()

    def saveModels(self):
        self._saveModel(self.model, "optnet2")

    def loadModels(self,):
        from aimaker.models import ModelFactory
        model_factory = ModelFactory(self.settings)

        try:
            model_path = util.fcnt_load(self.checkpoints_dir, "optnet2", "pth")
            self.model = self._loadModel(model_path, self.gpu_ids[0])
        except:
            import traceback
            traceback.print_exc()
            self.model = model_factory.create(self.model_name).to(self.gpu_ids[0]) 
        self._setPredictModel(self.model)
        self.show_model_lst += [self.model]
        return True

    def getLosses(self):
        return {"tex_loss": self.tex_loss.item(),
                "diff_loss": self.diff_loss.item(),
                "rec_loss": self.rec_loss.item(),
                }

    def _deHSV(self, tensor):
        lst = []
        for t in tensor:
            lst += [torch.Tensor(array(I.fromarray(((t + 1) / 2 * 255).cpu().detach().numpy().astype(uint8).transpose(1,2,0), mode="HSV").convert("RGB")).transpose(2,0,1))[None]]
        arr = torch.cat(lst, 0)
        return arr

    def getImages(self):
        ret_dic = {"real": self._deHSV(self.real),
                   "fake": self._deHSV(self.diff + self.tex),
                   "tex": self._deHSV(self.tex), 
                   "tex_tar": self._deHSV(self.tex_tar),
                   "diff": self._deHSV(self.diff),
                   "diff_tar": self._deHSV(self.diff_tar)
                   }
        ret_dic.update(self.model.getFeature())
        return ret_dic
