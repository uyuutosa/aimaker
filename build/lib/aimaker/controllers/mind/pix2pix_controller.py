#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import aimaker.utils.util as util
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import cv2


def convertDevice(device):
    if ':' in device:
        device, n = device.split(':')
        if device == 'cuda':
            return int(n)


class Pix2PixForMDController:
    def __init__(self, setting):
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.setting         = setting
        ch                   = util.SettingHandler(setting)
        self.gpu_ids         = ch.get_GPU_ID()
        self.sheckpoints_dir = ch.get_check_points_dir()
        self.pool = util.ImagePool(int(ch.setting['controllers']['pix2pix']['imagePoolSize']))

#        model_factory        = mf.ModelFactory(setting)
        loss_factory         = lf.LossFactory(setting)
        optimizer_factory    = of.OptimizerFactory(setting)

        self.is_feature      = setting['controllers']['pix2pix']['isFeature']

        self.loadModels()

        self.generator_criterion  = loss_factory\
                                        .create(setting['controllers']['pix2pix']['generatorCriterion']).to(self.gpu_ids[0])



        #if len(self.gpu_ids):
        #    self.generator = self.generator.to(self.gpu_ids[0])

        if len(self.gpu_ids):
            self.discriminator = self.discriminator.to(self.gpu_ids[0])

        self.discriminator_criterion = loss_factory.create(setting['controllers']['pix2pix']['discriminatorCriterion']).to(self.gpu_ids[0])
        if len(self.gpu_ids):
            self.generator_criterion     = self.generator_criterion.to(self.gpu_ids[0])
            self.discriminator_criterion = self.discriminator_criterion.to(self.gpu_ids[0])

        self.light_criterion = loss_factory.create("L1").to(self.gpu_ids[0])
        self.model_light_criterion = loss_factory.create("L1").to(self.gpu_ids[0])
        self.texture_criterion = loss_factory.create("L1").to(self.gpu_ids[0])
        self.info_criterion = loss_factory.create("MSE").to(self.gpu_ids[0])
         

        self.generator_optimizer     = optimizer_factory.create(\
                                           setting['controllers']['pix2pix']['generatorOptimizer'])\
                                           (self.generator.parameters(), setting)
        self.discriminator_optimizer = optimizer_factory.create(setting['controllers']['pix2pix']['discriminatorOptimizer'])\
                                               (self.discriminator.parameters(), setting)

        if setting['ui']['base']['isShowModelInfo']:
            self.showModel()

    def _data_paralel_if_use(self, input, model):
        if self.setting['base']['isDataParallel']:
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def setInput(self, inputs, is_volatile=False):
        self.real, self.target, self.info = inputs

        if len(self.gpu_ids):
            self.real   = self.real.to(self.gpu_ids[0])
            self.target = self.target.to(self.gpu_ids[0])
            self.info   = self.info.to(self.gpu_ids[0])

    def forward(self):
        self.fake = self._data_paralel_if_use(self.real, self.generator)
        if self.real.size() == self.fake.size():
            self.fake_AB = self.pool.query(torch.cat((self.real, self.fake), 1))
        else:
            self.fake_AB = self.pool.query(torch.cat((self.fake, self.fake), 1))
        self.generator_loss = self._calcGeneratorLoss(self.fake, self.fake_AB, self.target)

        if self.real.size() == self.fake.size():
            real_AB = torch.cat((self.real, self.target), 1)
        else:
            real_AB = torch.cat((self.target, self.target), 1)

        self.discriminator_loss = self._calcDiscriminatorLoss(real_AB, self.fake_AB, self.target)

        #if self.is_feature:
        #    self.tf_loss, self.tf_fake =  self.feature_criterion(self.fake, self.tf_map)
        #    self.discriminator_loss = self.discriminator_loss + 0.1 * self.tf_loss

    def backward(self):
        # generator
        self.generator_optimizer.zero_grad()
        self.generator_loss.backward()
        self.generator_optimizer.step()

        # discriminator
        self.discriminator_optimizer.zero_grad()
        self.discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def _calcGeneratorLoss(self, fake, fake_AB, target):
        _lambda = float(self.setting['controllers']['pix2pix']['lambda'])
        d_out, info = self._data_paralel_if_use(fake_AB, self.discriminator)
        self.generator_disc_out_loss = self.discriminator_criterion(d_out, True)
        self.texture_loss = self.texture_criterion(fake[:, :3], target[:, :3])
        self.light_loss = self.light_criterion(fake[:, 3], target[:, 3])
        self.model_light_loss = self.model_light_criterion(fake[:, 4], target[:, 4])
        total_loss = (self.texture_loss + self.model_light_loss + self.light_loss)  * _lambda
        total_loss = total_loss + self.generator_disc_out_loss
        return total_loss


    def _calcDiscriminatorLoss(self, real_AB, fake_AB, target):
        real_out, real_info = self._data_paralel_if_use(real_AB, self.discriminator)
        fake_out, fake_info = self._data_paralel_if_use(fake_AB.detach(), self.discriminator)
        real_loss = self.discriminator_criterion(real_out, True)
        fake_loss = self.discriminator_criterion(fake_out, False)
        real_info_loss = self.info_criterion(real_info, self.info)
        fake_info_loss = self.info_criterion(fake_info, self.info)
        self.discriminator_out_loss = (real_loss + fake_loss) * 0.5
        self.discriminator_info_loss = (real_info_loss + fake_info_loss) * 0.5
        return self.discriminator_out_loss + self.discriminator_info_loss

    def setMode(self, mode):
        if mode == 'train':
            self.generator     = self.generator.train(True)
            self.discriminator = self.discriminator.train(True)
        elif mode == 'test' or mode == 'valid':
            self.generator     = self.generator.eval()
            self.discriminator = self.discriminator.eval()

    def showModel(self):
        print('---------- Networks initialized ----------')
        self.printNetwork(self.generator)
        #if self.setting['data']['base']['isTrain']:
        self.printNetwork(self.discriminator)
        print('------------------------------------------')

    def _saveModel(self, model, file_name, is_fcnt=True):
        model_name = str(model.__class__).split("'")[1]
        if model_name.split(".")[-1] == "BaseSequential":
            self._saveModel(list(model)[0], file_name)
        else:
            if is_fcnt:
                path = util.fcnt(self.sheckpoints_dir, file_name, "pth")
            else:
                path = file_name
            state_dict = model.state_dict()
            state_dict['model_name'] = model_name
            state_dict['setting'] = self.setting
            torch.save(state_dict, path)
            print("{} is saved".format(path))

    def saveModels(self):
        self._saveModel(self.generator, "pix2pix_generator")
        self._saveModel(self.discriminator, "pix2pix_discriminator")
            

    def _loadModel(self, path, gpu_id):
        import importlib 
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("{} could not be found. ".format(path))
            state_dict = torch.load(path)
            m_lst = state_dict['model_name'].split(".")
            state_dict.pop('model_name')
            model_class = eval('importlib.import_module(".".join(m_lst[:-1])).{}'.format(m_lst[-1]))
            model = model_class(state_dict['setting']).to(gpu_id)
            state_dict.pop('setting')
            model.load_state_dict(state_dict)

        except:
            model = torch.load(path, map_location=torch.device(gpu_id))
        print("{} has been loaded successfully.".format(path))
        return model

    def loadModels(self,):
        import aimaker.models.model_factory as mf
        model_factory = mf.ModelFactory(self.setting)
        generator_name = self.setting['controllers']['pix2pix']['generatorModel']
        discriminator_name = self.setting['controllers']['pix2pix']['discriminatorModel']

        try:
            generator_path = util.fcnt_load(self.sheckpoints_dir, "pix2pix_generator", "pth")
            self.generator = self._loadModel(generator_path, self.gpu_ids[0])
        except:
            import traceback
            traceback.print_exc()
            self.generator = model_factory.create(generator_name).to(self.gpu_ids[0]) 
            
        try:
            discriminator_path = util.fcnt_load(self.sheckpoints_dir, "pix2pix_discriminator", "pth")
            self.discriminator = self._loadModel(discriminator_path, self.gpu_ids[0])
        except:
            import traceback
            traceback.print_exc()
            self.discriminator = model_factory.create(discriminator_name).to(self.gpu_ids[0]) 

        return True


    def printNetwork(self, net):
        import pprint as pp
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        pp.pprint(net)
        print('Total number of parameters: %d' % num_params)



    def getLosses(self):
        return {"generator"             : self.generator_loss.item(),
                "generator_texture"     : self.texture_loss.item(),
                "generator_light"       : self.light_loss.item(),
                "generator_model_light" : self.model_light_loss.item(),
                "generator_disc"        : self.generator_disc_out_loss.item(),
                "discriminator"         : self.discriminator_loss.item(),
                "discriminator_out"     : self.discriminator_out_loss.item(),
                "discriminator_info"    : self.discriminator_info_loss.item(),
                }

    def _putPoint(self, img, maps):
        maps   = maps.data.cpu().numpy()
        img    = img.data.cpu().numpy()
        x_max  = maps.max(axis=2).argmax(2)[0:1]
        y_max  = maps.max(axis=3).argmax(2)[0:1]
        points = np.concatenate([x_max,y_max]).T
        img = img[0].transpose(1,2,0).copy()
        for p in points:
            cv2.circle(img, tuple(p), 2, (1, 0, 0), -1)
        return torch.Tensor(img.transpose(2,0,1))

    def getImages(self):
        ret_dic = {"image"    : self.real[:,0:3],
                   "z"            : self.real[:,3:],
                   "texture_fake" : self.fake[:,0:3],
                   "light_fake" : self.fake[:,3:4],
                   "model_light_fake" : self.fake[:,4:],
                   "texture_target" : self.target[:,0:3],
                   "light_target" : self.target[:, 3:4],
                   "model_light_target" : self.target[:,4:],
                   }

        ret_dic.update(self.generator.getFeature())
        ret_dic.update(self.discriminator.getFeature())
        return ret_dic


    def predict(self, tensor, isreturn=False):
        if not isinstance(tensor, autograd.Variable):
            tensor = autograd.Variable(tensor, volatile=True)
        predicted_tensor = self.generator.eval()(tensor).data.cpu()
        return predicted_tensor

