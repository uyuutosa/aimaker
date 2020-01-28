#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

import aimaker.utils as util
from aimaker.loss import LossFactory
from aimaker.optimizers import OptimizerFactory


def convertDevice(device):
    if ':' in device:
        device, n = device.split(':')
        if device == 'cuda':
            return int(n)

class Pix2pixController:
    def __init__(self, setting):

        self.setting         = setting
        ch                   = util.SettingHandler(setting)
        self.gpu_ids         = ch.getGPUID()
        self.sheckpoints_dir = ch.getCheckPointsDir()
        self.pool = util.ImagePool(int(ch.setting['controllers']['pix2pix']['imagePoolSize']))

#        model_factory        = mf.ModelFactory(setting)
        loss_factory         = LossFactory(setting)
        optimizer_factory    = OptimizerFactory(setting)

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

        self.feature_criterion = loss_factory.create('TF')
         

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
        if len(inputs) == 2:
            self.real, self.target = inputs
        else:
            self.real, self.target, self.feature = inputs

        if len(self.gpu_ids):
            self.real   = self.real.to(self.gpu_ids[0])
            self.target = self.target.to(self.gpu_ids[0])
            #if len(inputs) != 2:
            #    self.tf_map   = self.feature['tf_map'].to(self.gpu_ids[0])

        #self.real   = autograd.Variable(real,   volatile=is_volatile)
        #self.target = autograd.Variable(target, volatile=is_volatile)

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
        loss = self.discriminator_criterion(self._data_paralel_if_use(fake_AB, self.discriminator), True)
        self.objective_loss = self.generator_criterion(fake, target)
        total_loss = self.objective_loss * _lambda
        total_loss = total_loss + loss
        return total_loss


    def _calcDiscriminatorLoss(self, real_AB, fake_AB, target):
        real_loss = self.discriminator_criterion(self._data_paralel_if_use(real_AB, self.discriminator), True)
        fake_loss = self.discriminator_criterion(self._data_paralel_if_use(fake_AB.detach(), self.discriminator), False)
        loses = (real_loss + fake_loss) * 0.5
        return loses

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

        #try:
        #    print(generator_name)
        #    self.generator     = torch.load(generator_name, map_location=lambda storage, loc: storage.cuda(convertDevice(self.gpu_ids[0])))
        #    self.generator.setSetting(self.setting)
        #except:
        #    print("Checkpoint directory or files could not be found."+
        #          "New directory {} will be created.".format(self.sheckpoints_dir))
        #    import traceback
        #    traceback.print_exc()
        #    return False

        #if self.setting['data']['base']['isTrain']:
        #    try:
        #        discriminator_name = util.fcnt_load(self.sheckpoints_dir, "pix2pix_discriminator", "pth")
        #        print(discriminator_name)
        #        self.discriminator = torch.load(discriminator_name).to(self.gpu_ids[0])
        #        self.discriminator.setSetting(self.setting)
        #    except:
        #        print("Checkpoint directory or files could not be found."+
        #              "New directory {} will be created.".format(self.sheckpoints_dir))
        #        import traceback
        #        traceback.print_exc()
        #        return False
        #return True

    #def loadModels(self,):
    #    try:
    #        generator_name = util.fcnt_load(self.sheckpoints_dir, "pix2pix_generator",     "pth")
    #        print(generator_name)
    #        self.generator     = torch.load(generator_name, map_location=lambda storage, loc: storage.cuda(convertDevice(self.gpu_ids[0])))
    #        self.generator.setSetting(self.setting)
    #    except:
    #        print("Checkpoint directory or files could not be found."+
    #              "New directory {} will be created.".format(self.sheckpoints_dir))
    #        import traceback
    #        traceback.print_exc()
    #        return False

    #    if self.setting['data']['base']['isTrain']:
    #        try:
    #            discriminator_name = util.fcnt_load(self.sheckpoints_dir, "pix2pix_discriminator", "pth")
    #            print(discriminator_name)
    #            self.discriminator = torch.load(discriminator_name).to(self.gpu_ids[0])
    #            self.discriminator.setSetting(self.setting)
    #        except:
    #            print("Checkpoint directory or files could not be found."+
    #                  "New directory {} will be created.".format(self.sheckpoints_dir))
    #            import traceback
    #            traceback.print_exc()
    #            return False
    #    return True

    def printNetwork(self, net):
        import pprint as pp
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        pp.pprint(net)
        print('Total number of parameters: %d' % num_params)



    def getLosses(self):
        return {"generator"      : self.generator_loss.item(),
                "discriminator"  : self.discriminator_loss.item(),
                #"tf_loss"        : self.tf_loss.item(),
                "objective"      : self.objective_loss.item()}

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
        b,c,h,w = self.real.shape

        arr = self.real.data[0][0:3]
        ret_dic = {"real"   : self.real.data[0][0:3],
                   "pointedreal": self._putPoint(self.real.data, self.target.data),
                   "pointedfake": self._putPoint(self.real.data, self.fake.data),
                   #"fake_interp" : torch.where(self.fake.data[0] > -0.9, self.real.data[0], torch.ones_like(arr).to(self.gpu_ids[0]) * -1).float(),
                   #"target_interp" : torch.where(self.target.data[0] > -0.9, self.real.data[0], torch.ones_like(arr).to(self.gpu_ids[0]) * -1).float(),
                   "target" : self.target.data[0][0:3],
                   #"fake"   : torch.cat([self.real.data[0][0:3], self.fake.data[0]]).max(0)[0],
                   "fake" : self.fake.data[0][0:3],
                   #"tf_real" : self.tf_map,
                   #"tf_fake" : self.tf_fake,
                   }
        #for i in range(0, self.target.shape[1], 3):
        #    ret_dic.update({"target_{}-{}".format(i,i+2) : self.target.data[0][i:i+3]})

        ret_dic.update(self.generator.getFeature())
        ret_dic.update(self.discriminator.getFeature())
        return ret_dic


    def predict(self, tensor, isreturn=False):
        if not isinstance(tensor, autograd.Variable):
            tensor = autograd.Variable(tensor, volatile=True)
        predicted_tensor = self.generator.eval()(tensor).data.cpu()
        return predicted_tensor

