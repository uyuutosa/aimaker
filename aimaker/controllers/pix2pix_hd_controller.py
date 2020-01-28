#!/usr/bin/env python
# -*- coding: utf-8 -*-
import aimaker.utils.util as util
import os
import itertools as it
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



class Pix2pixController:
    def __init__(self, setting):
        import aimaker.models.model_factory as mf
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.setting         = setting
        ch                   = util.SettingHandler(setting)
        self.gpu_ids         = ch.getGPUID()
        self.sheckpoints_dir = ch.getCheckPointsDir()
        self.pool = util.ImagePool(int(ch.setting['controllers']['pix2pix']['imagePoolSize']))

        model_factory        = mf.ModelFactory(setting)
        loss_factory         = lf.LossFactory(setting)
        optimizer_factory    = of.OptimizerFactory(setting)

        name                 = setting['controllers']['pix2pix']['generatorModel']

        if not self.loadModels():
            self.generator = model_factory.create(name) 
            if setting['data']['base']['isTrain']:
                name = setting['controllers']['pix2pix']['discriminatorModel']
                self.discriminator = model_factory.create(name) 

        self.generator_criterion  = loss_factory\
                                        .create(setting['controllers']['pix2pix']['generatorCriterion'])

        if len(self.gpu_ids):
            self.generator = self.generator.to(self.gpu_ids[0])

        if setting['data']['base']['isTrain']:
            if len(self.gpu_ids):
                self.discriminator = self.discriminator.to(self.gpu_ids[0])
                
            self.discriminator_criterion = loss_factory.create(setting['controllers']['pix2pix']['discriminatorCriterion'])
            if len(self.gpu_ids):
                self.generator_criterion     = self.generator_criterion.to(self.gpu_ids[0])   
                self.discriminator_criterion = self.discriminator_criterion.to(self.gpu_ids[0])   

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
        self.real, self.target  = inputs

        if len(self.gpu_ids):
            self.real   = self.real.to(self.gpu_ids[0])
            self.target = self.target.to(self.gpu_ids[0])

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
        #input(self.discriminator(fake_AB).shape)
        loss = self.discriminator_criterion(self._data_paralel_if_use(fake_AB, self.discriminator), True)
        self.objective_loss = self.generator_criterion(fake, target)
        return loss + self.objective_loss * _lambda 

    def _calcDiscriminatorLoss(self, real_AB, fake_AB, target):
        real_loss = self.discriminator_criterion(self._data_paralel_if_use(real_AB, self.discriminator), True)
        fake_loss = self.discriminator_criterion(self._data_paralel_if_use(fake_AB.detach(), self.discriminator), False)
        return (real_loss + fake_loss) * 0.5

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
        if self.setting['data']['base']['isTrain']:
            self.printNetwork(self.discriminator)
        print('------------------------------------------')

    def saveModels(self):
        name = util.fcnt(self.sheckpoints_dir, "pix2pix_generator", "pth")
        torch.save(self.generator, name)
        print("{} is saved".format(name))
        name = util.fcnt(self.sheckpoints_dir, "pix2pix_discriminator", "pth")
        torch.save(self.discriminator, name)
        print("{} is saved".format(name))
#        torch.save(self.discriminator, util.fcnt(self.sheckpoints_dir, "pix2pix_discriminator", "pth"))
        print("done save models")

    def loadModels(self,):
        try:
            generator_name = util.fcnt_load(self.sheckpoints_dir, "pix2pix_generator",     "pth")
            print(generator_name)
            self.generator     = torch.load(generator_name, map_location=lambda storage, loc: storage.cuda(convertDevice(self.gpu_ids[0])))
            self.generator.setSetting(self.setting)
        except:
            print("Checkpoint directory or files could not be found."+
                  "New directory {} will be created.".format(self.sheckpoints_dir))
            import traceback
            traceback.print_exc()
            return False

        if self.setting['data']['base']['isTrain']:
            try:
                discriminator_name = util.fcnt_load(self.sheckpoints_dir, "pix2pix_discriminator", "pth")
                print(discriminator_name)
                self.discriminator = torch.load(discriminator_name).to(self.gpu_ids[0])
                self.discriminator.setSetting(self.setting)
            except:
                print("Checkpoint directory or files could not be found."+
                      "New directory {} will be created.".format(self.sheckpoints_dir))
                import traceback
                traceback.print_exc()
                return False
        return True

    def printNetwork(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)



    def getLosses(self):
        return {"generator"      : self.generator_loss.data[0],
                "discriminator"  : self.discriminator_loss.data[0],
                "objective"     : self.objective_loss.data[0],
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
        b,c,h,w = self.real.shape
        pointed_real = self._putPoint(self.real, self.target)
        pointed_fake = self._putPoint(self.real, self.fake)

        arr = self.real.data[0][0:3]
        ret_dic = {"real"   : self.real.data[0][0:3],
                   #"fake_interp" : torch.where(self.fake.data[0] > -0.9, self.real.data[0], torch.ones_like(arr).to(self.gpu_ids[0]) * -1).float(),
                   #"target_interp" : torch.where(self.target.data[0] > -0.9, self.real.data[0], torch.ones_like(arr).to(self.gpu_ids[0]) * -1).float(),
                   #"fake"   : torch.cat([self.real.data[0][0:3], self.fake.data[0]]).max(0)[0],
                   "fake" : self.fake.data[0][0:3],
                   #"fake_interp" : torch.where(self.fake.data[0] > -0.9, self.real.data[0], torch.ones_like(arr).to(self.gpu_ids[0]) * -1).float(),
                   "pointed_real" : pointed_real,
                   "pointed_fake" : pointed_fake
                   }
        for i in range(0, self.target.shape[1], 3):
            ret_dic.update({"target_{}-{}".format(i,i+2) : self.target.data[0][i:i+3]})

        ret_dic.update(self.generator.getFeature())
        ret_dic.update(self.discriminator.getFeature())
        return ret_dic


    def predict(self, tensor, isreturn=False):
        if not isinstance(tensor, autograd.Variable):
            tensor = autograd.Variable(tensor, volatile=True)
        predicted_tensor = self.generator.eval()(tensor).data.cpu()
        return predicted_tensor

