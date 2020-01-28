#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import os

import torch 
import torch.nn as nn
import torch.autograd as autograd

import aimaker.utils.util as util


class DeepVoxelFlowController:
    def __init__(self, config):
        import aimaker.models.model_factory as mf
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.config              = config
        self.ch = ch             = util.ConfigHandler(config)
        self.gpu_ids             = ch.getGPUID()
        self.checkpoints_dir     = ch.getCheckPointsDir()
        self.pool = util.ImagePool(int(ch.config['controller']['voxcelFlow']['imagePoolSize']))

        self.lambda_1 = float(config['controller']['voxcelFlow']['lambda1'])
        self.lambda_2 = float(config['controller']['voxcelFlow']['lambda2'])

        model_factory             = mf.ModelFactory(config)
        loss_factory              = lf.LossFactory(config)
        optimizer_factory         = of.OptimizerFactory(config)

        name                      = config['controller']['voxcelFlow']['generatorModel']
        self.generator = model_factory.create(name) 
        self.generator_criterion  = loss_factory.create(\
                                       config['controller']['voxcelFlow']['generatorCriterion'])

        if len(self.gpu_ids):
            self.generator = self.generator.cuda(self.gpu_ids[0])

        if config['data']['isTrain']:
            name = config['controller']['voxcelFlow']['discriminatorModel']
            self.discriminator = model_factory.create(name) 
            if len(self.gpu_ids):
                self.discriminator = self.discriminator.cuda(self.gpu_ids[0])

        self.loadModels()

        if config['data']['isTrain']:
                
            self.discriminator_criterion = loss_factory.create(\
                                               config['controller']['voxcelFlow']\
                                                     ['discriminatorCriterion'])
            if len(self.gpu_ids):
                self.generator_criterion     = self.generator_criterion.cuda(self.gpu_ids[0])   
                self.discriminator_criterion = self.discriminator_criterion.cuda(self.gpu_ids[0])   

            self.generator_optimizer     = optimizer_factory.create(\
                                               config['controller']['voxcelFlow']['generatorOptimizer'])\
                                               (self.generator.parameters(), config)
            self.discriminator_optimizer = optimizer_factory.create(config['controller']['voxcelFlow']['discriminatorOptimizer'])\
                                               (self.discriminator.parameters(), config)

        if config['ui settings']['isShowModelInfo']:
            self.showModel()


    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def setMode(self, mode):
        if mode == 'train':
            self.generator     = self.generator.train(True)
            self.discriminator = self.discriminator.train(True)
        elif mode == 'test' or mode == 'valid':
            self.generator     = self.generator.eval()
            self.discriminator = self.discriminator.eval()

    def setInput(self, inputs, is_volatile=False):
        # images is pre and post frames
        # vector_map is (delta_x, delta_y) map on 2D pixels
        real, target  = inputs

        if len(self.gpu_ids):
            real = real.cuda(self.gpu_ids[0])
            target  = target.cuda(self.gpu_ids[0])

        self.real   = autograd.Variable(real, volatile=is_volatile)
        self.target = autograd.Variable(target, volatile=is_volatile)

    def _calcImageFromOpticalFlow(self, input, predicted_map):
        n_batch, channels, rows, cols = predicted_map.shape
        image_size = rows * cols
        x_mesh, y_mesh = util.meshgrid((0, cols), (0, rows))
        x_mesh = autograd.Variable(x_mesh.repeat(n_batch, 1, 1).cuda(self.gpu_ids[0]))
        y_mesh = autograd.Variable(y_mesh.repeat(n_batch, 1, 1).cuda(self.gpu_ids[0]))

        delta_x = (predicted_map[:,0,:,:].contiguous() + 1) * (cols  - 1) / 2
        delta_y = (predicted_map[:,1,:,:].contiguous() + 1) * (rows - 1) / 2
        delta_t = (predicted_map[:,2,:,:].contiguous() + 1) * 0.5

        image_0_r = input[:,0,:,:].contiguous()
        image_0_g = input[:,1,:,:].contiguous()
        image_0_b = input[:,2,:,:].contiguous()
        image_1_r = input[:,3,:,:].contiguous()
        image_1_g = input[:,4,:,:].contiguous()
        image_1_b = input[:,5,:,:].contiguous()


        L_0_x = x_mesh - delta_x
        L_1_x = x_mesh + delta_x

        L_0_y = y_mesh - delta_y
        L_1_y = y_mesh + delta_y

        image_0_r = torch.gather(image_0_r, 1, L_0_y.floor().long().clamp(0, rows-1))
        X_0_r     = torch.gather(image_0_r, 2, L_0_x.floor().long().clamp(0, cols-1))
        image_0_g = torch.gather(image_0_g, 1, L_0_y.floor().long().clamp(0, rows-1))
        X_0_g     = torch.gather(image_0_g, 2, L_0_x.floor().long().clamp(0, cols-1))
        image_0_b = torch.gather(image_0_b, 1, L_0_y.floor().long().clamp(0, rows-1))
        X_0_b     = torch.gather(image_0_b, 2, L_0_x.floor().long().clamp(0, cols-1))
        image_1_r = torch.gather(image_1_r, 1, L_1_y.floor().long().clamp(0, rows-1))
        X_1_r     = torch.gather(image_1_r, 2, L_1_x.floor().long().clamp(0, cols-1))
        image_1_g = torch.gather(image_1_g, 1, L_1_y.floor().long().clamp(0, rows-1))
        X_1_g     = torch.gather(image_1_g, 2, L_1_x.floor().long().clamp(0, cols-1))
        image_1_b = torch.gather(image_1_b, 1, L_1_y.floor().long().clamp(0, rows-1))
        X_1_b     = torch.gather(image_1_b, 2, L_1_x.floor().long().clamp(0, cols-1))


        L_0_x_delta = (L_0_x - L_0_x.floor())
        L_1_x_delta = (L_1_x - L_1_x.floor())

        L_0_y_delta = (L_0_y - L_0_y.floor())
        L_1_y_delta = (L_1_y - L_1_y.floor())


        I_r = self._calcImage(L_0_x_delta, L_0_y_delta, L_1_x_delta, L_1_y_delta, 
                              delta_t, X_0_r, X_1_r, n_batch, rows, cols)[:, None, :, :]
        I_g = self._calcImage(L_0_x_delta, L_0_y_delta, L_1_x_delta, L_1_y_delta, 
                              delta_t, X_0_g, X_1_g, n_batch,  rows, cols)[:, None, :, :]
        I_b = self._calcImage(L_0_x_delta, L_0_y_delta, L_1_x_delta, L_1_y_delta, 
                              delta_t, X_0_b, X_1_b, n_batch, rows, cols)[:, None, :, :]


        return torch.cat((I_r, I_g, I_b), 1) 

    def _calcImage(self, L_0_x_delta, L_0_y_delta, L_1_x_delta, L_1_y_delta, 
                   delta_z, X_0, X_1, n_batch, rows, cols):
        W_000 = (1 - L_0_x_delta)  * (1 - L_0_y_delta) * (1 - delta_z)
        W_100 = L_0_x_delta        * (1 - L_0_y_delta) * (1 - delta_z)
        W_010 = (1 - L_0_x_delta)  * L_0_y_delta       * (1 - delta_z)
        W_110 = L_0_x_delta        * L_0_y_delta       * (1 - delta_z)
        W_001 = (1 - L_1_x_delta)  * (1 - L_1_y_delta) * delta_z
        W_101 = L_1_x_delta        * (1 - L_1_y_delta) * delta_z
        W_011 = (1 - L_1_x_delta)  * L_1_y_delta       * delta_z
        W_111 = L_1_x_delta        * L_1_y_delta       * delta_z

        I =  (W_000 * X_0)        
        I[:, :  , :-1] = I[:, :  , :-1] + (W_100 * X_0)[:,  :, 1:]
        I[:, :-1, : ]  = I[:, :-1, :  ] + (W_010 * X_0)[:, 1:,  :]
        I[:, :-1, :-1] = I[:, :-1, :-1] + (W_110 * X_0)[:, 1:, 1:]
        I              = I              + (W_001 * X_1)
        I[:, :  , :-1] = I[:, :  , :-1] + (W_101 * X_1)[:,  :, 1:]
        I[:, :-1, : ]  = I[:, :-1, :  ] + (W_011 * X_1)[:, 1:,  :]
        I[:, :-1, :-1] = I[:, :-1, :-1] + (W_111 * X_1)[:, 1:, 1:]

        return I

    def forward(self):
        self.fake = self.generator(self.real)

        self.I = I = self._calcImageFromOpticalFlow(self.real, self.fake)

        self.generator_loss = self._calcGeneratorLoss(I, self.target)

        self.discriminator_loss = self._calcDiscriminatorLoss(I, self.target)

    def backward(self):
        # generator
        self.generator_optimizer.zero_grad()
        self.generator_loss.backward()
        self.generator_optimizer.step()

        # discriminator
        self.discriminator_optimizer.zero_grad()
        self.discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def _calcGeneratorLoss(self, fake, target):
        _lambda = float(self.config['controller']['voxcelFlow']['lambda'])
        discriminator_loss = self.discriminator_criterion(self.discriminator(fake), True)
        return discriminator_loss + self.generator_criterion(fake, target) * _lambda + \
               self._calcRegularization()

    def _calcRegularization(self):
        x_div = self.fake[:, 0,  :, 1:] - self.fake[:, 0, :,   :-1]
        y_div = self.fake[:, 1, 1:,  :] - self.fake[:, 1, :-1, :  ]
        loss_motion = self.lambda_1 * (x_div.abs().sum() + y_div.abs().sum())
        loss_mask   = self.lambda_2 * (self.fake[:,2] - self.fake[:, 2]).abs().sum()
        return loss_motion + loss_mask

    def _calcDiscriminatorLoss(self, fake, target):
        real_loss = self.discriminator_criterion(self.discriminator(target), True)
        fake_loss = self.discriminator_criterion(self.discriminator(fake.detach()), False)
        return (real_loss + fake_loss) * 0.5

    def showModel(self):
        print('---------- Networks initialized ----------')
        self.printNetwork(self.generator)
        if self.config['data']['isTrain']:
            self.printNetwork(self.discriminator)
        print('------------------------------------------')

    def saveModels(self):
        torch.save(self.generator, util.fcnt(self.checkpoints_dir, "voxel_flow_generator", "pth"))
        torch.save(self.discriminator, util.fcnt(self.checkpoints_dir, "voxel_flow_discriminator", "pth"))
        print("done save models")

    def loadModels(self,):
        try:
            generator_model     = util.fcnt_load(self.checkpoints_dir, "voxel_flow_generator",     "pth")
            discriminator_model = util.fcnt_load(self.checkpoints_dir, "voxel_flow_discriminator", "pth")
            self.generator      = torch.load(generator_model).cuda(self.gpu_ids[0])
            self.discriminator  = torch.load(discriminator_model).cuda(self.gpu_ids[0])
            self.generator.setConfig(self.config)
            self.discriminator.setConfig(self.config)
        except:
            print("Checkpoint directory or files could not be found."+
                  "New directory {} will be created.".format(self.checkpoints_dir))

    def printNetwork(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)



    def getLosses(self):
        return {"generator"      : self.generator_loss.data[0],
                "discriminator"  : self.discriminator_loss.data[0]}

    def getImages(self):
        fake = self.fake
        fake = (fake - fake.min()) / (fake.max() - fake.min())
        ret_dic = {"real"   : self.real.data[0][:3],
                   "fake"   : self.I.data[0],
                   "target" : self.target.data[0],
                   "flow"   : self.fake.data[0],
                   "diff"   : (self.target - self.I).data[0],
                   }
        ret_dic.update(self.generator.getFeature())
        ret_dic.update(self.discriminator.getFeature())
        return ret_dic


    def predict(self, tensor, isreturn=False):
        if not isinstance(tensor, autograd.Variable):
            tensor  = autograd.Variable(tensor,  volatile=True)
        predicted_map = self.generator(tensor)
        n_batch, channel, rows, cols = tensor.shape
        return self._calcImageFromOpticalFlow(tensor, predicted_map)

