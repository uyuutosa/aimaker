#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import itertools as it
import torch 
import torch.nn as nn
import torch.autograd as autograd

from aimaker.utils import SettingHandler

class CycleGANController:
    def __init__(self, settings):
        import aimaker.models.model_factory as mf
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.settings = settings
        ch = SettingHandler(settings)
        
        self.gpu_ids = ch.get_GPU_ID()
        self.checkpoints_dir = ch.get_check_points_dir()
        model_factory        = mf.ModelFactory(settings)
        loss_factory         = lf.LossFactory(settings)
        optimizer_factory    = of.OptimizerFactory(settings)

        # for discriminator regularization
        self.pool_fake_A = util.ImagePool(int(settings['controllers']['cycleGAN']['imagePoolSize']))
        self.pool_fake_B = util.ImagePool(int(settings['controllers']['cycleGAN']['imagePoolSize']))

        name = settings['controllers']['cycleGAN']['generatorModel']
        self.netG_A = model_factory.create(name) 
        self.netG_B = model_factory.create(name)
        if len(self.gpu_ids):
            self.netG_A = self.netG_A.cuda(self.gpu_ids[0])
            self.netG_B = self.netG_B.cuda(self.gpu_ids[0])

            name = settings['controllers']['cycleGAN']['discriminatorModel']
            self.netD_A = model_factory.create(name) 
            self.netD_B = model_factory.create(name) 
            if len(self.gpu_ids):
                self.netD_A = self.netD_A.cuda(self.gpu_ids[0])
                self.netD_B = self.netD_B.cuda(self.gpu_ids[0])
                
        self.loadModels()
            

        self.criterionGAN   = loss_factory.create("GANLoss")
        self.criterionCycle = loss_factory.create(settings['controllers']['cycleGAN']['cycleLoss'])
        self.criterionIdt   = loss_factory.create(settings['controllers']['cycleGAN']['idtLoss'])
        if len(self.gpu_ids):
            self.criterionGAN   = self.criterionGAN.cuda(self.gpu_ids[0])   
            self.criterionCycle = self.criterionCycle.cuda(self.gpu_ids[0]) 
            self.criterionIdt   = self.criterionIdt.cuda(self.gpu_ids[0])  

        # initialize optimizers
        self.optimizer_G    = optimizer_factory.create(settings['controllers']['cycleGAN']['generatorOptimizer'])(it.chain(self.netG_A.parameters(),self.netG_B.parameters()), settings)

        if settings['data']['base']['isTrain']:
            self.optimizer_D_A  = optimizer_factory.create(settings['controllers']['cycleGAN']['D_AOptimizer'])(self.netD_A.parameters(), settings)
            self.optimizer_D_B  = optimizer_factory.create(settings['controllers']['cycleGAN']['D_BOptimizer'])(self.netD_B.parameters(), settings)

        if settings['ui']['base']['isShowModelInfo']:
            self.showModel();

    def setMode(self, mode):
        if mode == 'train':
            self.netG_A = self.netG_A.train(True)
            self.netG_B = self.netG_B.train(True)
            self.netD_A = self.netD_A.train(True)
            self.netD_B = self.netD_B.train(True)
        elif mode == 'test' or mode == 'valid':
            self.netG_A = self.netG_A.eval(True)
            self.netG_B = self.netG_B.eval(True)
            self.netD_A = self.netD_A.eval(True)
            self.netD_B = self.netD_B.eval(True)

    def showModel(self):
        print('---------- Networks initialized ----------')
        self.printNetwork(self.netG_A)
        self.printNetwork(self.netG_B)
        if self.settings['data']['base']['isTrain']:
            self.printNetwork(self.netD_A)
            self.printNetwork(self.netD_B)
        print('------------------------------------------')

    def saveModels(self):
        torch.save(self.netG_A, util.fcnt(self.checkpoints_dir, "netG_A", "pth"))
        torch.save(self.netG_B, util.fcnt(self.checkpoints_dir, "netG_B", "pth"))
        torch.save(self.netD_A, util.fcnt(self.checkpoints_dir, "netD_A", "pth"))
        torch.save(self.netD_B, util.fcnt(self.checkpoints_dir, "netD_B", "pth"))
        print("done save models")

    def loadModels(self,):
        try:
            self.netG_A = torch.load(util.fcnt_load(self.checkpoints_dir, "netG_A", "pth")).cuda(self.gpu_ids[0])
            self.netG_B = torch.load(util.fcnt_load(self.checkpoints_dir, "netG_B", "pth")).cuda(self.gpu_ids[0])
            self.netD_A = torch.load(util.fcnt_load(self.checkpoints_dir, "netD_A", "pth")).cuda(self.gpu_ids[0])
            self.netD_B = torch.load(util.fcnt_load(self.checkpoints_dir, "netD_B", "pth")).cuda(self.gpu_ids[0])
        except:
            print("Checkpoint directory could not be found."+
                  "New directory {} is created.".format(self.checkpoints_dir))

    def printNetwork(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)



    def getLosses(self):
        
        return {"all"     : util.peelVariable(self.loss_G)[0],
                "DA"      : util.peelVariable(self.loss_D_A)[0],
                "DB"      : util.peelVariable(self.loss_D_B)[0],
                "cycleA"  : util.peelVariable(self.loss_cycle_A)[0],
                "cycleB"  : util.peelVariable(self.loss_cycle_B)[0],
                "cycleA2" : util.peelVariable(self.loss_cycle_A_2)[0],
                "cycleB2" : util.peelVariable(self.loss_cycle_B_2)[0],
                "idtA"    : util.peelVariable(self.loss_idt_A)[0],
                "idtB"    : util.peelVariable(self.loss_idt_B)[0]}


    def getImages(self):
        
        return {"realA"   : util.peelVariable(self.real_A)[0],
                "fakeB"   : util.peelVariable(self.fake_B)[0],
                "realB"   : util.peelVariable(self.real_B)[0],
                "fakeA"   : util.peelVariable(self.fake_A)[0],
                "cycleA"  : util.peelVariable(self.rec_A)[0],
                "cycleB"  : util.peelVariable(self.rec_B)[0],
                "cycleA2" : util.peelVariable(self.rec_A_2)[0],
                "cycleB2" : util.peelVariable(self.rec_B_2)[0],
                "idtA"    : util.peelVariable(self.idt_A)[0],
                "idtB"    : util.peelVariable(self.idt_B)[0]}

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile)

    def setInput(self, inputs, is_volatile=False):
        self.input_A = inputs[0]
        self.input_B = inputs[1]
        if len(self.gpu_ids):
            self.input_A = self.input_A.cuda(self.gpu_ids[0])
            self.input_B = self.input_B.cuda(self.gpu_ids[0])

        self.real_A = autograd.Variable(self.input_A, volatile=is_volatile)
        self.real_B = autograd.Variable(self.input_B, volatile=is_volatile)

    def forward(self):
        self.fake_A = self.netG_B(self.real_B)
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A  = self.netG_B(self.fake_B)
        self.rec_B  = self.netG_A(self.fake_A)
        self.idt_A  = self.netG_B(self.fake_A)
        self.idt_B  = self.netG_A(self.fake_B)
        if self.settings['controllers']['cycleGAN']['secondCycle']:
            self.rec_A_2 = self.netG_A(self.rec_A)
            self.rec_B_2 = self.netG_B(self.rec_B)
        else:
            self.rec_A_2    = [None]
            self.rec_B_2    = [None]
            self.loss_cycle_A_2 = [None]
            self.loss_cycle_B_2 = [None]

        self._forwardGenerator()
        self._forwardDiscriminator_A()
        self._forwardDiscriminator_B()
            
    def _forwardDiscriminator_A(self):
        fake_B        = self.pool_fake_B.query(self.fake_B)
        self.loss_D_A = self._backwardDiscriminator(self.netD_A, self.real_B, fake_B)

    def _forwardDiscriminator_B(self):
        fake_A        = self.pool_fake_A.query(self.fake_A)
        self.loss_D_B = self._backwardDiscriminator(self.netD_B, self.real_A, fake_A)

    def backward(self):
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.loss_D_A.backward()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.loss_D_B.backward()
        self.optimizer_D_B.step()


    # Backwards
    def _backwardDiscriminator(self, netD, real, fake):
        # Real
        pred_real   = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        return loss_D



    def _forwardGenerator(self):
        lambda_idt = float(self.settings['controllers']['cycleGAN']['identity'])
        lambda_A   = float(self.settings['controllers']['cycleGAN']['lambda_A'])
        lambda_B   = float(self.settings['controllers']['cycleGAN']['lambda_B'])

        ## GAN loss
        # GAN loss D_A(G_A(A))
        pred_fake = self.netD_A(self.fake_B)
        loss_G_A  = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(self.fake_A)
        loss_G_B  = self.criterionGAN(pred_fake, True)

        self.loss_GAN = loss_G_A + loss_G_B

        ## cycle consistency loss
        # Forward cycle loss
        self.loss_cycle_A   = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B   = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_first_cycle = self.loss_cycle_A + self.loss_cycle_B

        ## Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

            self.loss_idt   = self.loss_idt_A + self.loss_idt_B
        else:         
            loss_idt = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        ## second cycle consistensy (optional)
        if self.settings['controllers']['cycleGAN']['secondCycle']:
            self.loss_cycle_B_2 = self.criterionCycle(self.rec_B_2, self.real_B) * lambda_B 
            self.loss_cycle_A_2 = self.criterionCycle(self.rec_A_2, self.real_A) * lambda_A
            self.loss_second_cycle = self.loss_cycle_A_2 + self.loss_cycle_B_2 
        else:
            loss_cycle_B_2 = None
            loss_cycle_A_2 = None
            self.loss_second_cycle = 0

        # combined loss
        self.loss_G = self.loss_GAN +  self.loss_idt + self.loss_first_cycle + self.loss_second_cycle


