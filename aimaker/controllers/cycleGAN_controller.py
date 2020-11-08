#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from easydict import EasyDict
import itertools as it
import torch
import torch.nn as nn
import torch.autograd as autograd

from aimaker.utils import SettingHandler, ImagePool, fcnt, fcnt_load, peelVariable


class CycleGANController:
    def __init__(self, settings: EasyDict):
        import aimaker.models.model_factory as mf
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.settings = settings
        ch = SettingHandler(settings)

        self.gpu_ids = ch.get_GPU_ID()
        self.checkpoints_dir = ch.get_check_points_dir()
        model_factory = mf.ModelFactory(settings)
        loss_factory = lf.LossFactory(settings)
        optimizer_factory = of.OptimizerFactory(settings)

        # for discriminator regularization
        self.pool_fake_A = ImagePool(int(settings.controllers.cyclegan.image_poolSize))
        self.pool_fake_B = ImagePool(int(settings.controllers.cyclegan.image_poolSize))

        name = settings.controllers.cycle_gan.generator_model
        self.netG_A = model_factory.create(name)
        self.netG_B = model_factory.create(name)
        if len(self.gpu_ids):
            self.netG_A = self.netG_A.cuda(self.gpu_ids[0])
            self.netG_B = self.netG_B.cuda(self.gpu_ids[0])

            name = settings.controllers.cyclegan.discriminatorModel
            self.netD_A = model_factory.create(name)
            self.netD_B = model_factory.create(name)
            if len(self.gpu_ids):
                self.netD_A = self.netD_A.cuda(self.gpu_ids[0])
                self.netD_B = self.netD_B.cuda(self.gpu_ids[0])

        self.load_models()

        self.criterion_gan = loss_factory.create("GANLoss")
        self.criterion_cycle = loss_factory.create(settings.controllers.cycleGAN.cycle_loss)
        self.criterion_idt = loss_factory.create(settings.controllers.cycleGAN.idt_loss)
        if len(self.gpu_ids):
            self.criterion_gan = self.criterion_gan.cuda(self.gpu_ids[0])
            self.criterion_cycle = self.criterion_cycle.cuda(self.gpu_ids[0])
            self.criterion_idt = self.criterion_idt.cuda(self.gpu_ids[0])

            # initialize optimizers
        self.optimizer_G = optimizer_factory.create(settings['controllers']['cycleGAN']['generatorOptimizer'])(
            it.chain(self.netG_A.parameters(), self.netG_B.parameters()), settings)

        if settings['data']['base']['isTrain']:
            self.optimizer_D_A = optimizer_factory.create(settings['controllers']['cycleGAN']['D_AOptimizer'])(
                self.netD_A.parameters(), settings)
            self.optimizer_D_B = optimizer_factory.create(settings['controllers']['cycleGAN']['D_BOptimizer'])(
                self.netD_B.parameters(), settings)

        if settings['ui']['base']['isShowModelInfo']:
            self.show_model()

    def set_mode(self, mode):
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

    def show_model(self):
        print('---------- Networks initialized ----------')
        self.print_network(self.netG_A)
        self.print_network(self.netG_B)
        if self.settings['data']['base']['isTrain']:
            self.print_network(self.netD_A)
            self.print_network(self.netD_B)
        print('------------------------------------------')

    def save_models(self):
        torch.save(self.netG_A, fcnt(self.checkpoints_dir, "netG_A", "pth"))
        torch.save(self.netG_B, fcnt(self.checkpoints_dir, "netG_B", "pth"))
        torch.save(self.netD_A, fcnt(self.checkpoints_dir, "netD_A", "pth"))
        torch.save(self.netD_B, fcnt(self.checkpoints_dir, "netD_B", "pth"))
        print("done save models")

    def load_models(self, ):
        try:
            self.netG_A = torch.load(fcnt_load(self.checkpoints_dir, "netG_A", "pth")).cuda(self.gpu_ids[0])
            self.netG_B = torch.load(fcnt_load(self.checkpoints_dir, "netG_B", "pth")).cuda(self.gpu_ids[0])
            self.netD_A = torch.load(fcnt_load(self.checkpoints_dir, "netD_A", "pth")).cuda(self.gpu_ids[0])
            self.netD_B = torch.load(fcnt_load(self.checkpoints_dir, "netD_B", "pth")).cuda(self.gpu_ids[0])
        except:
            print("Checkpoint directory could not be found." +
                  "New directory {} is created.".format(self.checkpoints_dir))

    def print_network(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    def get_losses(self):

        return {"all": peelVariable(self.loss_G).item(),
                "DA": peelVariable(self.loss_D_A).item(),
                "DB": peelVariable(self.loss_D_B).item(),
                "cycleA": peelVariable(self.loss_cycle_A).item(),
                "cycleB": peelVariable(self.loss_cycle_B).item(),
                "cycleA2": peelVariable(self.loss_cycle_A_2).item(),
                "cycleB2": peelVariable(self.loss_cycle_B_2).item(),
                "idtA": peelVariable(self.loss_idt_A).item(),
                "idtB": peelVariable(self.loss_idt_B).item()}

    def get_images(self):

        return {"realA": peelVariable(self.real_A)[0],
                "fakeB": peelVariable(self.fake_B)[0],
                "realB": peelVariable(self.real_B)[0],
                "fakeA": peelVariable(self.fake_A)[0],
                "cycleA": peelVariable(self.rec_A)[0],
                "cycleB": peelVariable(self.rec_B)[0],
                "cycleA2": peelVariable(self.rec_A_2)[0],
                "cycleB2": peelVariable(self.rec_B_2)[0],
                "idtA": peelVariable(self.idt_A)[0],
                "idtB": peelVariable(self.idt_B)[0]}

    def __call__(self, inputs, is_volatile=False):
        self.set_input(inputs, is_volatile)

    def set_input(self, inputs, is_volatile=False):
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
        self.rec_A = self.netG_B(self.fake_B)
        self.rec_B = self.netG_A(self.fake_A)
        self.idt_A = self.netG_B(self.fake_A)
        self.idt_B = self.netG_A(self.fake_B)
        if self.settings['controllers']['cycleGAN']['secondCycle']:
            self.rec_A_2 = self.netG_A(self.rec_A)
            self.rec_B_2 = self.netG_B(self.rec_B)
        else:
            self.rec_A_2 = [None]
            self.rec_B_2 = [None]
            self.loss_cycle_A_2 = [None]
            self.loss_cycle_B_2 = [None]

        self._forward_generator()
        self._forward_discriminator_A()
        self._forward_discriminator_B()

    def _forward_discriminator_A(self):
        fake_B = self.pool_fake_B.query(self.fake_B)
        self.loss_D_A = self._backward_discriminator(self.netD_A, self.real_B, fake_B)

    def _forward_discriminator_B(self):
        fake_A = self.pool_fake_A.query(self.fake_A)
        self.loss_D_B = self._backward_discriminator(self.netD_B, self.real_A, fake_A)

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
    def _backward_discriminator(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        return loss_D

    def _forward_generator(self):
        lambda_idt = float(self.settings['controllers']['cycleGAN']['identity'])
        lambda_A = float(self.settings['controllers']['cycleGAN']['lambda_A'])
        lambda_B = float(self.settings['controllers']['cycleGAN']['lambda_B'])

        ## GAN loss
        # GAN loss D_A(G_A(A))
        pred_fake = self.netD_A(self.fake_B)
        loss_G_A = self.criterion_gan(pred_fake, True)

        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(self.fake_A)
        loss_G_B = self.criterion_gan(pred_fake, True)

        self.loss_GAN = loss_G_A + loss_G_B

        ## cycle consistency loss
        # Forward cycle loss
        self.loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * lambda_B

        self.loss_first_cycle = self.loss_cycle_A + self.loss_cycle_B

        ## Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.loss_idt_A = self.criterion_idt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.loss_idt_B = self.criterion_idt(self.idt_B, self.real_A) * lambda_A * lambda_idt

            self.loss_idt = self.loss_idt_A + self.loss_idt_B
        else:
            loss_idt = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        ## second cycle consistensy (optional)
        if self.settings['controllers']['cycleGAN']['secondCycle']:
            self.loss_cycle_B_2 = self.criterion_cycle(self.rec_B_2, self.real_B) * lambda_B
            self.loss_cycle_A_2 = self.criterion_cycle(self.rec_A_2, self.real_A) * lambda_A
            self.loss_second_cycle = self.loss_cycle_A_2 + self.loss_cycle_B_2
        else:
            loss_cycle_B_2 = None
            loss_cycle_A_2 = None
            self.loss_second_cycle = 0

        # combined loss
        self.loss_G = self.loss_GAN + self.loss_idt + self.loss_first_cycle + self.loss_second_cycle
