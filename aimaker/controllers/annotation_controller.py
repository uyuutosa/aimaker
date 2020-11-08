# [0:3ifrom models.model_factory import ModelFactory
# import model_factory as mf
import aimaker.utils.util as util
import os
import itertools as it
import torch
import torch.nn as nn
import torch.autograd as autograd


class AnnotationController:
    def __init__(self, settings: EasyDict):
        import aimaker.models.model_factory as mf
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.settings = settings
        ch = util.ConfigHandler(settings)
        self.gpu_ids = ch.get_GPU_ID()
        self.checkpoints_dir = ch.get_check_points_dir()
        self.pool = util.ImagePool(int(ch.settings['annotation']['imagePoolSize']))

        model_factory = mf.ModelFactory(settings)
        loss_factory = lf.LossFactory(settings)
        optimizer_factory = of.OptimizerFactory(settings)

        name = settings['annotation']['generatorModel']

        self.downsampler = nn.AvgPool2d(8)
        self.upsampler = nn.Upsample(scale_factor=8)

        if not self.load_models():
            self.generator = model_factory.create(name)
            if settings['dataset'].getboolean('isTrain'):
                name = settings['annotation']['discriminatorModel']
                self.discriminator = model_factory.create(name)

        self.generator_criterion = loss_factory \
            .create(settings['annotation']['generatorCriterion'])

        if len(self.gpu_ids):
            self.generator = self.generator.cuda(self.gpu_ids[0])

        if settings['dataset'].getboolean('isTrain'):
            if len(self.gpu_ids):
                self.discriminator = self.discriminator.cuda(self.gpu_ids[0])

            self.discriminator_criterion = loss_factory.create(settings['annotation']['discriminatorCriterion'])
            if len(self.gpu_ids):
                self.generator_criterion = self.generator_criterion.cuda(self.gpu_ids[0])
                self.discriminator_criterion = self.discriminator_criterion.cuda(self.gpu_ids[0])

            self.generator_optimizer = optimizer_factory.create( \
                settings['annotation']['generatorOptimizer']) \
                (self.generator.parameters(), settings)
            self.discriminator_optimizer = optimizer_factory.create(settings['annotation']['discriminatorOptimizer']) \
                (self.discriminator.parameters(), settings)

        if settings['ui'].getboolean('isShowModelInfo'):
            self.show_model()

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def set_input(self, inputs, is_volatile=False):
        real, target = inputs

        if len(self.gpu_ids):
            real = real.cuda(self.gpu_ids[0])
            target = target.cuda(self.gpu_ids[0])
            # real  = real.cuda(self.gpu_ids[0], async=True)
            # target = target.cuda(self.gpu_ids[0], async=True)

        self.real = autograd.Variable(real, volatile=is_volatile)
        self.target = autograd.Variable(target, volatile=is_volatile)

    def forward(self):
        self.out1, self.out2, self.out3, self.out4, self.fake = self.generator(self.real)
        self.generator_loss = self._calc_generator_loss((self.out1, self.out2, self.out3, self.out4, self.fake),
                                                        self.target)
        self.discriminator_loss = self._calc_discriminator_loss(self.real, self.fake, self.target)

    def backward(self):
        # generator
        self.generator_optimizer.zero_grad()
        self.generator_loss.backward()
        self.generator_optimizer.step()

        # discriminator
        self.discriminator_optimizer.zero_grad()
        self.discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def _calc_generator_loss(self, outputs, target):
        out1, out2, out3, out4, fake = outputs

        _lambda = float(self.settings['annotation']['lambda'])
        _lambda /= 5
        loss = self.discriminator_criterion(self.discriminator(fake), True)
        downsampled_target = self.downsampler(target)
        loss += self.generator_criterion(out1, downsampled_target) * _lambda
        loss += self.generator_criterion(out2, downsampled_target) * _lambda
        loss += self.generator_criterion(out3, downsampled_target) * _lambda
        loss += self.generator_criterion(out4, downsampled_target) * _lambda
        loss += self.generator_criterion(fake, downsampled_target) * _lambda
        return loss

    def _calc_discriminator_loss(self, real, fake, target):
        real_loss = self.discriminator_criterion(self.discriminator(real), True)
        fake_loss = self.discriminator_criterion(self.discriminator(fake.detach()), False)
        return (real_loss + fake_loss) * 0.5

    def set_mode(self, mode):
        if mode == 'train':
            self.generator = self.generator.train(True)
            self.discriminator = self.discriminator.train(True)
        elif mode == 'test' or mode == 'valid':
            self.generator = self.generator.eval()
            self.discriminator = self.discriminator.eval()

    def show_model(self):
        print('---------- Networks initialized ----------')
        self.print_network(self.generator)
        if self.settings['global'].getboolean('isTrain'):
            self.print_network(self.discriminator)
        print('------------------------------------------')

    def save_models(self):
        name = util.fcnt(self.checkpoints_dir, "annotation_controller_generator", "pth")
        torch.save(self.generator, name)
        print("{} is saved".format(name))
        name = util.fcnt(self.checkpoints_dir, "annotation_controller_discriminator", "pth")
        torch.save(self.discriminator, name)
        print("{} is saved".format(name))
        #        torch.save(self.discriminator, util.fcnt(self.checkpoints_dir, "annotation controller_discriminator", "pth"))
        print("done save models")

    def load_models(self, ):
        try:
            self.generator = torch.load(
                util.fcnt_load(self.checkpoints_dir, "annotation_controller_generator", "pth")).cuda(self.gpu_ids[0])
            self.discriminator = torch.load(
                util.fcnt_load(self.checkpoints_dir, "annotation_controller_discriminator", "pth")).cuda(
                self.gpu_ids[0])
            self.generator.setConfig(self.settings)
            self.discriminator.setConfig(self.settings)
            return True
        except:
            print("Checkpoint directory or files could not be found." +
                  "New directory {} will be created.".format(self.checkpoints_dir))
            return False

    def print_network(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    def get_losses(self):
        return {"generator": self.generator_loss.data[0],
                "discriminator": self.discriminator_loss.data[0]}

    def get_images(self):
        ret_dic = {"real": self.real.data[0][0:3],
                   "fake": self.upsampler(self.fake).data[0][0:3],
                   "out1": self.upsampler(self.out1).data[0][0:3],
                   "out2": self.upsampler(self.out2).data[0][0:3],
                   "out3": self.upsampler(self.out3).data[0][0:3],
                   "out4": self.upsampler(self.out4).data[0][0:3],
                   "target": self.downsampler(self.target.data[0][0:3]),

                   "interp": torch.cat([self.real.data[0][0:3], self.target.data[0][0:3]]).max(0)[0],
                   }
        ret_dic.update(self.generator.getFeature())
        ret_dic.update(self.discriminator.getFeature())
        return ret_dic

    def predict(self, tensor, isreturn=False):
        if not isinstance(tensor, autograd.Variable):
            tensor = autograd.Variable(tensor, volatile=True)
        predicted_tensor = self.generator.eval()(tensor).data.cpu()
        return predicted_tensor

# class AnnotationController:
#    def __init__(self, config):
#        import aimaker.models.model_factory as mf
#        import aimaker.loss.loss_factory as lf
#        import aimaker.optimizers.optimizer_factory as of
#
#        self.config          = config
#        ch                   = util.ConfigHandler(config)
#        self.gpu_ids         = ch.getGPUID()
#        self.checkpoints_dir = ch.getCheckPointsDir()
#        self.pool = util.ImagePool(int(ch.config['annotation']['imagePoolSize']))
#
#        model_factory        = mf.ModelFactory(config)
#        loss_factory         = lf.LossFactory(config)
#        optimizer_factory    = of.OptimizerFactory(config)
#
#        name                 = config['annotation']['generatorModel']
#
#        self.downsampler     = nn.AvgPool2d(8)
#        self.upsampler       = nn.Upsample(scale_factor=8)
#
#
#        if not self.loadModels():
#            self.generator       = model_factory.create(name) 
#            if config['dataset'].getboolean('isTrain'):
#                name = config['annotation']['discriminatorModel']
#                self.discriminator = model_factory.create(name) 
#
#        self.generator_criterion     = loss_factory\
#                                       .create(config['annotation']['generatorCriterion'])
#                                       
#
#        if len(self.gpu_ids):
#            self.generator = self.generator.cuda(self.gpu_ids[0])
#
#        if config['dataset'].getboolean('isTrain'):
#            if len(self.gpu_ids):
#                self.discriminator = self.discriminator.cuda(self.gpu_ids[0])
#                
#            self.discriminator_criterion = loss_factory.create(config['annotation']['discriminatorCriterion'])
#            if len(self.gpu_ids):
#                self.generator_criterion     = self.generator_criterion.cuda(self.gpu_ids[0])   
#                self.discriminator_criterion = self.discriminator_criterion.cuda(self.gpu_ids[0])   
#
#            self.generator_optimizer     = optimizer_factory.create(\
#                                               config['annotation']['generatorOptimizer'])\
#                                               (self.generator.parameters(), config)
#            self.discriminator_optimizer = optimizer_factory.create(config['annotation']['discriminatorOptimizer'])\
#                                               (self.discriminator.parameters(), config)
#
#        if config['ui'].getboolean('isShowModelInfo'):
#            self.showModel()
#
#
#    def __call__(self, inputs, is_volatile=False):
#        self.setInput(inputs, is_volatile=is_volatile)
#
#    def setInput(self, inputs, is_volatile=False):
#        real, target  = inputs
#
#        if len(self.gpu_ids):
#            real  = real.cuda(self.gpu_ids[0], async=True)
#            target = target.cuda(self.gpu_ids[0], async=True)
#
#        self.real   = autograd.Variable(real, volatile=is_volatile)
#        self.target = autograd.Variable(target, volatile=is_volatile)
#
#    def forward(self):
#        self.out1, self.out2, self.out3, self.out4, self.fake = self.generator(self.real)
#        if self.real.size() == self.fake.size():
#            self.fake_AB = self.pool.query(torch.cat((self.real, self.fake), 1))
#        else:
#            self.fake_AB = self.pool.query(torch.cat((self.fake, self.fake), 1))
#        self.generator_loss = self._calcGeneratorLoss((self.out1, self.out2, self.out3, self.out4, self.fake), self.fake_AB, self.target)
#
#        if self.real.size() == self.fake.size():
#            real_AB = torch.cat((self.real, self.target), 1)
#        else:
#            real_AB = torch.cat((self.target, self.target), 1)
#        self.discriminator_loss = self._calcDiscriminatorLoss(real_AB, self.fake_AB, self.target)
#
#    def backward(self):
#        # generator
#        self.generator_optimizer.zero_grad()
#        self.generator_loss.backward()
#        self.generator_optimizer.step()
#
#        # discriminator
#        self.discriminator_optimizer.zero_grad()
#        self.discriminator_loss.backward()
#        self.discriminator_optimizer.step()
#
#    def _calcGeneratorLoss(self, outputs, fake_AB, target):
#        out1, out2, out3, out4, fake = outputs
#
#        _lambda = float(self.config['annotation']['lambda'])
#        _lambda /= 5
#        #print(fake_AB.shape)
#        #print(out1.shape)
#        #print(out2.shape)
#        #print(out3.shape)
#        #print(out4.shape)
#        loss = self.discriminator_criterion(self.discriminator(fake_AB), True)
#        downsampled_target = self.downsampler(target)
#        loss += self.generator_criterion(out1, downsampled_target) * _lambda
#        loss += self.generator_criterion(out2, downsampled_target) * _lambda
#        loss += self.generator_criterion(out3, downsampled_target) * _lambda
#        loss += self.generator_criterion(out4, downsampled_target) * _lambda
#        loss += self.generator_criterion(fake, downsampled_target) * _lambda
#        return loss
#
#    def _calcDiscriminatorLoss(self, real_AB, fake_AB, target):
#        real_loss = self.discriminator_criterion(self.discriminator(real_AB), True)
#        fake_loss = self.discriminator_criterion(self.discriminator(fake_AB.detach()), False)
#        return (real_loss + fake_loss) * 0.5
#
#    def setMode(self, mode):
#        if mode == 'train':
#            self.generator     = self.generator.train(True)
#            self.discriminator = self.discriminator.train(True)
#        elif mode == 'test' or mode == 'valid':
#            self.generator     = self.generator.eval()
#            self.discriminator = self.discriminator.eval()
#        
#
#    def showModel(self):
#        print('---------- Networks initialized ----------')
#        self.printNetwork(self.generator)
#        if self.config['global'].getboolean('isTrain'):
#            self.printNetwork(self.discriminator)
#        print('------------------------------------------')
#
#    def saveModels(self):
#        name = util.fcnt(self.checkpoints_dir, "annotation_controller_generator", "pth")
#        torch.save(self.generator, name)
#        print("{} is saved".format(name))
#        name = util.fcnt(self.checkpoints_dir, "annotation_controller_discriminator", "pth")
#        torch.save(self.discriminator, name)
#        print("{} is saved".format(name))
##        torch.save(self.discriminator, util.fcnt(self.checkpoints_dir, "annotation controller_discriminator", "pth"))
#        print("done save models")
#
#    def loadModels(self,):
#        try:
#            self.generator     = torch.load(util.fcnt_load(self.checkpoints_dir, "annotation_controller_generator",     "pth")).cuda(self.gpu_ids[0])
#            self.discriminator = torch.load(util.fcnt_load(self.checkpoints_dir, "annotation_controller_discriminator", "pth")).cuda(self.gpu_ids[0])
#            self.generator.setConfig(self.config)
#            self.discriminator.setConfig(self.config)
#            return True
#        except:
#            print("Checkpoint directory or files could not be found."+
#                  "New directory {} will be created.".format(self.checkpoints_dir))
#            return False
#
#    def printNetwork(self, net):
#        num_params = 0
#        for param in net.parameters():
#            num_params += param.numel()
#        print(net)
#        print('Total number of parameters: %d' % num_params)
#
#
#
#    def getLosses(self):
#        return {"generator"      : self.generator_loss.data[0],
#                "discriminator"  : self.discriminator_loss.data[0]}
#
#    def getImages(self):
#        ret_dic = {"real"   : self.real.data[0][0:3],
#                   "fake"   : self.upsampler(self.fake).data[0][0:3],
#                   "out1"   : self.upsampler(self.out1).data[0][0:3],
#                   "out2"   : self.upsampler(self.out2).data[0][0:3],
#                   "out3"   : self.upsampler(self.out3).data[0][0:3],
#                   "out4"   : self.upsampler(self.out4).data[0][0:3],
#                   "target" : self.target.data[0][0:3],
#
#                   "interp" : torch.cat([self.real.data[0][0:3], self.target.data[0][0:3]]).max(0)[0],
#                   }
#        ret_dic.update(self.generator.getFeature())
#        ret_dic.update(self.discriminator.getFeature())
#        return ret_dic
#
#
#    def predict(self, tensor, isreturn=False):
#        if not isinstance(tensor, autograd.Variable):
#            tensor = autograd.Variable(tensor, volatile=True)
#        predicted_tensor = self.generator.eval()(tensor).data.cpu()
#        return predicted_tensor
#
