import aimaker.utils.util as util
import os
import itertools as it
import torch 
import torch.nn as nn
import torch.autograd as autograd
import itertools
import numpy as np


class StarGANController:
    def __init__(self, config):
        import aimaker.models.model_factory as mf
        import aimaker.loss.loss_factory as lf
        import aimaker.optimizers.optimizer_factory as of

        self.config          = config
        ch                   = util.ConfigHandler(config)
        self.gpu_ids         = ch.getGPUID()
        self.checkpoints_dir = ch.getCheckPointsDir()

        model_factory        = mf.ModelFactory(config)
        loss_factory         = lf.LossFactory(config)
        optimizer_factory    = of.OptimizerFactory(config)
        self.n_batch         = ch.getBatchSize()

        self.cDim            = len(self.config['starGAN settings']['typeNames'].strip().split(","))

        name                 = config['starGAN settings']['generatorModel']

        if not self.loadModels():
            self.generator       = model_factory.create(name) 
            if config['dataset settings'].getboolean('isTrain'):
                name = config['starGAN settings']['discriminatorModel']
                self.discriminator = model_factory.create(name) 

        self.reconstruct_criterion = loss_factory\
                                       .create(config['starGAN settings']['reconstructCriterion'])
        self.cls_criterion         = loss_factory\
                                       .create(config['starGAN settings']['clsCriterion'])
                                       
        self.lambda_rec = float(config['starGAN settings']['lambdaRec'])
        self.lambda_cls = float(config['starGAN settings']['lambdaCls'])
        self.lambda_gp  = float(config['starGAN settings']['lambdaGp'])

        if len(self.gpu_ids):
            self.generator = self.generator.cuda(self.gpu_ids[0])

        if config['dataset settings'].getboolean('isTrain'):
            if len(self.gpu_ids):
                self.discriminator = self.discriminator.cuda(self.gpu_ids[0])
                
            self.discriminator_criterion = loss_factory.create(config['starGAN settings']['discriminatorCriterion'])
            if len(self.gpu_ids):
                #self.generator_criterion     = self.generator_criterion.cuda(self.gpu_ids[0])   
                self.discriminator_criterion = self.discriminator_criterion.cuda(self.gpu_ids[0])   

            self.generator_optimizer     = optimizer_factory.create(\
                                               config['starGAN settings']['generatorOptimizer'])\
                                               (self.generator.parameters(), config)
            self.discriminator_optimizer = optimizer_factory.create(config['starGAN settings']['discriminatorOptimizer'])\
                                               (self.discriminator.parameters(), config)

        if config['ui settings'].getboolean('isShowModelInfo'):
            self.showModel()

    def __call__(self, inputs, is_volatile=False):
        self.setInput(inputs, is_volatile=is_volatile)

    def setInput(self, inputs, is_volatile=False):
        real, real_c  = inputs
        rand_idx = torch.randperm(real_c.size(0))
        fake_c = real_c[rand_idx]

        if len(self.gpu_ids):
            real   = real.cuda(self.gpu_ids[0])
            real_c = real_c.cuda(self.gpu_ids[0])
            fake_c = fake_c.cuda(self.gpu_ids[0])

        self.real   = autograd.Variable(real,   volatile=is_volatile)
        self.real_c = autograd.Variable(real_c, volatile=is_volatile)
        self.fake_c = autograd.Variable(fake_c, volatile=is_volatile)

    def _concatLabelsToImages(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return x

    def forward(self):
        #print(self._concatLabelsToImages(self.real, self.fake_c).shape)
        self.fake = self.generator(self._concatLabelsToImages(self.real, self.fake_c))
        self.rec  = self.generator(self._concatLabelsToImages(self.fake, self.real_c))

        self.generator_loss     = self._calcGeneratorLoss(self.real, self.fake, self.rec, self.fake_c)
        self.generator_optimizer.zero_grad()
        self.generator_loss.backward()
        self.generator_optimizer.step()

        self.fake = self.generator(self._concatLabelsToImages(self.real, self.fake_c))
        self.discriminator_loss = self._calcDiscriminatorLoss(self.real, self.fake, self.real_c)

        self.discriminator_optimizer.zero_grad()
        self.discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def backward(self):
        # generator
        # discriminator
        pass

    def _calcGeneratorLoss(self, real, fake, rec, fake_c):
        fake_discriminated, fake_cls = self.discriminator(fake)
        fake_loss = self.discriminator_criterion(fake_discriminated, True)
        #fake_loss = -torch.mean(fake_discriminated)#, True)
        #rec_loss  = torch.mean(torch.abs(real - rec))
        rec_loss   = self.reconstruct_criterion(rec, real)
        cls_loss   = self.cls_criterion(fake_cls, fake_c.detach())
        loss       = 10 * fake_loss + self.lambda_rec * rec_loss + self.lambda_cls * cls_loss

        self.g_fake_loss = fake_loss 
        self.g_cls_loss  = cls_loss 
        self.g_rec_loss  = rec_loss 
        return loss 

    def _calcGradientPenalty(self, real, fake):
        alpha      = torch.rand(real.size(0), 1, 1, 1).cuda(self.gpu_ids[0])
        x_hat = alpha * real.data + (1 - alpha) * fake.data
        x_hat[x_hat == 0] = 0.1
        x_hat      = autograd.Variable(x_hat, requires_grad=True) 
        out_src, _ = self.discriminator(x_hat)

        weight     = torch.ones(out_src.size()).cuda(self.gpu_ids[0])
        dydx       = torch.autograd.grad(outputs=out_src,
                                     inputs=x_hat,
                                     grad_outputs=weight,
                                     retain_graph=True,
                                     create_graph=True,
                                     only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1) ** 2)


    def _calcDiscriminatorLoss(self, real, fake, real_c):
        real_discriminated, real_cls = self.discriminator(real)
        fake_discriminated, fake_cls = self.discriminator(fake.detach())
        #real_loss = - torch.mean(real_discriminated)
        real_loss = self.discriminator_criterion(real_discriminated, True)
        fake_loss = self.discriminator_criterion(fake_discriminated, False)
        #fake_loss = torch.mean(fake_discriminated)
        cls_loss  = self.cls_criterion(real_cls, real_c)
        self.d_gp_loss = self._calcGradientPenalty(self.real, self.fake)
        loss =  (real_loss + fake_loss) + self.lambda_cls * cls_loss 
        #loss =  (real_loss + fake_loss + self.lambda_gp * self.d_gp_loss) + self.lambda_cls * cls_loss 

        self.d_real_loss = real_loss
        self.d_fake_loss = fake_loss
        self.d_cls_loss = cls_loss
#        self.cls_loss = cls_loss
        return loss

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
        if self.config['dataset settings'].getboolean('isTrain'):
            self.printNetwork(self.discriminator)
        print('------------------------------------------')

    def saveModels(self):
        name = util.fcnt(self.checkpoints_dir, "starGAN_generator", "pth")
        torch.save(self.generator, name)
        print("{} is saved".format(name))
        name = util.fcnt(self.checkpoints_dir, "starGAN_discriminator", "pth")
        torch.save(self.discriminator, name)
        print("{} is saved".format(name))
#        torch.save(self.discriminator, util.fcnt(self.checkpoints_dir, "starGAN_discriminator", "pth"))
        print("done save models")

    def loadModels(self,):
        try:
            self.generator     = torch.load(util.fcnt_load(self.checkpoints_dir, "starGAN_generator",     "pth")).cuda(self.gpu_ids[0])
            self.discriminator = torch.load(util.fcnt_load(self.checkpoints_dir, "starGAN_discriminator", "pth")).cuda(self.gpu_ids[0])
            self.generator.setConfig(self.config)
            self.discriminator.setConfig(self.config)
            return True
        except:
            print("Checkpoint directory or files could not be found."+
                  "New directory {} will be created.".format(self.checkpoints_dir))
            return False

    def printNetwork(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)



    def getLosses(self):
        return {"generator"     : self.generator_loss.data[0],
                "discriminator" : self.discriminator_loss.data[0],
                "g_fake_loss"   : self.g_fake_loss.data[0],
                "g_cls_loss"    : self.g_cls_loss.data[0],
                "g_rec_loss"    : self.g_rec_loss.data[0],
                "d_real_loss"   : self.d_real_loss.data[0],
                "d_fake_loss"   : self.d_fake_loss.data[0],
                "d_cls_loss"    : self.d_cls_loss.data[0],
                "d_gp_loss"     : self.d_gp_loss.data[0]}

    def _genAllImages(self):
        nums = [0,1]
        lst = []
        

        for num in itertools.product(nums, repeat=int(self.cDim)):
            lst += [num]
            #type_arr = torch.Tensor([num])
            #if len(self.gpu_ids):
            #    type_arr = type_arr.cuda(self.gpu_ids[0])

            #type_arr = autograd.Variable(type_arr, volatile=True)
            #lst += [self.generator(self._concatLabelsToImages(self.real[0:1].repeat(self.n_batch, 1, 1, 1), type_arr))[0:1, 0:3].data]
            #lst += [self.generator(self._concatLabelsToImages(self.real[0:1].repeat(self.n_batch, 1, 1, 1), type_arr.repeat(self.n_batch, 1)))[0:1, 0:3].data]
#            lst += [self.generator(self._concatLabelsToImages(self.real, self.fake_c)).data[0:1, 0:3]]
        type_arr = torch.Tensor(lst)
        type_arr = torch.eye(self.cDim)
        if len(self.gpu_ids):
            type_arr = type_arr.cuda(self.gpu_ids[0])

        type_arr = autograd.Variable(type_arr, volatile=True)
        return self.generator(self._concatLabelsToImages(self.real[0:1].repeat(self.cDim, 1, 1, 1), type_arr[0:self.cDim]))[:, 0:3].data
        #return torch.cat(lst, 0)

    def getImages(self):


        ret_dic = {"real"   : self.real.data[:,0:3],
                   "fake"   : self.fake.data[0][0:3],
                   "rec"    : self.rec.data[0][0:3],
                   "all"    : self._genAllImages(),
                   }
                 #  "target" : self.target.data[0][0:3]}
        ret_dic.update(self.generator.getFeature())
        ret_dic.update(self.discriminator.getFeature())
        return ret_dic


    def predict(self, tensor, type_arr, isreturn=False, gpu_id=''):
        if not isinstance(tensor, autograd.Variable):
            tensor = autograd.Variable(tensor, volatile=True)
        if not isinstance(type_arr, autograd.Variable):
            type_arr = autograd.Variable(type_arr, volatile=True)

        if not len(gpu_id):
            self.generator = self.generator.cpu()
        else:
            self.generator = self.generator.cuda(int(gpu_id))

        predicted_tensor = self.generator.eval()(self._concatLabelsToImages(tensor, type_arr)).data.cpu().numpy()
        predicted_arr = ((predicted_tensor[0].transpose(1,2,0) + 1) / 2 * 255).astype(np.uint8)
        return predicted_arr

