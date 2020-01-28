import torch.nn as nn

#from aimaker.models.resnet_model                   import ResNetForGeneratorModel, ResNetForClassificationModel
#from aimaker.models.patchGAN_model                 import PatchGANModel
#from aimaker.models.patchGAN_md_model              import PatchGANMDPlusModel
#from aimaker.models.mind.patchGAN_md_model         import PatchGANMDPlusModelForMD
#from aimaker.models.mind.optical_net               import OpticalNet
#from aimaker.models.unet_model                     import UnetForGeneratorModel
#from aimaker.models.SRGAN_model                    import SRGANDiscriminator, SRGANGenerator
#from aimaker.models.adapter_model                  import AdapterModel, LinearModel
#from aimaker.models.longbottle_model               import LongBottleModel
#from aimaker.models.identity_model                 import IdentityModel
#from aimaker.models.base_sequential                import BaseSequential
#from aimaker.models.global_local_model             import GLPatchGANModel, GLGeneratorModel
#from aimaker.models.stargan_model                  import StarGANGeneratorModel, StarGANDiscriminatorModel
#from aimaker.models.PSPNet_model                   import PSPNetGeneratorModel, PSPNetGeneratorModelDownsampled
#from aimaker.models.mixing_model                   import MixingGeneratorModel
#from aimaker.models.pix2pixHD_models.generator     import LocalEnhancer
#from aimaker.models.pix2pixHD_models.discriminator import MultiscaleDiscriminator
#from aimaker.models.deeplab_v3_plus.generator      import DeepLabV3PlusGenerator
#from aimaker.models.pyramid_model.generator               import PyramidModelGenerator
import aimaker

from aimaker.layers.initialize_factory             import InitializeFactory

class ModelFactory:
    def __init__(self, settings):
        import aimaker.models as m
        self.dic = dir(m)
        self.settings = settings


    def create(self, name):
        import aimaker.models as m
        names = name.split("_")
        model_lst = []
        for i, name in enumerate(names):
            #if not name in self.dic:
            #    if not name.strip(name.strip('linear')) in self.model_dic:
            #        raise NotImplementedError(('{} is wrong key word for ' + \
            #                               '{}. choose {}')\
            #                               .format(name, self.__class__.__name__, self.model_dic.keys()))

            if name == 'adapter':
                pre_model = eval(f"m.{names[i-1]}(settings=self.settings)")
                #post_model = model_dic[names[i+1]](settings=self.settings)
                post_model = eval(f"{names[i+1]}(settings=self.settings)")
                model = eval(f"{name}(self.settings, pre_model, post_model)")
            elif 'linear' in name:
                n_out = int(name.strip('linear'))
#                pre_model = model_dic[names[i-1]](self.settings)
                pre_model = eval(f"{names[i-1]}(self.settings)")
                model = eval(f"m.{name}(self.settings, pre_model, n_out)")
            else:
                model = eval(f"m.{name}(settings=self.settings)")

            InitializeFactory(self.settings).create(self.settings['base']['initType'])(model)
            model_lst += [model]

        return m.BaseSequential(self.settings, *model_lst)
