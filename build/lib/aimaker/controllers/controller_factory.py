from aimaker.controllers.cycleGAN_controller import CycleGANController
from aimaker.controllers.pix2pix_controller import Pix2pixController
from aimaker.controllers.deep_voxel_flow_controller import DeepVoxelFlowController
from aimaker.controllers.starGAN_controller import StarGANController
from aimaker.controllers.pix2pix_divided_gpu_controller import Pix2pixDividedGPUController
from aimaker.controllers.simple_controller import SimpleController
from aimaker.controllers.mind.pix2pix_controller import Pix2PixForMDController
from aimaker.controllers.mind.optNet_controller import OptNetController, OptNetController2


class ControllerFactory:
    def __init__(self, settings):
        self.controller_dic = {'cycleGAN': CycleGANController,
                               'pix2pix': Pix2pixController,
                               'pix2pixMulti': Pix2pixDividedGPUController,
                               'voxel': DeepVoxelFlowController,
                               'stargan': StarGANController,
                               'mind': Pix2PixForMDController,
                               'optNet': OptNetController,
                               'optNet2': OptNetController2,
                               'simple': SimpleController,
                               }
        self.settings = settings

    def create(self, name):
        if not name in self.controller_dic:
            raise NotImplementedError(('{} is wrong key word for ' +
                                       '{}. choose {}')
                                      .format(name, self.__class__.__name__, self.controller_dic.keys()))
        return self.controller_dic[name](self.settings)
