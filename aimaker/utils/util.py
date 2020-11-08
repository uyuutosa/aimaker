import glob
import json
import numbers
import os
import importlib

import glob as g
import numpy as np
import torch
import torch.autograd as autograd
import torchvision.transforms as tt
import torchvision.models as m
from torch.nn import *

import aimaker.utils.transforms as my_transforms


# from aimaker.data.datasets import DatasetOfPair


class BaseFactory:
    def __init__(self, settings):
        self.settings = settings
        self.module_name = None
        self.suffix = ""

    def _import_if_exists(self, name):
        try:
            self.mod = importlib.import_module("{}".format(self.module_name))
            self.suffix = 'self.mod.'
        except:
            pass
            # import traceback
            # traceback.print_exc()
            # raise NotImplementedError(('{} is wrong key word for ' + \
            #                           '{}. choose {}')\
            #                           .format(name, self.__class__.__name__, self.data_dic.keys()))
            if self.module_name is not None:
                raise ImportError(f"{self.module_name} is not None, but it couldn't be imported.")

    def _create(self, name):
        return eval(self.suffix + f"{name}(settings=self.settings)")

    def create(self, name):
        self._import_if_exists(name)
        return self._create(name)


def load_setting(path):
    n = len(path.split('/'))
    path_lst = []
    path_lst += g.glob("{}/*.json".format(path))
    path_lst += g.glob("{}/*/*.json".format(path))
    path_lst += g.glob("{}/*/*/*.json".format(path))
    path_lst += g.glob("{}/*/*/*/*.json".format(path))
    lst_nested = [x.split('/')[n:] for x in [x.replace('.json', '') for x in path_lst]]
    setting_dic = {}

    for lst, path in zip(lst_nested, path_lst):
        try:
            setting_dic = load_setting_rec(lst, path, setting_dic)
        except:
            import traceback
            traceback.print_exc()
            raise ValueError('Loading "{}" is failed.'.format(path))
    return setting_dic


def load_setting_rec(lst, path, setting_dic):
    name = lst[0]
    if len(lst) == 1:
        setting_dic[name] = json.load(open(path))
    else:
        lst = lst[1:]
        try:
            input_dic = setting_dic[name]
        except:
            input_dic = {}
        setting_dic[name] = load_setting_rec(lst, path, input_dic)
    return setting_dic


class DivideImage(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        cols, rows = img.size
        height, width = rows // self.size[0], cols // self.size[1]

        lst = []

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                idx = (i * 3 * self.size[1]) + (j * 3)
                idx_y = i * height
                idx_x = j * width
                lst += [img.crop((idx_x, idx_y, idx_x + width, idx_y + height))]
        return lst
        # c, rows, cols = img.shape
        # height, width = rows // self.size[0], cols // self.size[1]
        # ret = torch.zeros(int(3 * self.size[1] * self.size[0]), height, width)
        #
        # for i in range(self.size[0]):
        #    for j in range(self.size[1]):
        #        idx   = (i*3*self.size[1])+(j*3)
        #        idx_y =  i * height
        #        idx_x =  j * width
        #        ret[idx:idx+3] = img[:, idx_y:idx_y+height, idx_x:idx_x+width]

        # return ret


class AggregateImage(object):
    def __init__(self, size, module=torch):
        self.module = module
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        img = img * 1.
        c, rows, cols = img.shape

        height, width = rows * self.size[0], cols * self.size[1]
        ret = self.module.zeros((3, height, width))

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                idx = (i * 3 * self.size[1]) + (j * 3)
                idx_y = i * rows
                idx_x = j * cols

                ret[:, idx_y:idx_y + rows, idx_x:idx_x + cols] = img[idx:idx + 3]

        return ret


def meshgrid(xlim, ylim):
    x_arr_for_gen = torch.arange(*xlim)
    y_arr_for_gen = torch.arange(*ylim)
    x = x_arr_for_gen[None, :] * torch.ones(y_arr_for_gen.numel())[:, None]
    y = y_arr_for_gen[:, None] * torch.ones(x_arr_for_gen.numel())
    return x, y


def peelVariable(v):
    if isinstance(v, autograd.Variable):
        return v.data
    else:
        return v


def fcnt(dir=".", fname="hello", ext="txt"):
    if not os.path.exists(dir):
        os.makedirs(dir)

    lst = glob.glob('{}/*'.format(dir))
    lst.sort()
    if len(lst):
        path_lst = np.array(lst)[np.array([True if "{}/{}".format(dir, fname) in x else False for x in lst])]
        if len(path_lst):
            max_path = path_lst[-1]
            max_num = int(max_path.split(".")[-2].split("_")[-1]) + 1
        else:
            max_num = 0
    else:
        max_num = 0
    return "{}/{}_{:04d}.{}".format(dir, fname, max_num, ext)


def fcnt_load(dir=".", fname="hello", ext="txt"):
    if not os.path.exists(dir):
        raise FileNotFoundError("Directory {} is not exits.".format(dir))

    cnt = 0
    lst = glob.glob('{}/*'.format(dir))
    lst.sort()

    return np.array(lst)[np.array([True if "{}/{}".format(dir, fname) in x else False for x in lst])][-1]


class SettingHandler:
    def __init__(self, setting=None):
        self.setting = setting

    def get_tensor(self):
        if len(self.setting['base']['gpu_ids']):
            return torch.cuda.FloatTensor
        else:
            return torch.FloatTensor

    def get_number_of_input_image_channels(self):
        return int(self.setting['base']['numberOfInputImageChannels'])

    def get_number_of_output_image_channels(self):
        return int(self.setting['base']['numberOfOutputImageChannels'])

    def get_GPU_ID(self):
        return self.setting['base']['gpu_ids']

    def get_res_net(self):

        net_dic = {"18": m.resnet18,
                   "34": m.resnet34,
                   "50": m.resnet50,
                   "101": m.resnet101,
                   "152": m.resnet152}

        pretrained = self.setting['models']['resNet']['pretrain']['isPretrain']

        name = str(self.setting['models']['resNet']['pretrain']['layerNum'])

        if not name in net_dic:
            raise ValueError("Wrong resnet number. choose in 18, 34, 50, 101, 152")

        net = net_dic[name](pretrained=pretrained)
        return net

    def get_check_points_dir(self):
        return self.setting['base']['checkPointsDirectory']

    def get_batch_size(self, mode="train"):
        batch_size = self.setting['data']['base'][mode]['batchSize'] if self.setting is not None else 1
        return int(batch_size)

    def get_model_save_interval(self):
        model_save_interval = self.setting['base']['modelSaveInterval'] if self.setting is not None else 10
        return int(model_save_interval)

    def get_model_save_interval_for_valid(self):
        return int(self.setting['base']['modelSaveIntervalForValid'] if self.setting is not None else 10)

    def get_input_transform(self, mode):
        names = self.setting['data']['base'][mode]['inputTransform'] if self.setting is not None else 'toTensor'
        lst = self.parse_transform(names)
        return tt.Compose(lst)

    def get_normalize_transform(self, values="0.5,0.5,0.5,0.5,0.5,0.5"):
        names = 'toTensor_normalize' + values
        lst = self.parse_transform(names)
        return tt.Compose(lst)

    def get_target_transform(self, mode):
        names = self.setting['data']['base'][mode]['targetTransform'] if self.setting is not None else 'toTensor'
        lst = self.parse_transform(names)
        return tt.Compose(lst)

    def get_common_transform(self, mode):
        names = self.setting['data']['base'][mode]['commonTransform'] if self.setting is not None else 'None'
        lst = self.parse_transform(names)
        return tt.Compose(lst)

    def _parse_x(self, value):
        if "x" in value:
            value = [int(x) for x in value.split("x")]
        else:
            value = int(value)

        return value

    def _parse_comma(self, value):
        if "," in value:
            ret = []
            for x in value.split(','):
                if 'x' in x:
                    ret += [self._parse_x(x)]
                else:
                    ret += [float(x)]

        else:
            raise ValueError("comma is needed")
        return ret

    def parse_transform(self, names):
        lst = []
        for name in names.split('_'):
            if "resize" in name:
                value = name.strip("resize")
                lst += [tt.Resize(self._parse_x(value))]
            elif "normalize" in name:
                value = self._parse_comma(name.strip("normalize"))
                if len(value) == 2:
                    lst += [tt.Normalize(value[0:1], value[1:2])]
                else:
                    lst += [tt.Normalize(value[0:3], value[3:6])]
            elif "toTensor" in name:
                lst += [tt.ToTensor()]
            elif "randomCrop" in name:
                value = name.strip("randomCrop")
                lst += [tt.RandomCrop(self._parse_x(value))]
            elif "centerCrop" in name:
                value = name.strip("centerCrop")
                lst += [tt.CenterCrop(self._parse_x(value))]
            elif "randomVerticalFlip" in name:
                lst += [tt.RandomVerticalFlip()]
            elif "randomResizedCrop" in name:
                v = self._parse_comma(name.strip("randomResizedCrop"))
                lst += [tt.RandomResizedCrop(int(v[0]), v[1:3], v[3:5], int(v[5]))]
            elif "grayScale" in name:
                value = int(name.strip("grayScale"))
                lst += [tt.grayScale(value)]
            elif "randomRotation" in name:
                striped_name = name.strip("randomRotation")
                if ',' in striped_name:
                    v = self._parse_comma(striped_name)
                else:
                    v = int(striped_name)
                lst += [tt.RandomRotation(v)]
            elif "randomGrayscale" in name:
                v = float(value)
                lst += [tt.RandomGrayscale(v)]
            elif "toPILImage" in name:
                lst += [tt.ToPILImage()]
            elif "randomHorizontalFlip" in name:
                lst += [tt.RandomHorizontalFlip()]
            elif "randomVerticalFlip" in name:
                lst += [tt.RandomVerticalFlip()]
            elif "randomBackground" in name:
                lst += [tt.RandomBackground(self.setting['data']['base'])]
            elif "pad" in name:
                striped_name = name.strip("pad")
                v = int(striped_name)
                lst += [tt.Pad(v)]
            elif "randomPad" in name:
                striped_name = name.strip("randomPad")
                v = self._parse_comma(striped_name)
                lst += [my_transforms.RandomPad(v[0], v[1])]
            elif "colorJitter" in name:
                v = self._parse_comma(name.strip("colorJitter"))
                # default
                # tt.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                lst += [tt.ColorJitter(brightness=v[0], contrast=v[1], saturation=v[2], hue=v[3])]
            elif 'toNumpy' in name:
                lst += [my_transforms.ToNumpy()]
            elif 'correctExif' in name:
                lst += [my_transforms.CorrectExif()]
            elif 'randomAffine' in name:
                value = self._parse_comma(name.strip('randomAffine'))
                lst += [tt.RandomAffine(degrees=value[0:2], translate=value[2:4], scale=value[4:6], shear=value[6:8])]
            elif 'humanCrop' in name:
                value = self._parse_comma(name.strip("humanCrop"))
                lst += [
                    my_transforms.HumanCrop(margin=value[0], weight_path=self.setting['data']['base']['openPosePath'],
                                            scale=value[1], gpu_ids=str(int(value[2])))]
            elif 'toHSV' in name:
                lst += [my_transforms.ToHSV()]
            elif name == 'None':
                pass
            else:
                raise NotImplementedError("{} is could not parse, this function is not implemented.".format(name))

        return lst

    def get_controller(self):
        from aimaker.controllers.controller_factory import ControllerFactory
        factory = ControllerFactory(self.setting)
        return factory.create(self.setting['base']['controller'])

    def get_norm_layer(self):
        from aimaker.models import NormalizeFactory
        factory = NormalizeFactory(self.setting)
        norm = self.setting['base']['normalize'] if self.setting is not None else "batch"
        return factory.create(norm)

    def get_visdom_graph_tags(self):
        self.viz_graph_tags = self.setting['ui']['base']['visdomGraphTags'].split(',')
        return self.viz_graph_tags

    def get_visdom_graph_x_labels(self):
        ret = {}
        for tag in self.viz_graph_tags:
            ret[tag] = "iter. num"
        return ret

    def get_visdom_graph_y_labels(self):
        return {"all": "all generator loss",
                "DA": "discliminator loss of A",
                "DB": "discliminator loss of B",
                "cycleA": "cycle loss of A",
                "cycleB": "cycle loss of B",
                "cycleA2": "cycle loss of A2",
                "cycleB2": "cycle loss of B2",
                "idtA": "idt loss of A",
                "idtB": "idt loss of B",
                'generator': 'generator',
                'discriminator': 'discriminator'}

    def get_visdom_graph_titles(self):
        return {"all": "",
                "DA": "",
                "DB": "",
                "cycleA": "",
                "cycleB": "",
                "cycleA2": "",
                "cycleB2": "",
                "idtA": "",
                "idtB": "",
                'generator': '',
                'discriminator': ''}

    def get_visdom_image_tags(self):
        self.viz_image_tags = self.setting['ui']['base']['visdomImageTags'].split(',')
        return self.viz_image_tags

    def get_visdom_image_titles(self):
        return {"realA": "realA",
                "fakeA": "fakeA",
                "realB": "realB",
                "fakeB": "fakeB",
                "cycleA": "cycleA",
                "cycleB": "cycleB",
                "cycleA2": "cycleA2",
                "cycleB2": "cycleB2",
                "idtA": "idtA",
                "idtB": "idtB",
                'real': 'real',
                'fake': 'fake',
                'target': 'target',
                'upsample_1': 'upsample 1',
                'upsample_2': 'upsample 2',
                'upsample_3': 'upsample 3',
                'residual_block_1': 'residual_block1',
                'residual_block_150': 'residual_block150'
                }

    def get_update_interval_of_graphs(self, dataset):
        value = self.setting['ui']['base']['updateIntervalOfGraphs']
        if value == 'epoch':
            return len(dataset) // self.get_batch_size()
        else:
            return int(value)

    def get_update_interval_of_images(self, dataset):
        value = self.setting['ui']['base']['updateIntervalOfImages']
        if value == 'epoch':
            return len(dataset) // self.get_batch_size()
        else:
            return int(value)

    def get_visdom_port_number(self):
        return int(self.setting['ui']['base']['visdomPortNumber'])


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool = []

    def query(self, images):
        try:
            if isinstance(images, autograd.Variable):
                images = images.data
            ret = []
            for image in images:
                image = torch.unsqueeze(image, 0)
                size = len(self.pool)
                if size < self.pool_size:
                    ret += [image]
                    self.pool += [image]
                else:
                    if np.random.uniform(0, 1) > 0.5:
                        idx = np.random.randint(0, size - 1)
                        ret += [self.pool[idx].clone()]
                        self.pool[idx] = image
                    else:
                        ret += [image]
            # return torch.cat(ret, 0)
            return autograd.Variable(torch.cat(ret, 0))
        except:
            self.pool = []
            return self.query(images)


class Calculator:
    def __init__(self, ):
        pass

    def get_output_size_of_conv(self, n, kernel, stride=1, padding=0, dilation=1):
        return np.floor((n + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

    def get_output_size_of_conv_transposed(self, n, kernel, stride=1, padding=0, dilation=1, output_padding=0):
        return (n - 1) * stride - 2 * padding + kernel + output_padding
