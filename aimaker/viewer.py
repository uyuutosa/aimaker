import sys

import torch
import torchvision.utils as vutils
import numpy as np

from aimaker.utils import SettingHandler
from aimaker.utils import TimeHandler
from visdom import Visdom
from tensorboardX import SummaryWriter


class TrainMonitor:
    def __init__(self, setting):
        self.ch = SettingHandler(setting)
        self.loss_tags = self.ch.get_visdom_graph_tags()
        self.loss_dic = self._genLossDic()
        self.th = TimeHandler()
        self._initialize()

    def _initialize(self):
        self.progress_bar = ''
        self.progress_value = 0
        self.thresh = 0
        for tag in self.loss_tags:
            self.loss_dic[tag] = []


    def flash(self):
        self._initialize()

    def _genLossDic(self):
        ret = {}
        for tag in self.loss_tags:
            ret[tag] = []
        return ret

    def setLosses(self, loss_dic):
        for tag in self.loss_tags:
            self.setLoss(tag, loss_dic[tag])

    def setLoss(self, tag, value):
        self.loss_dic[tag] += [value]

    def _updateProgress(self,
                        now_progress_value,
                        now_progress_bar,
                        current_n_iter,
                        n_iter):

        if now_progress_value > self.thresh:
            now_progress_bar += r'8'
            self.thresh += 10
        progress_value = (1 + current_n_iter) / n_iter * 100
        return progress_value, now_progress_bar


    def dumpCurrentProgress(self, n_epoch, current_n_iter, n_iter):
        self.progress_value, self.progress_bar = \
            self._updateProgress(self.progress_value,
                                 self.progress_bar,
                                 current_n_iter,
                                 n_iter)
                                 
        losses = 'losses: '
        for k, v in self.loss_dic.items():
            if v[-1] is None:
                line = "{}:{}, "
            else:
                line = "{}:{:6.3f}, "

            losses += line.format(k, v[-1])
        sys.stdout.write("\repoch {:03d}: [{}] [{:10s}] {:4.0f} % {} {:3d}/{:3d}"
                         .format(n_epoch,
                                 self.th.getElapsedTime(is_now=False),
                                 self.progress_bar,
                                 self.progress_value,
                                 losses,
                                 current_n_iter,
                                 n_iter))

    def dumpAverageLossOnEpoch(self, n_epoch):
        ave_losses = ''
        for k, v in self.loss_dic.items():
            if v[-1] is not None:
                ave_losses += \
                    "{} ave. loss: {:6.3f}, ".format(k, np.array(v).mean())
            else:
                ave_losses += "{} ave. loss: {}, ".format(k, None)

        sys.stdout.write("\repoch {:03d}: [{}] [{:10s}] {:4.0f} % {}\n"
                         .format(n_epoch,
                                 self.th.getElapsedTime(is_now=False),
                                 self.progress_bar,
                                 self.progress_value,
                                 ave_losses))


class VisdomViewer:
    def __init__(self, setting):
        self.ch = SettingHandler(setting)
        portNumber = self.ch.get_visdom_port_number()
        self.setting = setting
        self.viz = Visdom(port=portNumber)
        self.viz_image_dic = {}
        self.viz_graph_dic = {}
        self.title_dic = {}

        self.graph_taglst = self.ch.get_visdom_graph_tags()
        self.graph_xlabel_dic = self.ch.get_visdom_graph_x_labels()
        self.graph_ylabel_dic = self.ch.get_visdom_graph_y_labels()
        self.graph_title_dic = self.ch.get_visdom_graph_titles()
        self.image_taglst = self.ch.get_visdom_image_tags()
        self.image_title_dic = self.ch.get_visdom_image_titles()

    def initGraph(self, tag, xlabel=None, ylabel=None, title=None):
        self.title_dic[tag] = title
        self.viz_graph_dic[tag] = self.viz.line(np.array([[0, 0, 0]]),
                                                np.array([[np.nan, np.nan, np.nan]]),
                                                opts=dict(
                                                    xlabel=xlabel,
                                                    ylabel=ylabel)
                                                )

    def initGraphs(self):
        for tag in self.graph_taglst:
            self.initGraph(tag,
                           self.graph_xlabel_dic[tag],
                           tag,
                           tag)

    def updateGraph(self, tag, x_value, y_value,opts=None, idx=0):
        y_arr = np.array([[np.nan, np.nan, np.nan]])
        y_arr[0,idx] = y_value
        if y_value is not None:
            self.viz.line(X=np.array([[x_value]*3]),
                          Y=y_arr,
                          win=self.viz_graph_dic[tag],
                          update='append',
                          opts=opts
                          )

    def updateGraphs(self, x_value, value_dic,opts=None, idx=0):
        for tag in self.graph_taglst:
            self.updateGraph(tag, x_value, value_dic[tag], opts=opts, idx=idx)

    def initImage(self, tag, title, dummy=torch.Tensor(3, 100, 100)):
        self.title_dic[tag] = title
        self.viz_image_dic[tag] = self.viz.image(dummy, opts=dict(title=title))

    def initImages(self, dummy=torch.Tensor(3, 100, 100)):
        for tag in self.image_taglst:
            self.initImage(tag, tag, dummy=dummy)
            #self.initImage(tag, self.image_title_dic[tag], dummy=dummy)

    def updateImage(self, tag, image, title, n_iter):
        self.title_dic[tag] = title

        if self.is_cuda(image):
            image = image.cpu()

        if image is not None:
            #if 'normalize' in self.setting['dataset settings']['targetTransform']:
            if 'normalize' in self.setting['data']['base']['inputTransform']:
                image = (image + 1) / 2.0 * 255
            if image.dim() == 3:
                self.viz.image(image,
                               opts=dict(title=title),
                               win=self.viz_image_dic[tag])
            else:
                self.viz.images(image,
                                opts=dict(title=title),
                                win=self.viz_image_dic[tag])

    def updateImages(self, image_dic, n_iter):
        for tag in self.image_taglst:
            self.updateImage(tag, image_dic[tag], tag, n_iter)

    def is_cuda(self, tensor):
        return "cuda" in str(type(tensor))

    def destructVisdom(self):
        pass
       # self.popen.kill()
        
class TensorBoardXViewer:
    def __init__(self, setting):
        self.ch = SettingHandler(setting)
        self.setting = setting
        self.total_dataset_length = 0
        self.mode = "train"

        self.n_iter = 0
        self.writer_dic = {"train": SummaryWriter('runs/train'), "test": SummaryWriter('runs/test'), "valid": SummaryWriter('runs/valid')}
        self.graph_taglst = self.ch.get_visdom_graph_tags()
        self.image_taglst = self.ch.get_visdom_image_tags()

    def initGraph(self, tag, xlabel=None, ylabel=None, title=None):
        self.n_iter = 0

    def initGraphs(self):
        self.n_iter = 0

    def setMode(self, mode):
        self.mode = mode

    def setTotalDataLoaderLength(self, total_dataset_length):
        self.total_dataset_length = total_dataset_length


    def updateGraph(self, tag, x_value, y_value,opts=None, idx=0):
        self.writer_dic[self.mode].add_scalar(tag, y_value, x_value)
        #self.writer.add_scalar(tag, y_value, x_value / self.total_dataset_length)


    def updateGraphs(self, x_value, value_dic,opts=None, idx=0):
        for tag in self.graph_taglst:
            self.updateGraph(tag, x_value, value_dic[tag], opts=opts, idx=idx)

    def initImage(self, tag, title, dummy=torch.Tensor(3, 100, 100)):
        pass

    def initImages(self, dummy=torch.Tensor(3, 100, 100)):
        pass

    def updateImage(self, tag, image, title, n_iter):
        if len(image.shape) != 3:
            image = image[None]
        if 'normalize' in self.setting['data']['base']['inputTransform']:
            image = ((image + 1) / 2.0 * 255).cpu().detach().numpy().astype(np.uint8)
        self.writer_dic[self.mode].add_image(tag, image, n_iter)


    def updateImages(self, image_dic, n_iter):
        for tag in self.image_taglst:

            #if 'normalize' in self.setting['data']['base']['inputTransform']:
            #    image_dic[tag] = ((image_dic[tag] + 1) / 2.0 * 255).cpu().detach().numpy().astype(np.uint8)
            self.writer_dic[self.mode].add_image(tag, vutils.make_grid(image_dic[tag], normalize=True, scale_each=True), n_iter)
            #self.writer.add_images(tag, vutils.make_grid(image_dic[tag], normalize=True, scale_each=True), n_iter)

    def is_cuda(self, tensor):
        return "cuda" in str(type(tensor))

    def destructVisdom(self):
        pass
       # self.popen.kill()
