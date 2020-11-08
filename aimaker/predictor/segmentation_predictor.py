import cv2
import numpy as np

import chainer
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.utils import read_image
from aimaker.predictor.pspnet import PSPNet
import PIL.Image as I


class SegmentationPredictor():
    def __init__(self, model='voc2012', gpu_ids='', port=1234):
        self.gpu_ids = gpu_ids
        self.to_pil_image = transforms.ToPILImage()
        self.model = self._loadModel(model)

    def _loadModel(self, model):
        if model == 'voc2012':
            model = PSPNet(pretrained_model='voc2012')
            #self.labels = voc_semantic_segmentation_label_names
            #self.colors = voc_semantic_segmentation_label_colors
        elif model == 'cityscapes':
            model = PSPNet(pretrained_model='cityscapes')
            #self.labels = cityscapes_label_names
            #self.colors = cityscapes_label_colors
        elif model == 'ade20k':
            model = PSPNet(pretrained_model='ade20k')
            #self.labels = ade20k_label_names
            #self.colors = ade20k_label_colors

        if len(self.gpu_ids):
            chainer.cuda.get_device_from_id(int(self.gpu_ids[0])).use()
            model.to_gpu(int(self.gpu_ids[0]))

        return model
        
    def predict(self, img, resize_ratio=(1, 1)):
        c, h, w = img.shape
        
        #print(int(w * resize_ratio[0]), int(h * resize_ratio[1]))
        resized_img = cv2.resize(img.transpose(1,2,0), (int(w * resize_ratio[0]), int(h * resize_ratio[1]))).transpose(2,0,1)
        img_cv = img.transpose(1,2,0) * 1.
        #print("resize is {}".format(resized_img.shape))
        
        pred = self.model.predict([resized_img.astype(np.float32)])[0]
        pred = cv2.resize(pred.astype(np.uint8), (w, h))
        
        ret =  np.where(pred[...,None] * np.ones(3)== 15, img_cv, np.zeros(3)).astype(np.uint8)
        
        return ret.transpose(2,0,1)


