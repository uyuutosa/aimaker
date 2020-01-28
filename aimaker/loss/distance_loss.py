import aimaker.utils.util as util
import torch.nn as nn
import torch.autograd as autograd
import aimaker.loss.base_loss as bs
import torch
import cv2
from numpy import *
import os


class DistanceLoss(bs.BaseLoss):
    def __init__(self, config):
        from aimaker.loss.loss_factory import LossFactory
        super(DistanceLoss, self).__init__(config)
        self.Tensor = self.ch.getTensor()

        self.loss = LossFactory(config).create(config['distance loss settings']['lossName'])
        self.p_lst = [[2,   3],
                      [4,   5],
                      [5,   6],
                      [4,   9],
                      [6,   7],
                      [8,   9],
                      [10, 11],
                      [12, 13],
                      [14, 15],
                      [16, 17],
                      [18, 19],
                      [20, 21],
                      [22, 40],
                      [23, 40],
                      [24, 25],
                      [26, 27],
                      [28, 29],
                      [30, 31],
                      [32, 33],
                      [34, 35],
                      [38, 40],
                      [39, 40],
                      [18, 36],
                      [19, 37]]

    def _setInputTarget(self, input, target):
        self.input, self.target = input, target
        b, c, h ,w = self.input.shape
        self.arr = ones((h, w), dtype=uint8)
        self.arr_target = ones((h, w), dtype=uint8)

    def _e(self, i, j):
        x_input_p1  = self.input[:,i:i+1].max(2)[0].argmax(2).float()
        y_input_p1  = self.input[:,i:i+1].max(3)[0].argmax(2).float()
        x_input_p2  = self.input[:,j:j+1].max(2)[0].argmax(2).float()
        y_input_p2  = self.input[:,j:j+1].max(3)[0].argmax(2).float()
        x_target_p1 = self.target[:,i:i+1].max(2)[0].argmax(2).float()
        y_target_p1 = self.target[:,i:i+1].max(3)[0].argmax(2).float()
        x_target_p2 = self.target[:,j:j+1].max(2)[0].argmax(2).float()
        y_target_p2 = self.target[:,j:j+1].max(3)[0].argmax(2).float()
        dist_input  = torch.sqrt((x_input_p1 - x_input_p2) ** 2   + (y_input_p1  - y_input_p2) ** 2)
        dist_target = torch.sqrt((x_target_p1 - x_target_p2) ** 2 + (y_target_p1 - y_target_p2) ** 2)
        #cv2.circle(self.arr, (int(x_input_p1), int(y_input_p1)), 2, 255, -1)
        #cv2.circle(self.arr, (int(x_input_p2), int(y_input_p2)), 2, 255, -1)
        #cv2.line(self.arr, (int(x_input_p2), int(y_input_p2)), (int(x_input_p1), int(y_input_p1)), 255, 4)
        #cv2.imwrite("auauau.jpg", self.arr)
        #cv2.circle(self.arr_target, (int(x_target_p1), int(y_target_p1)), 2, 255, -1)
        #cv2.circle(self.arr_target, (int(x_target_p2), int(y_target_p2)), 2, 255, -1)
        #cv2.line(self.arr_target, (int(x_target_p2), int(y_target_p2)), (int(x_target_p1), int(y_target_p1)), 255, 4)
        #cv2.imwrite("auauau_target.jpg", self.arr_target)
        
        return dist_input, dist_target

    def __call__(self, input_1, target):
        self._setInputTarget(input_1, target)
        dist_input_lst, dist_target_lst = [], []
        for i, j in self.p_lst:
            dist_input, dist_target = self._e(i, j)
            dist_input_lst += [dist_input]
            dist_target_lst += [dist_target]
       
        dist_input  = torch.cat(dist_input_lst, dim=1)
        dist_target = torch.cat(dist_target_lst, dim=1)
        ret = self.loss(dist_input, dist_target)                
        return ret 
