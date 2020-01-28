#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
from model_handler.model_handler import ModelHandler


class Validator:
    def __init__(self, settings):
        self.settings = settings
        self.model_root = settings['validator']['base']['modelRoot']
        self.task_name = settings['validator']['base']['taskName']
        self.m_lst = settings['validator']['base']['metrics']['names'].split(',')
        self.op_lst = settings['validator']['base']['metrics']['operators'].split(',')
        self.m_dic = dict([(x, y) for x, y in zip(self.m_lst, [dict() for x in range(len(self.m_lst))])])

        self.mh = ModelHandler(self.model_root, io_type=settings['validator']['base']['ioType'])
        self.eval_func = {'sum': sum, 'median': median}[settings['validator']['base']['evalFunc']]

    def setLosses(self, loss_dic):
        for metric, operator in zip(self.m_lst, self.op_lst):
            if metric in loss_dic:
                self.m_dic[metric]['value'] = loss_dic[metric]
                self.m_dic[metric]['operator'] = operator
            else:
                raise ValueError("{} could not be found.".format(metric))

    def upload(self):
        self.mh.save(task_name=self.settings['validator']['base']['taskName'],
                     model_path=self.settings['validator']['base']['modelPath'],
                     info_dic={'metrics': self.m_dic})

#    def uploadModelIfSOTA(self, loss_dic):
#        info_dic = self._getInfo()
#        import secrets
#        result_dic = {}
#        name = "{}".format(secrets.token_urlsafe(50))
#        order = 0
#        for key, value in info_dic.items():
#            n_win = 0
#            n_lose = 0
#            for metric in self.m_lst:
#                result = self.eval_func(self.m_dic[metric])
#                if self.metrics_inquery_dic[metric](result, value[metric]):
#                    n_win += 1
#                else:
#                    n_lose += 1
#
#            order=0
#            if n_win > n_lose:
#                order = value['order']
#
#            if order:
#                break
#
#        for key in info_dic.keys():
#            if info_dic[key]['order'] >= order:
#                info_dic[key]['order'] += 1
#
#        info_dic[name] = {}
#        for metric in self.m_lst:
#            info_dic[name][metric] = self.m_dic[metric]
#
#        info_dic[name]['order'] = order

