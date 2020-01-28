#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys 
import os

import aimaker.viewer as viewer
from aimaker.utils import load_setting, SettingHandler
from aimaker.validators import Validator
from aimaker.data.datasets import DatasetFactory
from easydict import EasyDict


class Trainer:
    def __init__(self, setting_path="settings"):
        self.settings = settings = EasyDict(load_setting(setting_path))
        self.sh = SettingHandler(settings)
        self.controller = self.sh.getController()
        self.dataset = DatasetFactory(settings).create(settings.data.base.datasetName)
        self.valid_dataset = None
        if settings.data.base.isValid:
            self.valid_dataset = DatasetFactory(settings).create(settings['data']['base']['valid']['datasetName'])
        self.data_loader = self.dataset.getDataLoader()
        if settings.data.base.isValid:
            self.valid_data_loader = self.valid_dataset.getDataLoader()

        self.sheckpoint_dir = self.sh.getCheckPointsDir()
        self.viz = viewer.TensorBoardXViewer(settings)
        self.train_monitor = viewer.TrainMonitor(settings)

        self.n_update_graphs = self.sh.getUpdateIntervalOfGraphs(self.dataset)
        self.n_update_images = self.sh.getUpdateIntervalOfImages(self.dataset)
        if settings.data.base.isValid:
            self.validator = Validator(settings)
        self.idx_dic = EasyDict({'train': 0, 'test': 1, 'valid': 2})

    def _getInfo(self):
        info = EasyDict()
        info.current_epoch = 0
        info.train = EasyDict({"v_iter": 0, "current_n_iter": 0})
        info.test = EasyDict({"v_iter": 0, "current_n_iter": 0})
        info.valid = EasyDict({"v_iter": 0, "current_n_iter": 0})
        return info

    def train(self):
        n_epoch = self.settings['base']['nEpoch']
        if self.settings['base']['isView']:
            self.viz.initGraphs()
            self.viz.initImages()
        
        train_n_iter = len(self.data_loader)
        if self.settings.data.base.isValid:
            valid_n_iter = len(self.valid_data_loader)

        if os.path.exists(self.settings.base.infoPath):
            info = EasyDict(json.load(open(self.settings.base.infoPath)))
        else:
            info = self._getInfo()

        try:
            model_save_interval = self.sh.getModelSaveInterval()
            if self.settings.data.base.isValid:
                model_save_interval_valid = self.sh.getModelSaveIntervalForValid()
            for current_epoch in range(info.current_epoch, n_epoch):
                info.current_epoch = current_epoch
                if self.settings['data']['base']['isTrain']:
                    info.mode = "train"
                    print('{}:'.format('train'))
                    self.dataset.setMode('train')
                    self.dataset.getTransforms()
                    self.viz.setMode('train')
                    self.data_loader = self.dataset.getDataLoader()
                    self.controller.setMode('train')
                    info = self._learning('train', info, current_epoch, self.data_loader, train_n_iter)
                if self.settings['data']['base']['isTest']:
                    info.mode = "test"
                    print('{}:'.format('test'))
                    self.dataset.setMode('test')
                    self.dataset.getTransforms()
                    self.viz.setMode('test')
                    self.data_loader = self.dataset.getDataLoader()
                    self.controller.setMode('test')
                    info = self._learning('test', info, current_epoch, self.data_loader, train_n_iter)
                if self.valid_dataset is not None:
                    if self.settings['data']['base']['isValid']:
                        info.mode = "valid"
                        print('{}:'.format('valid'))
                        self.valid_dataset.setMode('valid')
                        self.valid_dataset.getTransforms()
                        self.viz.setMode('valid')
                        self.valid_data_loader = self.valid_dataset.getDataLoader()
                        self.controller.setMode('valid')
                        info = self._learning('valid', info, current_epoch, self.valid_data_loader, valid_n_iter)
                        if current_epoch != 0 and not current_epoch % model_save_interval_valid:
                            self.controller._saveModel(self.controller.getModel(), self.settings['validator']['base']['modelPath'], is_fcnt=False)
                            self.validator.upload()#self.settings['valid_data']['data']['base']['datasetName'])

                    
                if not current_epoch % model_save_interval:
                    self.controller.saveModels()
        except:
            import traceback
            traceback.print_exc()
        self.controller.saveModels()
        if self.settings['base']['isView']:
            self.viz.destructVisdom()

    def _saveInfo(self, info):
        with open(self.settings.base.infoPath, 'w') as fp:
            json.dump(info, fp)

    def _learning(self, mode, info, current_epoch, data_loader, train_n_iter):
        n_iter = len(data_loader)
        ratio = train_n_iter / n_iter
        v_iter = info[mode].v_iter
        self.viz.setTotalDataLoaderLength(len(self.data_loader))
        is_volatile = False if mode == 'train' else True
        for current_n_iter, data in enumerate(data_loader):
            if current_n_iter < info[mode].current_n_iter:
                continue
            info[mode].current_n_iter = current_n_iter
            info[mode].v_iter = v_iter
            try:
                self.controller.setInput(data, is_volatile)
                self.controller.forward()
                if mode == 'train':
                    self.controller.backward()
                loss_dic = self.controller.getLosses()
                if mode == 'valid':
                    self.validator.setLosses(loss_dic)
                self.train_monitor.setLosses(loss_dic)
                self.train_monitor.dumpCurrentProgress(current_epoch, current_n_iter, n_iter)
            
                if not current_n_iter % self.n_update_graphs:
                    if self.settings['base']['isView']:
                        self.viz.updateGraphs(ratio * v_iter, loss_dic, idx=self.idx_dic[mode])
                if not current_n_iter % self.n_update_images:
                    if self.settings['base']['isView']:
                        self.viz.updateImages(self.controller.getImages(), current_n_iter)
            except KeyboardInterrupt:
                self._saveInfo(info)
                sys.exit()
                break
            except FileNotFoundError:
                import traceback
                traceback.print_exc()
            except:
                import traceback
                traceback.print_exc()
                break
            v_iter += 1
        #if mode == 'valid':
        #    self.validator.uploadModelIfSOTA(current_epoch)
        info[mode].v_iter = v_iter
        info[mode].current_n_iter = 0 # reset
        self.train_monitor.dumpAverageLossOnEpoch(current_epoch)
        self.train_monitor.flash()
        return info
