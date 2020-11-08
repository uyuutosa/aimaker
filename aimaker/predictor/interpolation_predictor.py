import aimaker.data.data_source_factory as dsf
import aimaker.data.datasets.dataset_factory   as df
import aimaker.utils.image_writer_factory as iwf
import cv2
import numpy as np
import torch.autograd as autograd
from aimaker.predictor.base_predictor import BasePredictor
import aimaker.utils.util as util
import torch
import torchvision.transforms as transforms
import aimaker.data.data_source as data_source
import shutil
import os


class InterpolationPredictor(BasePredictor):
    def __init__(self, trainer, source, divide_size=2):
        super(InterpolationPredictor, self).__init__(trainer, source)
        self.divide_size = divide_size
        self.divide_image = util.DivideImage(divide_size)
        self.aggregate_image = util.AggregateImage(divide_size)
        self.to_pil_image = transforms.ToPILImage()
        self.fps_mult = int(self.config['image writer settings']['fpsMult'])
        self.image_writer_factory = iwf.ImageWriterFactory(self.config)
        self._initialize()

    def _initialize(self):
        c, self.height, self.width = self.trainer.dataset.input_transform(self.ds[0]).shape
        self.width  *= self.divide_size
        self.height *= self.divide_size


    def predict(self, dump_path, is_cropper=True, fps=None, size=None, mode='default'):

        if isinstance(self.ds, data_source.DataSourceFromSocket):
            dump_path = self.source

        if fps is None:
            if hasattr(self.ds, "fps"):
                fps = self.ds.fps 
            else:
                fps = int(self.config['image writer settings']['fps'])

        if size is not None:
            self.width, self.height = size


        if mode == 'slash' or\
           mode == 'default':
            image_writer = self.image_writer_factory.create(dump_path, fps * self.fps_mult, (self.width, self.height))
        else:
            image_writer = self.image_writer_factory.create(dump_path, fps * self.fps_mult, (self.width * 2, self.height))

        if isinstance(self.ds, data_source.DataSourceFromSocket):
            image_writer.sendInfo()

        cv2.namedWindow("out", cv2.WINDOW_NORMAL)
        try:
            
            for i in range(0, len(self.ds)-1, 1):
                input_lst = []
                pred_lst = []
                if not i:
                    input1 = self.ds[i]
                else:
                    input1 = input2
                input2 = self.ds[i+1]
                if is_cropper:
                    input1 = self.cropper(input1)
                    input2 = self.cropper(input2)

                for divided_input1, divided_input2 in zip(self.divide_image(input1), self.divide_image(input2)):
                    divided_input1, divided_input2 = self._transformImage(divided_input1, divided_input2)
                    divided_input  = torch.cat((divided_input1, divided_input2), 1)
                    input_lst  += [util.peelVariable(divided_input1)[0]]
                    pred_lst   += [util.peelVariable(self.controller.predict(divided_input)[0])]

                predicted_tensor = self.aggregate_image(torch.cat(pred_lst, 0))
                input_tensor = self.aggregate_image(torch.cat(input_lst, 0))

                interpolated_image = predicted_tensor.cpu().numpy().transpose(1,2,0)
                interpolated_image = (((interpolated_image + 1)/2).clip(0,1) * 255)\
                                     .astype(np.uint8)[...,:3][...,::-1]

                input_arr1 = ((((input_tensor.numpy().transpose(1,2,0) + 1) /2)\
                            .clip(0, 1)) * 255).astype(np.uint8)
                pre_image                   = input_arr1[...,::-1]
                pre_image                   = cv2.resize(pre_image,(self.width, self.height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC) 
                interpolated_image          = cv2.resize(interpolated_image,(self.width, self.height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC) 

                if mode == 'slash':
                    pre_images = pre_image.copy()
                    pre_and_interpolated_images = interpolated_image.copy()

                    mask_for_separate   = np.tri(self.height, self.width)[::-1,:,None]
                    mask_for_white_line = np.eye(self.height, self.width)[::-1,:,None]
                    pre_and_interpolated_images = np.where(mask_for_separate  , pre_images, pre_and_interpolated_images) 
                    pre_images                  = np.where(mask_for_white_line, (255 * np.ones((self.height,self.width,3))).astype(np.uint8), pre_images) 
                    pre_and_interpolated_images = np.where(mask_for_white_line, (255 * np.ones((self.height,self.width,3))).astype(np.uint8), pre_and_interpolated_images) 
                elif mode == 'double':
                    pre_images                  = np.concatenate((pre_image, pre_image), 1)
                    pre_and_interpolated_images = np.concatenate((pre_image, interpolated_image), 1)
                    cv2.putText(pre_images, 'RAW Movie', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
                    cv2.putText(pre_images, 'Processed Movie', (10 + self.width, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                    cv2.putText(pre_and_interpolated_images, 'RAW Movie', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
                    cv2.putText(pre_and_interpolated_images, 'Processed Movie', (10 + self.width, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                elif mode == 'default':
                    pre_images = pre_image.copy()
                    pre_and_interpolated_images = interpolated_image.copy()
                else:
                    raise ValueError('{} is wrong parameter. slash, double and default are available.'.format(mode))

                #out.write(pre_images)
                #out.write(pre_and_interpolated_images)

                #print("write data")
                image_writer.dump(pre_images)
                image_writer.dump(pre_and_interpolated_images)
                #print("write data done")
                cv2.imshow("out", pre_images)
                cv2.waitKey(1)
                cv2.imshow("out", pre_and_interpolated_images)
                cv2.waitKey(1)


        except:
            import traceback
            traceback.print_exc()
            self.controller.save_models()

        #self.ds.sock.close()
        image_writer.release()
        cv2.destroyAllWindows()

    def predict_recursive(self, rec_num, dump_path, tmp_path='tmpdir', is_cropper=True, fps_mult=2):
        swap_path = tmp_path + "_swp"
        for n in range(rec_num-1): 
            self.predict(tmp_path, is_cropper=is_cropper)
            self.ds = dsf.DataSourceFactory(self.config).create(tmp_path)
            self._initialize()
            # now tmp_path emarge error in case of mp4. under inspection.
            fps = self._getFps()

            tmp_path, swap_path = swap_path, tmp_path
            if n:
                shutil.rmtree(tmp_path)

        self.predict(dump_path, is_cropper=is_cropper)
        if os.path.exists(swap_path):
            shutil.rmtree(swap_path)

    def _transformImage(self, input1, input2):
        input1 = self.input_transform(input1)
        input2 = self.input_transform(input2)
        if not isinstance(input1, autograd.Variable):
            input1  = autograd.Variable(input1[None].cuda(self.gpu_ids[0]))
            input2  = autograd.Variable(input2[None].cuda(self.gpu_ids[0]))
        return input1, input2

    def _getFps(self):
        if hasattr(self.ds, "fps"):
            fps = self.ds.fps
        else:
            fps = int(self.config['image writer settings']['fps']) 
        return fps

    def predict_server(self, dump_path):
        fps = self._getFps()
        image_writer = self.image_writer_factory.create(dump_path)
        image_writer.recvInfo()
        #image_writer.initialize()
        #print(self.ds)
        #print(len(self.ds))
        self.ds.connectAsServer()
        self.ds.sendInfo()
        self.ds.sendData()
        self.ds.sendData()
        image_writer.connectAsServer()
        #image_writer.connectAsClient(host)
        self.ds.sendData()
        try:
            for i in range(len(self.ds)-1):
                self.ds.sendData()
                image_writer.recvData()
                image_writer.recvData()
        except:
            import traceback
            traceback.print_exc()
        image_writer.release()
        print("hello")

