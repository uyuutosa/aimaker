import cv2
import numpy as np
import torch
import torch.autograd         as autograd
import torchvision.transforms as transforms

import aimaker.data.data_source_factory   as dsf
import aimaker.data.datasets.dataset_factory       as df
import aimaker.utils.image_writer_factory as iwf
import aimaker.utils.util                 as util
from   aimaker.predictor.base_predictor   import BasePredictor


class SuperresPredictor(BasePredictor):
    def __init__(self, trainer, source, divide_size=2):
        super(SuperresPredictor, self).__init__(trainer, source)
        self.divide_image = util.DivideImage(divide_size)
        self.aggregate_image = util.AggregateImage(divide_size)
        self.to_pil_image = transforms.ToPILImage()
        self.image_writer_factory = iwf.ImageWriterFactory(self.config)

    def predict(self, 
                is_view=False, 
                is_cropper=True, 
                is_return=False, 
                dump_path=None, 
                mode='default', 
                n_wait=1):

        if dump_path is not None:
            im = iwf.ImageWriterFactory(self.config).create(dump_path)
        for target in self.ds:
            input_lst = []
            pred_lst = []
            for divided_target in self.divide_image(target):
                divided_input, _    = self._transformImage(divided_target, is_cropper=is_cropper)
                input_lst += [divided_input[0]]
                pred_lst  += [(self.controller.predict(divided_input)[0] + 1) / 2]
            
            input_tensor     = self.aggregate_image(util.peelVariable(torch.cat(input_lst, 0)))
            predicted_tensor = self.aggregate_image(torch.cat(pred_lst, 0))
            predicted_tensor = self.to_pil_image(predicted_tensor)
            predicted_tensor = self.target_transform(predicted_tensor)

            _, target        = self._transformImage(target, is_cropper=is_cropper)
            input_image      = self._tensor2Image(input_tensor[None])
            target_image     = self._tensor2Image(target)
            predicted_image  = self._tensor2Image(predicted_tensor[None])
            if mode == 'concat':
                result_image = self._concatImage(input_image, predicted_image, target_image)
            elif mode == 'default':
                result_image = predicted_image
            else:
                raise ValueError('{} is wrong parameter. slash, double and default are available.'.format(mode))

            if is_view:
                if self._viewImage(result_image, "images", n_wait=n_wait):
                    break
            if dump_path is not None:
                im.dump(result_image)
            if is_return:
                return predicted_tensor

    def _transformImage(self, input, is_cropper): 
        #target = self.divide_image(util.peelVariable(input))
        if is_cropper:
            target = input = self.cropper(input)
        target = self.target_transform(target)
        input = self.input_transform(input)
        if not isinstance(input, autograd.Variable):
            input  = autograd.Variable(input[None].cuda(self.gpu_ids[0]))
            target = autograd.Variable(target[None].cuda(self.gpu_ids[0]))
        return input, target

    def _viewImage(self, image, image_name='image', n_wait=1):
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.imshow(image_name, image)
        if cv2.waitKey(n_wait) & 0xFF == ord('q'):
            return True

    def _tensor2Image(self, tensor):
        #return (util.peelVariable(tensor)[0].cpu().numpy().transpose(1,2,0).clip(0,1) * 255).astype(np.uint8)
        return (((util.peelVariable(tensor)[0].cpu().numpy().transpose(1,2,0)[:,:,::-1].clip(-1,1) + 1)/2) * 255).astype(np.uint8)

    def _concatImage(self, input_image, predict_image, target_image):
        height, width = predict_image.shape[:2]
            
        input_image_nearest        = cv2.resize(input_image,(width, height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST) 
        resized_image              = cv2.resize(input_image,(width, height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC) 
        predict_image              = cv2.resize(predict_image,(width, height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST) 
        target_image               = cv2.resize(target_image,(width, height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST) 
        images = np.zeros((height*2, width*2, 3), dtype=np.uint8)
        images[0:height, 0:width]  = input_image_nearest
        images[height:, 0:width]   = resized_image
        images[0:height, width:]   = target_image
        images[height:, width:]    = predict_image

        return images

    def predict_server(self, dump_path):
        image_writer = self.image_writer_factory.create(dump_path)
        self.ds.connectAsServer()
        self.ds.sendInfo()
        self.ds.sendData()
        self.ds.sendData()
        image_writer.connectAsClient(host)
        self.ds.sendData()
        for i in range(len(self.ds)):
            self.ds.sendData()
            image_writer.recvData()

