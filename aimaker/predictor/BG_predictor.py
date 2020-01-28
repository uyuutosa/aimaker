import cv2
import numpy as np
import torch
import PIL.Image as I


from   aimaker.predictor.base_predictor   import BasePredictor


class BGPredictor(BasePredictor):
    def __init__(self, config_path):
        super(BGPredictor, self).__init__(config_path)

    def predict(self, front_image, side_image):
        front_raw = I.open(front_image)
        side_raw = I.open(side_image)
        front = self.input_transform(front_raw)
        side = self.input_transform(side_raw)

        result = self.controller.predict(torch.cat([front, side], 0)[None])
        result = np.array(result.data.cpu())[0].flatten()

        front = (front.cpu().numpy().transpose(1,2,0) + 1) / 2.0 * 255
        side  = (side.cpu().numpy().transpose(1,2,0)  + 1) / 2.0 * 255

        pointed_front    = front.copy()
        pointed_front_gt = front.copy()
        pointed_side     = side.copy()
        pointed_side_gt  = side.copy()

        i = 0
        while i < 128:
            x, y = (int(result[i]), int(result[i+1]))
            x -= (3024-4000) / 2
            y -= (4032-4000) / 2
            x = int(x / 4000 * 512)
            y = int(y / 4000 * 512)

            if i < 100:
            	cv2.circle(pointed_front, (x, y), 4, (255, 0, 0), -1)
            else:
                cv2.circle(pointed_side, (x, y), 4, (255, 0, 0), -1)
            i += 2

        return result, pointed_front, pointed_side

class BGPredictor2(BasePredictor):
    def __init__(self, config_path):
        super(BGPredictor2, self).__init__(config_path)

    def predict(self, front_image, side_image, height, weight, age, gender):
        front_raw = I.open(front_image)
        side_raw = I.open(side_image)
        input_front = self.input_transform(front_raw)
        input_side = self.input_transform(side_raw)

        height /= 200
        weight /= 150
        age    /= 100
        height -= 1
        weight -= 1
        age    -= 1
        #print(height, weight, age)
        gender = -1 if gender == "M" else 1
        o = torch.ones((1, *input_front.shape[1:]))

        input = torch.cat([input_front, input_side, height * o , weight * o, age * o, gender * o], 0)
        #print(input.shape)
        result = self.controller.predict(input[None])
        #print(result)
        result = np.array(result.data.cpu())[0].flatten()

        return result
