from aimaker.predictor.interpolation_predictor import InterpolationPredictor
from aimaker.predictor.superres_predictor      import SuperresPredictor
import glob as g


class PredictorFactory():
    def __init__(self):
        self.predictor_dic = {'interp' : InterpolationPredictor,
                              'sr'     : SuperresPredictor,
                             }

        
    def create(self, name):

        if not name in self.predictor_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                      .format(name, self.__class__.__name__,\
                                              self.predictor_dic.keys()))

        return self.predictor_dic[name]



