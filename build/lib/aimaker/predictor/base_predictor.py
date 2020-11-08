import data_source.data_source_factory as dsf
import aimaker.utils.util as util
import configparser 

class BasePredictor:
    def __init__(self, config_path="setting.ini"):
        config = configparser.ConfigParser()
        config.read(config_path)
        self.config = config

        self.ch = util.ConfigHandler(config)

        self.controller  = self.ch.get_controller()
        self.input_transform  = self.ch.get_input_transform()


    def predict(self, dump_path):
        pass

    def predict_server(self, dump_path, host):
        pass
