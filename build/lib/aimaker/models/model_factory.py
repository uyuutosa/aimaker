from aimaker.utils import BaseFactory
from aimaker.layers.initialize_factory import InitializeFactory
import aimaker

class ModelFactory(BaseFactory):
    def __init__(self, settings, is_base_sequential=True):
        super().__init__(settings)
        self.module_name = settings.models.base.modelModule
        self.is_base_sequential = is_base_sequential

    def _create(self, name):
        names = name.split("_")
        model_lst = []
        for i, name in enumerate(names):
            if name == 'adapter':
                pre_model = eval(f"aimaker.models.{names[i-1]}(settings=self.settings)")
                #post_model = model_dic[names[i+1]](settings=self.settings)
                post_model = eval(f"{names[i+1]}(settings=self.settings)")
                model = eval(f"aimaker.models.{name}(self.settings, pre_model, post_model)")
            elif 'linear' in name:
                n_out = int(name.strip('linear'))
#                pre_model = model_dic[names[i-1]](self.settings)
                pre_model = eval(f"aimaker.models.{names[i-1]}(self.settings)")
                model = eval(f"aimaker.models.{name}(self.settings, pre_model, n_out)")
            else:
                model = eval(f"aimaker.models.{name}(settings=self.settings)")

            InitializeFactory(self.settings).create(self.settings['base']['initType'])(model)
            model_lst += [model]

        if self.is_base_sequential:
            return aimaker.models.BaseSequential(self.settings, *model_lst)
        else:
            return model_lst[0]
