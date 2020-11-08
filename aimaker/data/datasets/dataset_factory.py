import importlib
from aimaker.data.datasets.pair import DatasetOfPair
#import aimaker.data.datasets as datasets
from aimaker.utils import BaseFactory
import aimaker.data.datasets as da

class DatasetFactory(BaseFactory):
    def __init__(self, settings):
        super().__init__(settings)
        self.module_name = settings.data.base.datasetModule

    def _create(self, name):
        return eval(self.suffix+f"da.{name}(settings=self.settings)")
