import importlib
import aimaker.data.datasets as datasets
from aimaker.utils import BaseFactory

class DatasetFactory(BaseFactory):
    def __init__(self, settings):
        super().__init__(settings)
        self.module_name = settings.data.base.datasetModule
