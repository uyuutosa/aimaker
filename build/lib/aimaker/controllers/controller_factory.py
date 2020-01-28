from aimaker.utils import BaseFactory
import aimaker.controllers as cont

class ControllerFactory(BaseFactory):
    def __init__(self, settings):
        super().__init__(settings)
        self.module_name = settings.base.controllerModule

    def _create(self, name):
        return eval(self.suffix+f"cont.{name}(settings=self.settings)")
