from aimaker.data.datasets.base import BaseDataset

class DatasetOfEdges(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfEdges, self).__init__(setting)

    def _setDataSource(self):
        self.ds = dsf.DataSourceFactory(self.setting)\
                     .create(self.setting['dataset settings']['edgePath'])

    def _getInput(self, index):
        im = self.ds[index]
        return im.crop((0,   0, 256, 256))

    def _getTarget(self, index):
        im = self.ds[index]
        return im.crop((256, 0, 512, 256))
