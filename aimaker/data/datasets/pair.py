from aimaker.data.datasets.base import BaseDataset

class DatasetOfPair(BaseDataset):
    def __init__(self, settings):
        super(DatasetOfPair, self).__init__(settings)

    def _setDataSource(self):
        self.input_path  = self.setting['data']['datasets']['pair']['inputPath']
        self.target_path = self.setting['data']['datasets']['pair']['targetPath']

        self.ds_input    = self.dsf.create(self.input_path)
        self.ds_target   = self.dsf.create(self.target_path)

    def _getInput(self, index):
        return self.ds_input[index].convert("RGB")

    def _getTarget(self, index):
        return self.ds_target[index].convert("RGB")


    def setMode(self, mode):
        self.ds_input.setMode(mode)
        self.ds_target.setMode(mode)

    def __len__(self):
        return len(self.ds_input)
