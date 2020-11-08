from aimaker.data.datasets.base import BaseDataset

class DatasetOfSuperRes(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfSuperRes, self).__init__(setting)
        self.is_clip    = setting['dataset settings'].getboolean('isClipLowFrequency')

    def _setDataSource(self):
        self.ds = dsf.DataSourceFactory(self.setting)\
                     .create(self.setting['dataset settings']['superResPath'])

    def __getitem__(self, index):
        input  = self.ds[index].convert("RGB")
        if self.common_transform:
            input = self.common_transform(input)

        target = input.copy()


        if self.input_transform:
            input = self.input_transform(input)

        if self.target_transform:
            target = self.target_transform(target)

        if self.is_clip:
            target -= input

        return input, target
