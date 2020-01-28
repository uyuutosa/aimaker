from aimaker.data.datasets.base import BaseDataset

class DatasetOfInterpolation(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfInterpolation, self).__init__(setting)
        self.offset_input  = int(setting['dataset settings']['offsetInput'])
        self.offset_target = int(setting['dataset settings']['offsetTarget'])


        self.ds = dsf.DataSourceFactory(self.setting)\
                     .create(self.setting['dataset settings']['interpPath'])

    def _getInput(self, index):
        input1 = self.ds[index]
        input2 = self.ds[index + self.offset_input]
        return input1, input2

    def _getTarget(self, index):
        return self.ds[index + self.offset_target]

    def __getitem__(self, index):
        input1, input2 = self._getInput(index)
        target         = self._getTarget(index)
        if self.common_transform:
            seed = random.randint(2147483647)
            np.random.seed(seed)
            input1 = self.common_transform(input1)
            np.random.seed(seed)
            input2 = self.common_transform(input2)
            np.random.seed(seed)
            target = self.common_transform(target)

        if self.input_transform:
            input1 = self.input_transform(input1)
            input2 = self.input_transform(input2)
        if self.target_transform:
            target = self.target_transform(target)
        if self.setting['dataset settings'].getboolean('isDualOutput'):
            target = torch.cat((target, target), 0)

        return torch.cat((input1, input2), 0), target

    def __len__(self):
        offset = self.offset_input if self.offset_input > self.offset_target\
                 else self.offset_target

        return len(self.ds) - offset
