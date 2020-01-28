import aimaker.data.datasets.base.BaseDataset as BaseDataset
from aimaker.data.datasets.base import BaseDataset

class DatasetOfImageToCSV(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfStarGAN, self).__init__(setting)

    def _setDataSource(self):
        self.image_path       = self.setting['dataset settings']['ImagePath']
        self.label_path       = self.setting['dataset settings']['LabelPath']
        self.label_attributes = self.setting['dataset settings']['CSVLabelAttributes']

        self.ds_image = self.dsf.create(self.image_path)

        self.ds_label = DataSourceFromCSV(self.csv_label_path,
                                    self.label_atrributes,
                                    self.port,
                                    self.train_test_ratio)

    def _getInput(self, index):
        return self.ds_image[index]

    def _getTarget(self, index):
        return self.ds_label[index]

    def __getitem__(self, index):
        input   = self._getInput(index)
        target  = self._getTarget(index)

        if self.common_transform:
            seed = np.random.randint(0, 2147483647)
            np.random.seed(seed)
            input  = self.common_transform(input)

        if self.input_transform:
            input = self.input_transform(input)

        return input, target

    def setMode(self, mode):
        self.ds_image.setMode(mode)
        self.ds_label.setMode(mode)

    def __len__(self):
        return len(self.ds_image)

class DatasetOfCSV(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfStarGAN, self).__init__(setting)

    def _setDataSource(self):
        self.csv_data_path    = self.setting['dataset settings']['CSVDataPath']
        self.csv_label_path   = self.setting['dataset settings']['CSVLabelPath']
        self.data_attributes  = self.setting['dataset settings']['CSVDataAttributes']
        self.label_attributes = self.setting['dataset settings']['CSVLabelAttributes']

        self.ds_data = DataSourceFromCSV(self.csv_data_path,
                                    self.data_atrributes,
                                    self.port,
                                    self.train_test_ratio)

        self.ds_label = DataSourceFromCSV(self.csv_label_path,
                                    self.label_attributes,
                                    self.port,
                                    self.train_test_ratio)

    def _getInput(self, index):
        return self.ds_image[index]

    def _getTarget(self, index):
        return self.ds_label[index]

    def __getitem__(self, index):
        input   = self._getInput(index)
        target  = self._getTarget(index)

        return input, target

    def setMode(self, mode):
        self.ds_image.setMode(mode)
        self.ds_label.setMode(mode)


    def __len__(self):
        return len(self.ds_data)
