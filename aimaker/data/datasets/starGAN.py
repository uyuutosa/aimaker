from aimaker.data.datasets.base import BaseDataset

class DatasetOfStarGAN(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfStarGAN, self).__init__(setting)

    def _setDataSource(self):
        self.root_dir  = self.setting['dataset settings']['starGANPath']
        self.image_path = os.path.join(self.root_dir, "CelebA_nocrop")
        self.label_path = os.path.join(self.root_dir, "label.csv")
        self.type_path  = os.path.join(self.root_dir, "column_name.csv")
        self.type_name_lst = self.setting['starGAN settings']['typeNames'].strip().split(",")

        self.ds_image = self.dsf.create(self.image_path)
        self.ds_label = self.dsf.create(self.label_path)

        lst = []
        for type_name in self.type_name_lst:
            for i, type_name2 in enumerate(np.loadtxt(self.type_path, dtype="str")):
                if type_name == type_name2:
                    lst += [i]
        self.type_idx = lst

    def _getInput(self, index):
        return self.ds_image[index]

    def _getTarget(self, index):
        return self.ds_label[index][self.type_idx]

    def __getitem__(self, index):
        input   = self._getInput(index)
        target  = (self._getTarget(index) == 1).astype("float32")
        if self.common_transform:
            seed = np.random.randint(2147483647)
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

