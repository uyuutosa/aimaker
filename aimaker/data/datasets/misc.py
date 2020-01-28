
class DatasetOfBGForAutomaticAnnotation(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfBGForAutomaticAnnotation, self).__init__(setting)

    def _setDataSource(self):
        self.root_path                 = self.setting['dataset settings']['rootPath']
        self.front_images_path         = ",".join(g.glob("{}/*/img/front".format(self.root_path)))
        self.side_images_path          = ",".join(g.glob("{}/*/img/side".format(self.root_path)))
        self.properties_path           = ",".join(g.glob("{}/*/properties.csv".format(self.root_path)))
        self.gt_size_attributes        = self.setting['dataset settings']['propertiesAttributes'].split(',')
        self.input_features_attributes = self.setting['dataset settings']['inputFeaturesAttribues'].split(',')
        self.dumps_path                = ",".join(g.glob("{}/*/dump_flipped.csv".format(self.root_path)))

        self.length_attributes     = self.setting['dataset settings']['lengthAttributes'].split(',')
        self.ann_front_attributes  = self.setting['dataset settings']['annFrontAttributes'].split(',')
        self.ann_side_attributes   = self.setting['dataset settings']['annSideAttributes'].split(',')

        self.ds_front_image        = self.dsf.create(self.front_images_path)
        self.ds_side_image         = self.dsf.create(self.side_images_path)

        self.ds_gt_size            = self.dsf.create(self.properties_path, column_lst = self.gt_size_attributes)
        self.ds_input_features     = self.dsf.create(self.properties_path, column_lst = self.input_features_attributes)
        self.ds_gt_length          = self.dsf.create(self.dumps_path,      column_lst = self.length_attributes)
        self.ds_gt_ann_front       = self.dsf.create(self.dumps_path,      column_lst = self.ann_front_attributes)
        self.ds_gt_ann_side        = self.dsf.create(self.dumps_path,      column_lst = self.ann_side_attributes)

        self.width           = int(self.setting['dataset settings']['imageWidth'])
        self.height          = int(self.setting['dataset settings']['imageHeight'])
        self.open_pose_path  = self.setting['dataset settings']['openPosePath']
        self.bg_dividor      = mytransforms.BodygramDataTransform(self.width,
                                                                  self.height,
                                                                  weight_path=self.open_pose_path,
                                                                  scale=0.5,
                                                                  gpu_ids='0')

        self.num_front       = [int(x) for x in self.setting['dataset settings']['numFront'].split(',')]
        self.num_side        = [int(x) for x in self.setting['dataset settings']['numSide'].split(',')]
        self.normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    def _getInput(self, index):
        return self.ds_front_image[index],\
               self.ds_side_image[index],\
               self.ds_input_features[index]

    def _getTarget(self, index):
        return self.ds_gt_size[index],\
               self.ds_gt_ann_front[index],\
               self.ds_gt_ann_side[index]

    def _normalizeInputFeatures(self, height, weight, age, gender):
        height /= 200 * 2
        weight /= 150 * 2
        age    /= 100 * 2

        height -= 1
        weight -= 1
        age    -= 1
        gender = -1 if gender == "M" else 1
        return height, weight, age, gender

    def __getitem__(self, index):
        input_front, input_side, input_feature = self._getInput(index)
        gt_size, gt_ann_front, gt_ann_side     = self._getTarget(index)
        gt_ann_front += 500
        gt_ann_side += 500


        if self.cropper:
            seed = random.randint(0, 2147483647)
            random.seed(seed)
            input_front = self.cropper(input_front)
            random.seed(seed)
            input_side  = self.cropper(input_side)

        if self.input_transform:
            input_front = self.input_transform(input_front)
            input_side  = self.input_transform(input_side)

        # divide image and create the map pointed the location as gauusian.
        input_front_divided, gt_map_front_divided = self.bg_dividor(input_front, gt_ann_front.reshape(-1, 2), self.num_front)
        input_side_divided,  gt_map_side_divided  = self.bg_dividor(input_side,  gt_ann_side.reshape(-1, 2), self.num_side)
        self.input = input_front

        # normalize the input features.
        height, weight, age, gender = self._normalizeInputFeatures(*input_feature)

        # concatenae images and input features.
        input_front_divided  = input_front_divided.permute(0,3,1,2).reshape(-1, self.bg_dividor.height, self.bg_dividor.width)
        input_side_divided   = input_side_divided.permute(0,3,1,2).reshape(-1, self.bg_dividor.height, self.bg_dividor.width)
        gt_map_front_divided = gt_map_front_divided.reshape(-1, self.bg_dividor.height, self.bg_dividor.width)
        gt_map_side_divided  = gt_map_side_divided.reshape(-1,  self.bg_dividor.height, self.bg_dividor.width)
        #target = target / 200

        input, target = input_front_divided, gt_map_front_divided
        #input, target = input_front_divided, torch.cat([gt_map_front_divided, torch.zeros([1, self.height, self.width])])

        # normalize
        input  = self.normalize(input / 255)
        target = self.normalize(target / 255)
        return input, target
        #if self.target_transform:
        #    input  = self.target_transform(input / 255)
        #    target = self.target_transform(target / 255)
        #return input, target
        #return input_front_divided, input_side_divided, gt_map_front_divided, gt_map_side_divided
        #return input, torch.Tensor(target2)
        #return input, torch.Tensor(target1), torch.Tensor(target2)


    def setMode(self, mode):
        self.ds_front_image.setMode(mode)
        self.ds_side_image.setMode(mode)
        self.ds_input_features.setMode(mode)
        self.ds_gt_size.setMode(mode)
        self.ds_gt_ann_front.setMode(mode)
        self.ds_gt_ann_side.setMode(mode)

    def __len__(self):
        return len(self.ds_front_image)

class DatasetOfBGForAutomaticAnnotation2(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfBGForAutomaticAnnotation2, self).__init__(setting)
        self.correct_exif = mytransforms.CorrectExif()

    def _setDataSource(self):
        self.root_path                 = self.setting['dataset settings']['rootPath']
        self.front_images_path         = ",".join(g.glob("{}/*/img/front".format(self.root_path)))
        self.side_images_path          = ",".join(g.glob("{}/*/img/side".format(self.root_path)))
        self.properties_path           = ",".join(g.glob("{}/*/properties.csv".format(self.root_path)))
        self.gt_size_attributes        = self.setting['dataset settings']['propertiesAttributes'].split(',')
        self.input_features_attributes = self.setting['dataset settings']['inputFeaturesAttribues'].split(',')
        self.dumps_path                = ",".join(g.glob("{}/*/dump_flipped.csv".format(self.root_path)))

        self.length_attributes     = self.setting['dataset settings']['lengthAttributes'].split(',')
        self.ann_front_attributes  = self.setting['dataset settings']['annFrontAttributes'].split(',')
        self.ann_side_attributes   = self.setting['dataset settings']['annSideAttributes'].split(',')

        self.ds_front_image        = self.dsf.create(self.front_images_path)
        self.ds_side_image         = self.dsf.create(self.side_images_path)

        self.ds_gt_size            = self.dsf.create(self.properties_path, column_lst = self.gt_size_attributes)
        self.ds_input_features     = self.dsf.create(self.properties_path, column_lst = self.input_features_attributes)
        self.ds_gt_length          = self.dsf.create(self.dumps_path,      column_lst = self.length_attributes)
        self.ds_gt_ann_front       = self.dsf.create(self.dumps_path,      column_lst = self.ann_front_attributes)
        self.ds_gt_ann_side        = self.dsf.create(self.dumps_path,      column_lst = self.ann_side_attributes)

        self.width                            = int(self.setting['dataset settings']['imageWidth'])
        self.height                           = int(self.setting['dataset settings']['imageHeight'])
        self.open_pose_path                   = self.setting['dataset settings']['openPosePath']
        self.map_generator = mytransforms.MapGenerator()

        self.num_front       = [int(x) for x in self.setting['dataset settings']['numFront'].split(',')]
        self.num_side        = [int(x) for x in self.setting['dataset settings']['numSide'].split(',')]
        self.normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    def _getInput(self, index):
        return self.ds_front_image[index],\
               self.ds_side_image[index],\
               self.ds_input_features[index]

    def _getTarget(self, index):
        return self.ds_gt_size[index],\
               self.ds_gt_ann_front[index],\
               self.ds_gt_ann_side[index]

    def _normalizeInputFeatures(self, height, weight, age, gender):
        height /= 200 * 2
        weight /= 150 * 2
        age    /= 100 * 2

        height -= 1
        weight -= 1
        age    -= 1
        gender = -1 if gender == "M" else 1
        return height, weight, age, gender

    def _apply(self, lst, func, seed=None):
        if seed is None:
            return [func(x) for x in lst]
        else:
            ret_lst = []
            for i in lst:
                random.seed(seed)
                ret_lst += [func(i)]
            return ret_lst

    def __getitem__(self, index):
        input_front, input_side, input_feature = self._getInput(index)
        input_front = self.correct_exif(input_front)
        input_side  = self.correct_exif(input_side)

        gt_size, gt_ann_front, gt_ann_side     = self._getTarget(index)
        #gt_ann_front += 500
        #gt_ann_side += 500

        gt_map_front_lst = self.map_generator(input_front, gt_ann_front.reshape(-1, 2))
        gt_map_side_lst  = self.map_generator(input_side,  gt_ann_side.reshape(-1, 2))

        #if self.cropper:
        #    seed = random.randint(0, 2147483647)
        #    random.seed(seed)
        #    input_front = self.cropper(input_front)
        #    random.seed(seed)
        #    input_side  = self.cropper(input_side)
        #    gt_map_front_lst = self._apply(gt_map_front_lst, self.cropper, seed)
        #    gt_map_side_lst  = self._apply(gt_map_side_lst,  self.cropper, seed)

        #if self.input_transform:
        #    input_front = self.input_transform(input_front)
        #    input_side  = self.input_transform(input_side)

        #if self.target_transform:
        #    gt_map_front_lst = self._apply(gt_map_front_lst, self.target_transform)
        #    gt_map_side_lst  = self._apply(gt_map_side_lst,  self.target_transform)

        # normalize the input features.
        #height, weight, age, gender = self._normalizeInputFeatures(*input_feature)

        return input_front, input_side, gt_map_front_lst, gt_map_side_lst


    def setMode(self, mode):
        self.ds_front_image.setMode(mode)
        self.ds_side_image.setMode(mode)
        self.ds_input_features.setMode(mode)
        self.ds_gt_size.setMode(mode)
        self.ds_gt_ann_front.setMode(mode)
        self.ds_gt_ann_side.setMode(mode)

    def __len__(self):
        return len(self.ds_front_image)

class DatasetOfBGForAutomaticAnnotation3(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfBGForAutomaticAnnotation3, self).__init__(setting)
        self.correct_exif = mytransforms.CorrectExif()

    def _setDataSource(self):
        self.root_path        = self.setting['dataset settings']['rootPath']
        self.front_indices    = self.setting['dataset settings']['frontIndices']
        self.front_image_path = ",".join(g.glob("{}/*/img/front".format(self.root_path)))
        self.front_map_path   = ",".join(g.glob("{}/*/map/front".format(self.root_path)))

        self.ds_front_image        = self.dsf.create(self.front_image_path)
        self.ds_front_map          = self.dsf.create(self.front_map_path, indices=self.front_indices)

    def _getInput(self, index):
        return self.ds_front_image[index]

    def _getTarget(self, index):
        return self.ds_front_map[index]

    def setMode(self, mode):
        self.ds_front_image.setMode(mode)
        self.ds_front_map.setMode(mode)

    def __len__(self):
        return len(self.ds_front_image)

class DatasetOfBGForSideImages(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfBGForSideImages, self).__init__(setting)
        self.correct_exif = mytransforms.CorrectExif()

    def _setDataSource(self):
        self.root_path     = self.setting['dataset settings']['rootPath']
        self.side_image_path       = ",".join(g.glob("{}/*/img/side".format(self.root_path)))
        self.side_map_path         = ",".join(g.glob("{}/*/map/side".format(self.root_path)))
        self.ds_side_image = self.dsf.create(self.side_image_path)
        self.ds_side_map   = self.dsf.create(self.side_map_path)

    def _getInput(self, index):
        return self.ds_side_image[index]

    def _getTarget(self, index):
        return self.ds_side_map[index]

    def setMode(self, mode):
        self.ds_side_image.setMode(mode)
        self.ds_side_map.setMode(mode)

    def __len__(self):
        return len(self.ds_side_image)

class DatasetOfBGForSupervisely(BaseDataset):
    def __init__(self, setting):
        super(DatasetOfBGForAutomaticAnnotation3, self).__init__(setting)
        self.correct_exif = mytransforms.CorrectExif()

    def _setDataSource(self):
        self.root_path              = self.setting['dataset settings']['rootPath']
        self.front_image_path       = ",".join(g.glob("{}/*/img/front".format(self.root_path)))
        self.front_map_path         = ",".join(g.glob("{}/*/map/front".format(self.root_path)))

        self.ds_front_image        = self.dsf.create(self.front_image_path)
        self.ds_front_map          = self.dsf.create(self.front_map_path)


    def _getInput(self, index):
        return self.ds_front_image[index]

    def _getTarget(self, index):
        return self.ds_front_map[index]

    def setMode(self, mode):
        self.ds_front_image.setMode(mode)
        self.ds_front_map.setMode(mode)

    def __len__(self):
        return len(self.ds_front_image)
#class DatasetOfBGForAutomaticAnnotation(BaseDataset):
#    def __init__(self, setting):
#        super(DatasetOfBGForAutomaticAnnotation, self).__init__(setting)
#
#    def _setDataSource(self):
#        self.root_path         = self.setting['dataset settings']['rootPath']
#        self.front_images_path = ",".join(g.glob("{}/*/img/front".format(self.root_path)))
#        self.side_images_path  = ",".join(g.glob("{}/*/img/side".format(self.root_path)))
#        self.properties_path   = ",".join(g.glob("{}/*/properties.csv".format(self.root_path)))
#        self.properties_attributes = self.setting['dataset settings']['propertiesAttributes'].split(',')
#        self.input_feature     = self.setting['dataset settings']['inputFeatures'].split(',')
#        self.dumps_path        = ",".join(g.glob("{}/*/dump_flipped.csv".format(self.root_path)))
#        self.dumps_attributes  = self.setting['dataset settings']['dumpsAttributes'].split(',')
#
#        self.dumps_attributes  = self.setting['dataset settings']['dumpsAttributes'].split(',')
#
#        self.ds_front_image   = self.dsf.create(self.front_images_path)
#        self.ds_side_image    = self.dsf.create(self.side_images_path)
#
#        self.ds_properties    = self.dsf.create(self.properties_path, column_lst = self.properties_attributes)
#        self.ds_input_feature = self.dsf.create(self.properties_path, column_lst = self.input_feature)
#        self.ds_input_feature = self.dsf.create(self.properties_path, column_lst = self.input_feature)
#        self.ds_dumps         = self.dsf.create(self.dumps_path, column_lst = self.dumps_attributes)
#        #self.ds_label = DataSourceFromCSV(self.label_path,
#        #                                  self.label_attributes,
#        #                                  self.port,
#        #                                  self.train_test_ratio)
#
#    def _getInput(self, index):
#        return self.ds_front_image[index],\
#               self.ds_side_image[index],\
#               self.ds_input_feature[index]
#
#    def _getTarget(self, index):
#        return self.ds_properties[index],\
#               self.ds_dumps[index],
#
#    def __getitem__(self, index):
#        input_front, input_side, input_feature = self._getInput(index)
#        target1, target2  = self._getTarget(index)
#
#        if self.cropper:
#            seed = np.random.randint(2147483647)
#            np.random.seed(seed)
#            input_front = self.cropper(input_front)
#            input_side  = self.cropper(input_side)
#
#        if self.input_transform:
#            input_front = self.input_transform(input_front)
#            input_side  = self.input_transform(input_side)
#
#        #if self.target_transform:
#        #    target = self.target_transform(target)
#
#        height, weight, age, gender = input_feature
#        height /= 200 * 2
#        weight /= 150 * 2
#        age    /= 100 * 2
#
#        height -= 1
#        weight -= 1
#        age    -= 1
#        gender = -1 if gender == "M" else 1
#        o = torch.ones((1, *input_front.shape[1:]))
#
#        input = torch.cat([input_front, input_side, height * o , weight * o, age * o, gender * o], 0)
#        #target = target / 200
#
#        return input, torch.Tensor(target1)
#        #return input, torch.Tensor(target2)
#        #return input, torch.Tensor(target1), torch.Tensor(target2)
#
#    def setMode(self, mode):
#        self.ds_front_image.setMode(mode)
#        self.ds_side_image.setMode(mode)
#        self.ds_dumps.setMode(mode)
#        self.ds_properties.setMode(mode)
#
#    def __len__(self):
#        return len(self.ds_front_image)
