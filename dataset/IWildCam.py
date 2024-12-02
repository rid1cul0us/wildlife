import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms as transforms
from torch.utils.data import Dataset, ConcatDataset

from analysis.dist import dist
from reweighting import Reweight


class IWildCam(Dataset):
    num_classes = 182
    num_points = 243
    num_contexts = num_points

    # R_mean is 0.333401, G_mean is 0.341779, B_mean is 0.320772
    # R_var is 0.237052, G_var is 0.237999, B_var is 0.231040
    normMean = [0.3334, 0.3418, 0.3207]
    normStd = [0.2371, 0.2380, 0.2310]

    dataset_kwargs = {
        "num_workers": 2,
        "pin_memory": True,
    }

    def __init__(
        self,
        base_path,
        split,
        transform=None,
        target_transform=None,
        reweighting=None,
        weights_path=None,
    ):
        super().__init__()

        self.split = split
        self.base_path = base_path
        self.img_base_path = os.path.join(base_path, "train")

        self.transform = transform
        self.target_transform = target_transform

        self.y_class = []
        self.y_point = []
        self.imgpath = []
        self.globalp = []
        self.class_cnt = [0] * self.num_classes
        self.ponit_cnt = None  # TODO

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            self.target_transform = transforms.Compose([torch.Tensor])

        csv_path = os.path.join(self.base_path, f"{split}.csv")
        if not os.path.isfile(csv_path):
            metadata_filepath = os.path.join(self.base_path, "metadata.csv")
            categories_filepath = os.path.join(self.base_path, "categories.csv")
            dist(metadata_filepath, categories_filepath, self.base_path)

        # build idx_to_class, class_to_idx and y_reorder, y_original
        count_filepath = os.path.join(self.base_path, "dist.csv")
        fd = open(count_filepath)
        fd.readline()
        self.y_reorder, self.idx_to_class, self.class_to_idx = {}, {}, {}
        for idx, line in enumerate(fd.readlines()):
            _, _, y, name = line.strip().split(",")
            y = int(y)
            self.y_reorder[y] = idx
            self.class_to_idx[name] = y
            self.idx_to_class[y] = name
        self.y_original = dict(zip(self.y_reorder.values(), self.y_reorder.keys()))
        fd.close()

        # reweighting
        self.weights_path = weights_path
        if weights_path:
            self.loss_weights = np.load(self.weights_path)
        getattr(Reweight, reweighting)(self)

        metadata = pd.read_csv(os.path.join(self.base_path, "metadata.csv"))
        self.y_context = self.globalp
        if (
            hasattr(self, "loss_weights") and "train" in weights_path
        ):  # train split weights
            metadata = metadata[metadata["split"] == "train"]
            print("use train split weights")
            self.y_context = self.y_point
        remap = {
            location_remapped: y_point
            for y_point, location_remapped in enumerate(
                sorted((metadata["location_remapped"].unique()))
            )
        }

        # resolve split csv
        f = open(csv_path, "r")
        f.readline()
        for line in f:
            if self.split == "train":
                (
                    _,
                    location_remapped,
                    _,
                    _,
                    _,
                    clazz,
                    _,
                    _,
                    fname,
                    _,
                    point,
                ) = line.strip().split(",")
            else:
                (
                    _,
                    location_remapped,
                    _,
                    _,
                    _,
                    clazz,
                    _,
                    _,
                    fname,
                    _,
                ) = line.strip().split(",")
                point = location_remapped
            clazz, point = int(clazz), int(point)
            # assert clazz >= 0 and clazz < self.__class__.num_classes
            # if csv_path == 'train.csv' and not (point >= 0 and point < self.__class__.num_points):
            # print(f'caocaoocoaocaocoa {point} in {csv_path}')
            # os._exit(0)
            self.imgpath.append(fname)
            self.y_class.append(clazz)
            self.y_point.append(point)
            self.class_cnt[clazz] += 1
            # self.globalp records the global point number corresponds to loss weights for each sample
            self.globalp.append(remap[int(location_remapped)])
        f.close()

    def __getitem__(self, index):
        imgname = self.imgpath[index]
        img = Image.open(os.path.join(self.img_base_path, imgname))
        clazz, point = self.y_class[index], self.y_point[index]
        return self.reweight_item(self, index, img, clazz, point, imgname)

    def __len__(self):
        return len(self.imgpath)

    def dataset_split(
        base_path,
        split,
        transform,
        target_transform,
        reweighting="NoneWeighting",
        weights_path=None,
    ):
        assert split in ["train", "id_val", "val", "all_val", "id_test", "test"]
        if split == "all_val":
            val = IWildCam(
                base_path, "val", transform, target_transform, reweighting, weights_path
            )
            id_val = IWildCam(
                base_path,
                "id_val",
                transform,
                target_transform,
                reweighting,
                weights_path,
            )
            all_val = ConcatDataset([id_val, val])
            all_val.num_classes = val.num_classes
            all_val.idx_to_class = val.idx_to_class
            all_val.class_to_idx = val.class_to_idx
            return all_val
        return IWildCam(
            base_path, split, transform, target_transform, reweighting, weights_path
        )


if __name__ == "__main__":
    train_split = IWildCam.dataset_split(
        "data/iwildcam",
        "train",
        transform=None,
        target_transform=None,
        weights_path="weights/train/original_loss_weight.npy",
    )
    for idx, (img, target, side_info) in enumerate(train_split):
        p_target, c_index, p_index = side_info[0], side_info[1], side_info[2]
        assert p_target < 243 and p_target >= 0
    # IWildCam.dataset_split('data/iwildcam', 'id_val', None, None)
    # IWildCam.dataset_split('data/iwildcam', 'val', None, None)
    # IWildCam.dataset_split('data/iwildcam', 'all_val', None, None)
    # IWildCam.dataset_split('data/iwildcam', 'id_test', None, None)
    # IWildCam.dataset_split('data/iwildcam', 'test', None, None)
