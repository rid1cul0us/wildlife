import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms as transforms
from torch.utils.data import Dataset, ConcatDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from dataset.IWildCam import IWildCam
from reweighting import Reweight


class IWildCamFeature(IWildCam):
    num_classes = 182
    num_points = 243

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
        root,
        split,
        transform=None,
        target_transform=None,
        reweighting=None,
        weights_path=None,
    ):
        super().__init__(
            "data/iwildcam",
            split,
            transform,
            target_transform,
            reweighting,
            weights_path,
        )

    def __getitem__(self, index):
        imgname = self.imgpath[index]
        img = np.load(os.path.join("data/iwildcamfeature/feature", imgname + ".npy"))
        clazz, point = self.y_class[index], self.y_point[index]
        return self.reweight_item(self, index, img, clazz, point, imgname)

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
            val = IWildCamFeature(
                base_path, "val", transform, target_transform, reweighting, weights_path
            )
            id_val = IWildCamFeature(
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
        return IWildCamFeature(
            base_path, split, transform, target_transform, reweighting, weights_path
        )


class IWildCamRandAugFeature(IWildCamFeature):
    num_classes = 182
    num_points = 243

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
        root,
        split,
        transform=None,
        target_transform=None,
        reweighting=None,
        weights_path=None,
    ):
        super().__init__(
            "data/iwildcam",
            split,
            transform,
            target_transform,
            reweighting,
            weights_path,
        )

    def __getitem__(self, index):
        imgname = self.imgpath[index]
        img = np.load(
            os.path.join("data/iwildcamfeature/feature_randaug", imgname + ".npy")
        )
        clazz, point = self.y_class[index], self.y_point[index]
        return self.reweight_item(self, index, img, clazz, point, imgname)

    def dataset_split(
        base_path,
        split,
        transform,
        target_transform,
        reweighting="NoneWeighting",
        weights_path=None,
    ):
        assert split in ["train", "id_val", "val", "all_val", "id_test", "test"]
        if split == "train":
            return IWildCamRandAugFeature(
                base_path, split, transform, target_transform, reweighting, weights_path
            )
        return IWildCamFeature.dataset_split(
            base_path, split, transform, target_transform, reweighting, weights_path
        )


if __name__ == "__main__":
    from transform import transform

    train_split = IWildCamFeature.dataset_split(
        "data/iwildcam",
        "train",
        transform=transform.tensor_transform,
        target_transform=None,
        weights_path="weights/iwildcam/train/original_loss_weight.npy",
    )
    print(len(train_split))
    s0 = train_split[0]
    print(train_split[0])

    train_split = IWildCamFeature.dataset_split(
        "data/iwildcam",
        "train",
        transform=transform.tensor_transform,
        target_transform=None,
        weights_path="weights/iwildcam/train/original_loss_weight.npy",
    )
    print(len(train_split))
    as0 = train_split[0]
    print(train_split[0])
