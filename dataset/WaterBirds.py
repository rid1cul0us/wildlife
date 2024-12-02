import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms as transforms
from torch.utils.data import Dataset, ConcatDataset

from reweighting import Reweight


class WaterBirds(Dataset):
    num_classes = 200
    num_contexts = 2
    num_points = num_contexts

    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]

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
        self.img_base_path = base_path

        self.transform = transform
        self.target_transform = target_transform

        self.y_class = []
        self.y_context = []
        self.imgpath = []
        self.class_cnt = [0] * self.num_classes
        self.context_cnt = [0] * self.num_contexts

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            self.target_transform = transforms.Compose([torch.Tensor])

        metadata_filepath = os.path.join(self.base_path, "metadata.csv")
        metadata = pd.read_csv(metadata_filepath)
        metadata = metadata[
            metadata["split"] == {"train": 0, "val": 1, "test": 2}[split]
        ]
        for item in metadata.itertuples():
            _, _, imgpath, y, _, context, _ = item
            self.imgpath.append(imgpath)
            self.y_class.append(y)
            self.y_context.append(context)
            self.class_cnt[y] += 1
            self.context_cnt[context] += 1

        # reweighting
        self.weights_path = weights_path
        if weights_path:
            self.loss_weights = np.load(self.weights_path)
        getattr(Reweight, reweighting)(self)

    def __getitem__(self, index):
        imgname = self.imgpath[index]
        img = Image.open(os.path.join(self.img_base_path, imgname))
        clazz, context = self.y_class[index], self.y_context[index]
        return self.reweight_item(self, index, img, clazz, context, imgname)

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
        assert split in ["train", "val", "test"]
        return WaterBirds(
            base_path, split, transform, target_transform, reweighting, weights_path
        )


if __name__ == "__main__":
    WaterBirds.dataset_split(
        "data/waterbirds",
        "val",
        None,
        None,
    )
