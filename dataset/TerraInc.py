import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from reweighting import Reweight


class TerraInc(Dataset):
    num_classes = 10
    num_contexts = 2
    num_points = num_contexts

    # R_mean is 0.333401, G_mean is 0.341779, B_mean is 0.320772
    # R_var is 0.237052, G_var is 0.237999, B_var is 0.231040
    normMean = [0.3334, 0.3418, 0.3207]
    normStd = [0.2371, 0.2380, 0.2310]

    dataset_kwargs = {
        "num_workers": 2,
        "pin_memory": True,
    }

    location_map = {"38": 0, "43": 1, "46": 2, "100": 3}
    split_map = {0: "train", 1: "val", 2: "train", 3: "test"}
    species_map = {
        "bird": 0,
        "bobcat": 1,
        "cat": 2,
        "coyote": 3,
        "dog": 4,
        "empty": 5,
        "opossum": 6,
        "rabbit": 7,
        "raccoon": 8,
        "squirrel": 9,
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
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            self.target_transform = transforms.Compose([torch.Tensor])

        metadatapath = os.path.join(root, "metadata.csv")
        if not os.path.exists(metadatapath):
            TerraInc._build_metadata(root, metadatapath)
        metadata = pd.read_csv(metadatapath, index_col=0)
        metadata = metadata[metadata["split"] == split]
        self.data = list(metadata.itertuples())

        # reweighting
        self.weights_path = weights_path
        if weights_path:
            self.loss_weights = np.load(self.weights_path)
        getattr(Reweight, reweighting)(self)

        self.class_to_idx = self.species_map
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_cnt = [0] * self.num_classes
        self.y_context = []
        for item in self.data:
            self.class_cnt[item[3]] += 1
            self.y_context.append(item[2])

    def __getitem__(self, index: int):
        imgpath, split, context, clazz, species = self.data[index]
        img = Image.open(imgpath)
        return self.reweight_item(self, index, img, clazz, context, imgpath)

    def __len__(self):
        return len(self.data)

    def _build_metadata(root: str, output_path):
        with open(output_path, "w") as csv:
            csv.write("path,split,location,y,species\n")
            for base, dirs, files in os.walk(root):
                for f in files:
                    if not f.endswith(".jpg"):
                        continue
                    path = os.path.join(base, f)
                    location, species, _ = path.split("/")[2:]
                    location = location_map[location.split("_")[1]]
                    split = split_map[location]
                    y = species_map[species]
                    csv.write(f"{path},{split},{location},{y},{species}\n")

    def dataset_split(
        root,
        split,
        transform,
        target_transform,
        reweighting="NoneWeighting",
        weights_path=None,
    ):
        assert split in ["train", "val", "test"]
        return TerraInc(
            root, split, transform, target_transform, reweighting, weights_path
        )


if __name__ == "__main__":
    train_split = TerraInc.dataset_split("data/terrainc", "train", None, None)
    for idx, (img, target, side_info) in enumerate(train_split):
        print(idx, img, target, side_info)
        break
    print(len(train_split))
