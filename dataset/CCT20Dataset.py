import os
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms as transforms
from torch.utils.data import DataLoader, Dataset


class CCT20DataSet(Dataset):
    # dataset csv file path
    image_base_path = "/home/kafm/data/CCT20_benchmark/images/"
    train_path = "/home/kafm/program/dl/dataset/meta/CCT20/train.csv"
    cis_val_path = "/home/kafm/program/dl/dataset/meta/CCT20/cis_val.csv"
    trans_val_path = "/home/kafm/program/dl/dataset/meta/CCT20/trans_val.csv"
    cis_test_path = "/home/kafm/program/dl/dataset/meta/CCT20/cis_test.csv"
    trans_test_path = "/home/kafm/program/dl/dataset/meta/CCT20/trans_test.csv"

    # R_mean is 0.333401, G_mean is 0.341779, B_mean is 0.320772
    # R_var is 0.237052, G_var is 0.237999, B_var is 0.231040
    normMean = [0.3334, 0.3418, 0.3207]
    normStd = [0.2371, 0.2380, 0.2310]

    class_id = {
        "opossum": 0,
        "raccoon": 1,
        "squirrel": 2,
        "bobcat": 3,
        "skunk": 4,
        "dog": 5,
        "coyote": 6,
        "rabbit": 7,
        "bird": 8,
        "cat": 9,
        "badger": 10,
        "empty": 11,
        "car": 12,
        "deer": 13,
        "fox": 14,
        "rodent": 15,
    }
    classnames = dict(zip(class_id.values(), class_id.keys()))

    dataset_kwargs = {
        "num_workers": 4,
        "drop_last": True,
        "pin_memory": True,
    }

    def id2name(self, id: int) -> str:
        return CCT20DataSet.classnames[id]

    def name2id(self, classname: str) -> int:
        return CCT20DataSet.class_id[classname]

    def __init__(self, path, transform=None, target_transform=None, base_path=""):
        super().__init__()

        self.path = path
        self.base_path = base_path
        self.num_classes = len(CCT20DataSet.class_id)

        self.transform = transform
        self.target_transform = target_transform

        self.imgpaths = []
        self.targets = []
        self.target_cnt = [0] * num_classes

        f = open(path, "r")
        f.readline()
        for line in f:
            path, _, _, _, classname = line.strip().split(",")
            self.imgpaths.append(path)
            self.targets.append(name2id(classname))
            self.target_cnt[name2id(classname)] += 1

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            self.target_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        target = np.zeros((self.num_classes))
        target[self.targets[index]] = 1
        img = Image.open(os.path.join(self.base_path, self.imgpaths[index]))
        return self.transform(img), target

    def __len__(self):
        return len(self.imgpaths)

    def _train_dataset(transform, target_transform):
        return CCT20DataSet(
            CCT20DataSet.train_path,
            transform=transform,
            target_transform=target_transform,
            base_path=CCT20DataSet.image_base_path,
        )

    def iid_val_dataset(transform, target_transform):
        return CCT20DataSet(
            CCT20DataSet.cis_val_path,
            transform=transform,
            target_transform=target_transform,
            base_path=CCT20DataSet.image_base_path,
        )

    def iid_test_dataset(transform, target_transform):
        return CCT20DataSet(
            CCT20DataSet.cis_test_path,
            transform=transform,
            target_transform=target_transform,
            base_path=CCT20DataSet.image_base_path,
        )

    def ood_val_dataset(transform, target_transform):
        return CCT20DataSet(
            CCT20DataSet.trans_val_path,
            transform=transform,
            target_transform=target_transform,
            base_path=CCT20DataSet.image_base_path,
        )

    def ood_test_dataset(transform, target_transform):
        return CCT20DataSet(
            CCT20DataSet.trans_test_path,
            transform=transform,
            target_transform=target_transform,
            base_path=CCT20DataSet.image_base_path,
        )
