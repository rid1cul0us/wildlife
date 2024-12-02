import os
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms as transforms
from torch.utils.data import DataLoader, Dataset


class CCT20NoEmpty(Dataset):
    # dataset csv file path
    base_path = "/home/kafm/data/CCT20_benchmark/images/"
    train_path = "/home/kafm/program/dl/dataset/meta/noemptyCCT20/train.csv"
    cis_val_path = "/home/kafm/program/dl/dataset/meta/noemptyCCT20/cis_val.csv"
    trans_val_path = "/home/kafm/program/dl/dataset/meta/noemptyCCT20/trans_val.csv"
    cis_test_path = "/home/kafm/program/dl/dataset/meta/noemptyCCT20/cis_test.csv"
    trans_test_path = "/home/kafm/program/dl/dataset/meta/noemptyCCT20/trans_test.csv"

    # R_mean is 0.333401, G_mean is 0.341779, B_mean is 0.320772
    # R_var is 0.237052, G_var is 0.237999, B_var is 0.231040
    normMean = [0.3334, 0.3418, 0.3207]
    normStd = [0.2371, 0.2380, 0.2310]

    classid = {
        "opossum": 0,
        "raccoon": 1,
        "rabbit": 2,
        "coyote": 3,
        "bobcat": 4,
        "cat": 5,
        # 'empty': 6,
        "squirrel": 6,
        "dog": 7,
        "car": 8,
        "bird": 9,
        "skunk": 10,
        "rodent": 11,
        "deer": 12,
        "badger": 13,
        "fox": 14,
    }
    classnames = dict(zip(classid.values(), classid.keys()))

    dataset_kwargs = {
        "num_workers": 4,
        "drop_last": True,
        "pin_memory": True,
    }

    def id2name_noempty(self, id: int) -> str:
        return CCT20NoEmpty.classnames[id]

    def name2id_noempty(self, classname: str) -> int:
        return CCT20NoEmpty.classid[classname]

    def __init__(self, path, transform=None, target_transform=None, base_path=""):
        super().__init__()

        self.path = path
        self.base_path = base_path
        self.num_classes = len(CCT20NoEmpty.classid)

        self.transform = transform
        self.target_transform = target_transform

        self.targets = []
        self.imgpaths = []
        self.target_cnt = [0] * num_classes

        f = open(path, "r")
        f.readline()
        for line in f:
            path, _, _, _, classname = line.strip().split(",")
            self.imgpaths.append(path)
            self.targets.append(CCT20NoEmpty.name2id_noempty(classname))
            self.target_cnt[CCT20NoEmpty.name2id_noempty(classname)] += 1

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

    def _train_dataset(transform):
        return CCT20NoEmpty(
            CCT20NoEmpty.train_path,
            transform=transform,
            base_path=CCT20NoEmpty.base_path,
        )

    def iid_val_dataset(transform):
        return CCT20NoEmpty(
            CCT20NoEmpty.cis_val_path,
            transform=transform,
            base_path=CCT20NoEmpty.base_path,
        )

    def iid_test_dataset(transform):
        return CCT20NoEmpty(
            CCT20NoEmpty.cis_test_path,
            transform=transform,
            base_path=CCT20NoEmpty.base_path,
        )

    def ood_val_dataset(transform):
        return CCT20NoEmpty(
            CCT20NoEmpty.trans_val_path,
            transform=transform,
            base_path=CCT20NoEmpty.base_path,
        )

    def ood_test_dataset(transform):
        return CCT20NoEmpty(
            CCT20NoEmpty.trans_test_path,
            transform=transform,
            base_path=CCT20NoEmpty.base_path,
        )
