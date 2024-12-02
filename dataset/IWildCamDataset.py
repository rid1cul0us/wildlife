import os
import torch

from torchvision import transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from dataset.aux_wilds_iwildcam import wilds_iwildcam


class IWildCamDataset(Dataset):
    num_classes = 182
    num_points = None

    base_dir = "/data/wilds_rebuild/"
    train_dir, id_val_dir, id_test_dir, val_dir, test_dir = [
        "train",
        "id_val",
        "id_test",
        "val",
        "test",
    ]

    # R_mean is 0.384172, G_mean is 0.388226, B_mean is 0.357814
    # R_var is 0.262695, G_var is 0.262318, B_var is 0.262127
    normMean = [0.384172, 0.388226, 0.357814]
    normStd = [0.262695, 0.262318, 0.262127]

    dataset_kwargs = {
        "num_workers": 6,
        "pin_memory": True,
    }

    metric_name = ["macro-f1", "acc", "loss"]
    split_name = ["train", "id_val", "val", "id_test", "test"]

    def _train_dataset(transform, target_transform, use_mega_label_fix):
        return wilds_iwildcam(
            IWildCamDataset.base_dir,
            IWildCamDataset.train_dir,
            transform=transform,
            target_transform=target_transform,
            use_mega_label_fix=use_mega_label_fix,
        )

    def _iid_val_dataset(transform, target_transform, use_mega_label_fix):
        return wilds_iwildcam(
            IWildCamDataset.base_dir,
            IWildCamDataset.id_val_dir,
            transform=transform,
            target_transform=target_transform,
            use_mega_label_fix=use_mega_label_fix,
        )

    def _iid_test_dataset(transform, target_transform, use_mega_label_fix):
        return wilds_iwildcam(
            IWildCamDataset.base_dir,
            IWildCamDataset.id_test_dir,
            transform=transform,
            target_transform=target_transform,
            use_mega_label_fix=use_mega_label_fix,
        )

    def _ood_val_dataset(transform, target_transform, use_mega_label_fix):
        return wilds_iwildcam(
            IWildCamDataset.base_dir,
            IWildCamDataset.val_dir,
            transform=transform,
            target_transform=target_transform,
            use_mega_label_fix=use_mega_label_fix,
        )

    def _all_val_dataset(transform, target_transform, use_mega_label_fix):
        iid_val = IWildCamDataset._iid_val_dataset(
            transform, target_transform, use_mega_label_fix=use_mega_label_fix
        )
        ood_val = IWildCamDataset._ood_val_dataset(
            transform, target_transform, use_mega_label_fix=use_mega_label_fix
        )
        dataset = ConcatDataset([iid_val, ood_val])
        dataset.num_classes = iid_val.num_classes
        dataset.num_points = iid_val.num_points
        dataset.idx_to_class = iid_val.idx_to_class
        dataset.class_to_idx = iid_val.class_to_idx

        return dataset

    def _ood_test_dataset(transform, target_transform, use_mega_label_fix):
        return wilds_iwildcam(
            IWildCamDataset.base_dir,
            IWildCamDataset.test_dir,
            transform=transform,
            target_transform=target_transform,
            use_mega_label_fix=use_mega_label_fix,
        )

    def dataset_split(
        root_dir,
        split,
        transform,
        target_transform,
        reweighting=None,
        weights_path=None,
    ):
        return {
            "train": IWildCamDataset._train_dataset,
            "id_val": IWildCamDataset._iid_val_dataset,
            "val": IWildCamDataset._ood_val_dataset,
            "all_val": IWildCamDataset._all_val_dataset,
            "id_test": IWildCamDataset._iid_test_dataset,
            "test": IWildCamDataset._ood_test_dataset,
        }[split](transform, target_transform, use_mega_label_fix=True)
