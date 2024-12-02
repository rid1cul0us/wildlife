import os
import json


from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


# import numpy as np
# import torchvision.transforms as tv_trans
# import matplotlib.pyplot as plt


class wilds_iwildcam(ImageFolder):
    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_mega_label_fix: bool = False,
        threshold: float = 0.2,
    ):
        super().__init__(os.path.join(root, subset), transform, target_transform)
        self.num_classes = 182
        self.num_points = None
        self.class_cnt = [0] * self.num_classes
        self.use_mega_label_fix = use_mega_label_fix

        # # build idx_to_class, class_to_idx and y_reorder
        # count_filepath = os.path.join(
        #     "/home/kafm/program/wildlife/data/iwildcam/dist.csv"
        # )
        # fd = open(count_filepath)
        # fd.readline()
        # self.y_reorder, self.idx_to_class, self.class_to_idx = {}, {}, {}
        # for idx, line in enumerate(fd.readlines()):
        #     _, _, y, name = line.strip().split(",")
        #     self.y_reorder[int(y)] = idx
        #     self.idx_to_class[idx] = name
        #     self.class_to_idx[name] = idx

        # # build class_cnt
        # count_filepath = os.path.join(
        #     "/home/kafm/program/wildlife/data/iwildcam/dist_train.csv"
        # )
        # fd = open(count_filepath)
        # fd.readline()
        # for line in fd.readlines():
        #     _, count, y, name = line.strip().split(",")
        #     y = self.y_reorder[int(y)]
        #     self.class_cnt[y] = count

        self.idx_to_class = None
        self.class_to_idx = None

        # to fix ImageFolder target fault
        with open(f"{root}/metadata/cls_to_idx.json", "r") as f:
            cls_to_idx = json.load(f)
        self.cls_idx_fix = cls_to_idx

        if use_mega_label_fix:
            mega_label_path = os.path.join(
                root,
                f"metadata/mega_label/mega_label_p{threshold*100:.0f}_from_r20.json",
            )
            if os.path.exists(mega_label_path):
                with open(mega_label_path, "r") as f:
                    mega_label = json.load(f)
            else:
                from datasets.aux_wilds_mega_lable_rebuild import mega_label_rebuild

                mega_label = mega_label_rebuild(
                    src_path=os.path.join(root, "metadata/mega_raw/mega_raw_p20.json"),
                    threshold=threshold,
                    dump_path=mega_label_path,
                )
            self.mega_label = mega_label

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        _, pos, seq, f_name = path.rsplit("/", 3)
        if self.use_mega_label_fix:
            mega_y = self.mega_label[f_name]
        else:
            mega_y = -1

        # change class name to a unified index
        clazz = self.cls_idx_fix[self.classes[target]]
        point = int(pos[2:])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            clazz = self.target_transform(clazz)
            point = self.target_transform(point)

        if self.use_mega_label_fix:
            # only solve target is animal but mega found none
            if mega_y is False:
                target = self.cls_idx_fix["empty"]

        return sample, clazz, point
