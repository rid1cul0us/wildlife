import os
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms as transforms

# from dataset.CCT20Dataset import CCT20DataSet
# from dataset.CCT20NoEmpty import CCT20NoEmpty
from dataset.IWildCam import IWildCam
from dataset.TerraInc import TerraInc
from dataset.WaterBirds import WaterBirds
from dataset.IWildCamDataset import IWildCamDataset
from dataset.IWildCamFeature import IWildCamFeature, IWildCamRandAugFeature


def get_dataset(dataset):
    dataset_map = {
        # 'cct20' : CCT20DataSet,
        "iwildcam": IWildCam,
        "waterbirds": WaterBirds,
        "iwildcam-folder": IWildCamDataset,
        "terrainc": TerraInc,
        "iwildcamfeature": IWildCamFeature,
        "iwildcam_randaug_feature": IWildCamRandAugFeature,
    }

    dataset = dataset_map[dataset.lower()]

    return dataset
