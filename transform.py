import inspect
import random
import timm

import torch
import torchvision.transforms as transforms

import utils


class transform:
    def base_transform(scale, normMean, normStd):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(scale=(0.85, 1.0), size=scale),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(normMean, normStd),
            ]
        )

    def base_augment_transform(scale, normMean, normStd):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(scale=(0.85, 1.0), size=scale),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.5, contrast=0.2, saturation=0.2, hue=0.3
                        ),
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                utils.GaussianBlur(p=0.1),
                utils.Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(normMean, normStd),
            ]
        )

    def randaugment_transform(scale, normMean, normStd, **augment_kwargs):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(scale=(0.85, 1.0), size=scale),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.autoaugment.RandAugment(**augment_kwargs),
                transforms.ToTensor(),
                transforms.Normalize(normMean, normStd),
            ]
        )

    def timm_randaugment_transform(scale, normMean, normStd, **augment_kwargs):
        # in timm, color_jitter can not be used with randaugment together
        return timm.data.create_transform(
            input_size=scale,
            scale=(0.85, 1.0),
            is_training=True,
            mean=normMean,
            std=normStd,
            **augment_kwargs
        )

    def test_transform(scale, normMean, normStd):
        return transforms.Compose(
            [
                transforms.Resize(size=scale),
                transforms.ToTensor(),
                transforms.Normalize(normMean, normStd),
            ]
        )

    def target_transform(dataset):
        def to_one_hot(label):
            one_hot = [0] * dataset.num_classes
            one_hot[label] = 1
            return torch.Tensor(one_hot)

        return transforms.Compose([to_one_hot])

    # def tensor_transform(img):
    # return torch.Tensor(img)

    def tensor_transform(scale, normMean, normStd):
        return lambda img: torch.Tensor(img)

    def get_transform(transform_name: str):
        return __class__.__dict__[transform_name]
