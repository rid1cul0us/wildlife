import timm
import torch
import torch.nn as nn
from torchvision import models
import torch.distributed as dist
from timm.models.layers import trunc_normal_

import utils
import numpy as np
from copy import deepcopy

# from models.vision_transformer import vit_small, vit_base, vit_large


def _get_model(arch, resume=False):
    model_map = {
        "shufflenet": (
            models.shufflenet_v2_x1_0,
            models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
        ),
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
        "swinv2_vit_tiny": (models.swin_v2_t, models.Swin_V2_T_Weights.DEFAULT),
        "vit_base": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
        "linear_clf": None,
    }

    net, weights = model_map[arch]
    if resume:
        model = net()
    else:
        # download at main process other processes use cache
        with utils.main_process_first():
            model = net(weights=weights)
    return model


def _init_weights(m, arch: str):
    # for ly in m.modules()
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if arch.startswith("resnet"):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
        else:
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features: int, resume=False):
        super().__init__()
        self.arch = "LinearClassifier"
        self.encoder = nn.Identity()
        self.classifier = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        _init_weights(self.classifier, self.arch)

    def forward_feature(self, x):
        return x

    def forward(self, x):
        return self.classifier(x)


class ConvNet(nn.Module):
    def __init__(self, arch, out_features: int, resume=False):
        super().__init__()

        model = _get_model(arch=arch, resume=resume)

        if arch.startswith("convnext"):
            classifier = model.classifier
            fc = nn.Linear(
                in_features=classifier[-1].in_features,
                out_features=out_features,
                bias=True,
            )
            classifier[-1] = fc
            delattr(model, "classifier")
            _init_weights(classifier[-1], arch)
        elif arch.startswith("resnet") or arch.startswith("shufflenet"):
            fc_in_features = model.fc.in_features
            classifier = torch.nn.Linear(
                in_features=fc_in_features, out_features=out_features, bias=True
            )
            delattr(model, "fc")
            _init_weights(classifier, arch)

        self.encoder = model
        self.arch = self.encoder.arch = arch
        self.classifier = classifier

    def forward_feature(self, x):
        if self.arch.startswith("convnext"):
            x = self.encoder.features(x)
            embedding = self.encoder.avgpool(x)
        elif self.arch.startswith("resnet"):
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = self.encoder.avgpool(x)
            embedding = torch.flatten(x, 1)
        elif self.arch.startswith("shufflenet"):
            x = self.encoder.conv1(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.stage2(x)
            x = self.encoder.stage3(x)
            x = self.encoder.stage4(x)
            x = self.encoder.conv5(x)
            embedding = x.mean([2, 3])  # globalpool
        else:
            embedding = self.encoder(x)
        return embedding

    def forward(self, x):
        x = self.forward_feature(x)
        return self.classifier(x)


class VisionTransformer(nn.Module):
    def __init__(self, args, out_features: int, resume=False, fix_backbone=True):
        super().__init__()

        arch = args.arch

        if "dinov2" in arch:
            arch = arch.lower()[args.arch.index("_") + 1 :]
            vit_kwargs = dict(
                img_size=args.cfg.crops.global_crops_size,
                patch_size=args.cfg.patch_size,
                init_values=args.cfg.layerscale,
                ffn_layer=args.cfg.ffn_layer,
                block_chunks=args.cfg.block_chunks,
                qkv_bias=args.cfg.qkv_bias,
                proj_bias=args.cfg.proj_bias,
                ffn_bias=args.cfg.ffn_bias,
            )
            vit: nn.Module = {
                "vit_small": vit_small,
                "vit_base": vit_base,
                "vit_large": vit_large,
            }[arch](**vit_kwargs)

            with utils.main_process_first():
                state = torch.hub.load_state_dict_from_url(args.cfg.pretrained_weights)
                # state = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                vit.load_state_dict(state)

            vit.mask_token = None  # do not train with masks
            if fix_backbone:
                vit.requires_grad_(False)  # fix vit
        elif "swin" in arch:
            vit: nn.Module = _get_model(arch)
            vit.embed_dim = vit.head.in_features
            vit.head = nn.Identity()
        elif "vit" in arch:
            vit: nn.Module = _get_model(arch)
            vit.embed_dim = vit.heads.head.in_features
            vit.heads.head = nn.Identity()
        else:
            assert 0

        self.encoder = vit
        self.classifier = nn.Linear(
            in_features=vit.embed_dim, out_features=out_features, bias=True
        )
        _init_weights(self.classifier, arch)

    def forward_feature(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


class DoubleHeadModel(nn.Module):
    def __init__(self, arch, model, num_classes: int, num_points: int, resume=False):
        super(DoubleHeadModel, self).__init__()

        class_clf = model.classifier
        delattr(model, "classifier")
        if arch.startswith("convnext"):
            fc_in_features = class_clf[-1].in_features
            point_clf = deepcopy(class_clf)
            point_clf[-1] = nn.Linear(
                in_features=fc_in_features, out_features=num_points, bias=True
            )
        else:
            fc_in_features = class_clf.in_features
            point_clf = nn.Linear(
                in_features=fc_in_features, out_features=num_points, bias=True
            )
        _init_weights(point_clf, arch)

        self.encoder = model.encoder
        self.encoder.forward_feature = model.forward_feature
        self.class_clf = class_clf
        self.point_clf = point_clf

    def forward_feature(self, x):
        return self.encoder.forward_feature(x)

    def forward_all(self, x):
        embedding = self.forward_feature(x)
        class_output = self.class_clf(embedding)
        point_output = self.point_clf(embedding)
        return class_output, point_output

    def _train_forward(self, x):
        embedding = self.forward_feature(x)
        class_output = self.class_clf(embedding)
        point_output = self.point_clf(embedding)
        return class_output, point_output

    def _eval_forward(self, x):
        embedding = self.forward_feature(x)
        class_output = self.class_clf(embedding)
        return class_output

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.forward = self._train_forward if mode else self._eval_forward
        return self

    def eval(self):
        return self.train(False)


class DHConvNeXt(nn.Module):
    def __init__(self, arch, num_classes: int, num_points: int, resume=False):
        super().__init__()

        model = _get_model(arch=arch, resume=resume)

        class_clf = model.classifier
        fc_in_features = class_clf[-1].in_features
        delattr(model, "classifier")
        # model.classifier = nn.Identity()
        point_clf = deepcopy(class_clf)

        class_clf[-1] = nn.Linear(
            in_features=fc_in_features, out_features=num_points, bias=True
        )
        point_clf[-1] = nn.Linear(
            in_features=fc_in_features, out_features=num_classes, bias=True
        )
        _init_weights(class_clf[-1], arch)
        _init_weights(point_clf[-1], arch)

        self.encoder = model
        self.class_clf = class_clf
        self.point_clf = point_clf

    def forward_features(self, x):
        x = self.encoder.features(x)
        # x = self.encoder.avgpool(x)
        embedding = self.encoder.avgpool(x)
        return embedding

    def forward_class_only(self, x):
        embedding = self.forward_features(x)
        class_output = self.class_clf(embedding)
        return class_output

    def _train_forward(self, x):
        embedding = self.forward_features(x)
        class_output = self.class_clf(embedding)
        point_output = self.point_clf(embedding)
        return class_output, point_output

    def _eval_forward(self, x):
        embedding = self.forward_features(x)
        class_output = self.class_clf(embedding)
        return class_output

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.forward = self._train_forward if mode else self._eval_forward
        return self

    def eval(self):
        return self.train(False)


class DHResNet(nn.Module):
    def __init__(self, arch, num_classes: int, num_points: int, resume=False):
        super().__init__()

        model = _get_model(arch=arch, resume=resume)

        fc_in_features = model.fc.in_features
        delattr(model, "fc")
        # model.fc = nn.Identity()

        point_clf = nn.Linear(
            in_features=fc_in_features, out_features=num_points, bias=True
        )
        class_clf = nn.Linear(
            in_features=fc_in_features, out_features=num_classes, bias=True
        )
        _init_weights(class_clf, arch)
        _init_weights(point_clf, arch)

        self.encoder = model
        self.class_clf = class_clf
        self.point_clf = point_clf

    def forward_features(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        embedding = torch.flatten(x, 1)
        return embedding

    def forward_class_only(self, x):
        embedding = self.forward_features(x)
        class_output = self.class_clf(embedding)
        return class_output

    def _train_forward(self, x):
        embedding = self.forward_features(x)
        class_output = self.class_clf(embedding)
        point_output = self.point_clf(embedding)
        return class_output, point_output

    def _eval_forward(self, x):
        embedding = self.forward_features(x)
        class_output = self.class_clf(embedding)
        return class_output

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.forward = self._train_forward if mode else self._eval_forward
        return self

    def eval(self):
        return self.train(False)


def get_model(args, num_classes: int, num_points=None, resume=False):
    arch = args.arch
    algorithm = args.method if hasattr(args, "method") else "ERM"

    if any(name in arch for name in ["dino", "vit", "clip", "transformer"]):
        model = VisionTransformer(args=args, out_features=num_classes, resume=resume)
    elif (
        arch.startswith("convnext")
        or arch.startswith("resnet")
        or arch.startswith("shufflenet")
    ):
        model = ConvNet(arch=arch, out_features=num_classes, resume=resume)
    elif arch == "linear_clf":
        model = LinearClassifier(
            in_features=args.in_features, out_features=num_classes, resume=resume
        )
    else:
        raise NotImplementedError()

    if algorithm in ["DU", "ADU", "MT"]:
        assert num_points
        model = DoubleHeadModel(arch, model, num_classes, num_points, resume)
    # if algorithm == 'ADU':
    # model.adu = nn.Parameter(torch.Tensor(np.load(args.weights_path)), requires_grad=True) # AdaptiveWeightLayer(np.load(args.weights_path))
    # if algorithm == 'ADU':
    return model


if __name__ == "__main__":
    model = DoubleHeadModel(
        arch="resnet50",
        model=ConvNet(arch="resnet50", out_features=182),
        num_classes=182,
        num_points=243,
    )
    print(model)
    # print()
    # model.adu = nn.Parameter(torch.Tensor(np.load('weights/train/original_loss_weight.npy')), requires_grad=True)
    # for name, param in model.named_parameters():
    #     print(name, param)

    #     torch.nn.CrossEntropyLoss
