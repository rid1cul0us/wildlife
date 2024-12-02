import torch
import numpy as np
import torch.nn as nn


class Reweight:
    def NoneWeighting(dataset):
        dataset.sample_weight = lambda clazz: 1

        def reweight_item(self, index, img, clazz, point, imgname):
            return self.transform(img), clazz, (point, imgname)

        dataset.reweight_item = reweight_item

    def DU(dataset):  # DU reweighting Dual Uncertainty Reweighting
        assert dataset.weights_path
        dataset.loss_weights = np.load(dataset.weights_path)

        def reweight_item(self, index, img, clazz, point, imgname):
            # weight_class, weight_point = self.loss_weights[clazz][point]
            weight_class, weight_point = self.loss_weights[clazz][self.globalp[index]]
            return (
                self.transform(img),
                clazz,
                (point, weight_class, weight_point, imgname),
            )

        dataset.reweight_item = reweight_item

    def NumReweighting(dataset):
        dataset.resample_weight = 1.0 / torch.tensor(
            dataset.class_cnt, dtype=torch.float32
        )

        def reweight_item(self, index, img, clazz, point, imgname):
            return self.transform(img), clazz, (self.resample_weight[clazz], imgname)

        dataset.reweight_item = reweight_item

    def MaxNumReweighting(dataset):
        dataset.resample_weight = max(dataset.class_cnt) / torch.tensor(
            dataset.class_cnt, dtype=torch.float32
        )

        def reweight_item(self, index, img, clazz, point, imgname):
            return self.transform(img), clazz, (self.resample_weight[clazz], imgname)

        dataset.reweight_item = reweight_item

    def PU(dataset):  # PU reweighting Point Uncertainty (of Class) Reweighting
        assert dataset.weights_path

        def reweight_item(self, index, img, clazz, point, imgname):
            # weight_class, weight_point = self.loss_weights[clazz][point]
            weight_class, _ = self.loss_weights[clazz][self.y_context[index]]
            return self.transform(img), clazz, (weight_class, imgname)

        dataset.reweight_item = reweight_item

    def ADU(dataset):  # ADU reweighting Adaptive Dual Uncertainty Reweighting
        assert dataset.weights_path

        def reweight_item(self, index, img, clazz, point, imgname):
            return (
                self.transform(img),
                clazz,
                (point, clazz, self.globalp[index], imgname),
            )

        dataset.reweight_item = reweight_item


class WeightedLoss(nn.Module):
    def __init__(self, loss, weights_path, learnable):
        super().__init__()
        self.loss = loss
        self.learnable = learnable
        weights = np.load(weights_path)
        point_weights, class_weights = weights[0][:, 0], weights[:, 0][:, 1]
        self.class_weights = nn.Parameter(
            torch.Tensor(class_weights), requires_grad=learnable
        )
        self.point_weights = nn.Parameter(
            torch.Tensor(point_weights), requires_grad=learnable
        )

    # torch.nn.CrossEntropyLoss
    # def parameters(self, )

    def forward(self, input, target, index, type):
        losses = self.loss(input, target)
        weights = self.class_weights if type == "class" else self.point_weights
        weights.data.clamp_min_(0)
        loss_weights = list(map(lambda t: t.reshape(1), [weights[i] for i in index]))
        loss_weights = torch.cat(loss_weights, dim=0)
        return losses * loss_weights
