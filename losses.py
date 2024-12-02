import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, one_hot, softmax


def get_loss(args, **loss_kwargs):
    # losses ={
    #     'CBCELoss': CBCELoss,
    #     'FocalLoss': FocalLoss,
    #     'LDAMLoss': LDAMLoss,
    # }

    criterion_name = args.loss

    if hasattr(torch.nn, criterion_name):
        return getattr(torch.nn, criterion_name)(**loss_kwargs)

    loss_kwargs.pop("reduction")
    if criterion_name == "CBFocalLoss":
        assert all(
            arg in loss_kwargs.keys() for arg in ["alpha", "beta", "gamma"]
        ), "should set 'alpha', 'beta', 'gamma' for CB Focal Loss"
        return CBLoss(class_cnt=args.class_cnt, loss_type="focal", **loss_kwargs)
    if criterion_name == "CBCELoss":
        return CBLoss(class_cnt=args.class_cnt, loss_type="ce", **loss_kwargs)
    if criterion_name == "LDAMLoss":
        assert all(
            arg in loss_kwargs.keys() for arg in ["max_m", "s"]
        ), "should set 'max_m' and 's' for LDAM Loss"
        return LDAMLoss(class_cnt=args.class_cnt, **loss_kwargs)
    if criterion_name == "FocalLoss":
        assert all(
            arg in loss_kwargs.keys() for arg in ["alpha", "gamma"]
        ), "should set 'alpha', 'gamma' for Focal Loss"
        return FocalLoss(**loss_kwargs)
    if criterion_name == "SeesawLoss":
        assert all(
            arg in loss_kwargs.keys() for arg in ["p", "q", "eps"]
        ), "should set 'p', 'q', 'eps' for Seesaw Loss"
        return SeesawLoss(num_classes=args.num_classes, **loss_kwargs)
    if criterion_name == "EqualizedFocalLoss":
        assert all(
            arg in loss_kwargs.keys() for arg in ["gamma_b", "scale_factor"]
        ), "should set 'gamma_b', 'scale_factor' for Equalized Focal Loss"
        return EqualizedFocalLoss(**loss_kwargs)
    # if criterion_name == 'CPLoss':    # CP Loss implement in `reweighting.py`
    # return CPLoss(**loss_kwargs)

    assert None, f"unsupported loss function name: {criterion_name}"


# reference https://github.com/qiqi-helloworld/ABSGD/blob/main/mylosses.py
def focal_loss(input_values, alpha, gamma, reduction="mean"):
    """Computes the focal loss"""

    """
        input_values = -\log(p_t)
        loss = - \alpha_t (1-\p_t)\log(p_t)
        """
    p = torch.exp(-input_values)
    loss = alpha * (1 - p) ** gamma * input_values

    if reduction == "none":
        return loss
    else:
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        return focal_loss(
            F.cross_entropy(input, target, reduction="none", weight=weight),
            self.alpha,
            self.gamma,
            reduction=self.reduction,
        )


class LDAMLoss(nn.Module):
    def __init__(self, class_cnt, weight=None, max_m=0.8, s=30, reduction="mean"):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(class_cnt))  # 1/n_j^{1/4}
        m_list = m_list * (max_m / np.max(m_list))  # control the length of margin
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target):
        index = torch.zeros_like(output, dtype=torch.bool)
        # target = target.type(torch.cuda.LongTensor)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = output - batch_m

        output = torch.where(index, x_m, output)

        return F.cross_entropy(
            self.s * output, target, weight=self.weight, reduction=self.reduction
        )


class CBLoss(nn.Module):
    def __init__(
        self, beta, class_cnt, reduction="mean", loss_type="ce", alpha=0.25, gamma=2
    ):
        super(CBLoss, self).__init__()
        self.class_cnt = class_cnt
        self.reduction = reduction
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, out, target):
        ny = [self.class_cnt[i.item()] for i in target]
        ny = torch.Tensor(ny).to(target.get_device())
        if self.loss_type == "ce":
            loss = F.cross_entropy(input=out, target=target, reduction="none")
        elif self.loss_type == "focal":
            loss = focal_loss(
                F.cross_entropy(input=out, target=target, reduction="none"),
                self.alpha,
                self.gamma,
                reduction="none",
            )
        loss = (1 - self.beta) / (1 - self.beta**ny) * loss
        if self.reduction == "none":
            return loss
        else:
            return loss.mean()


# reference https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
class OfficialCBLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma=2):
        super(OfficialCBLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.gamma = gamma
        self.beta = beta

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        self.weights = torch.tensor(weights).cuda()

    def forward(self, input, target):
        return self.CB_loss(target, input)

    def CB_loss(self, labels, logits):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        samples_per_cls: A python list of size [no_of_classes].
        no_of_classes: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.

        Returns:
        cb_loss: A float tensor representing class balanced loss
        """

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = self.weights.clone().float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            labels, logits, alpha, gamma = labels_one_hot, logits, weights, self.gamma
            BCLoss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels, reduction="none"
            )

            if gamma == 0.0:
                modulator = 1.0
            else:
                modulator = torch.exp(
                    -gamma * labels * logits
                    - gamma * torch.log(1 + torch.exp(-1.0 * logits))
                )

            loss = modulator * BCLoss
            weighted_loss = alpha * loss

            focal_loss = torch.sum(weighted_loss)
            cb_loss = focal_loss / torch.sum(labels)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, weight=weights
            )
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred, target=labels_one_hot, weight=weights
            )
        return cb_loss


# reference https://github.com/jahongir7174/SeesawLoss/blob/master/seesawloss/seesawloss.py
class SeesawLoss(torch.nn.Module):
    """
    Implementation of seesaw loss.
    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>
    Args:
        num_classes (int): The number of classes.
                Default to 1000 for the ImageNet dataset.
        p (float): The ``p`` in the mitigation factor.
                Defaults to 0.8.
        q (float): The ``q`` in the compensation factor.
                Defaults to 2.0.
        eps (float): The min divisor to smooth the computation of compensation factor.
                Default to 1e-2.
    """

    def __init__(self, num_classes=1000, p=0.8, q=2.0, eps=1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps

        # cumulative samples for each category
        accumulated = torch.zeros(self.num_classes, dtype=torch.float).cuda()
        self.register_buffer("accumulated", accumulated)

    def forward(self, outputs, targets):
        # accumulate the samples for each category
        for unique in targets.unique():
            self.accumulated[unique] += (targets == unique.item()).sum()

        onehot_targets = one_hot(targets, self.num_classes)
        seesaw_weights = outputs.new_ones(onehot_targets.size())

        # mitigation factor
        if self.p > 0:
            matrix = self.accumulated[None, :].clamp(min=1) / self.accumulated[
                :, None
            ].clamp(min=1)
            index = (matrix < 1.0).float()
            sample_weights = matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[targets.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # compensation factor
        if self.q > 0:
            scores = softmax(outputs.detach(), dim=1)
            self_scores = scores[
                torch.arange(0, len(scores)).to(scores.device).long(), targets.long()
            ]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        outputs = outputs + (seesaw_weights.log() * (1 - onehot_targets))

        return cross_entropy(outputs, targets)


# reference https://github.com/tcmyxc/Equalized-Focal-Loss/blob/master/efl.py


class EqualizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma_b=2, scale_factor=8, reduction="mean"):
        super(EqualizedFocalLoss, self).__init__()

        self.gamma_b = gamma_b
        self.scale_factor = scale_factor
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        return equalized_focal_loss(
            input, target, self.gamma_b, self.scale_factor, self.reduction
        )


def equalized_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_b=2,
    scale_factor=8,
    reduction="mean",
):
    """EFL loss"""
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    outputs = F.cross_entropy(logits, targets)
    log_pt = -ce_loss
    pt = torch.exp(log_pt)

    targets = targets.view(-1, 1)
    grad_i = torch.autograd.grad(outputs=-outputs, inputs=logits)[0]
    grad_i = grad_i.gather(1, targets)
    pos_grad_i = F.relu(grad_i).sum()
    neg_grad_i = F.relu(-grad_i).sum()
    neg_grad_i += 1e-9
    grad_i = pos_grad_i / neg_grad_i
    grad_i = torch.clamp(grad_i, min=0, max=1)

    dy_gamma = gamma_b + scale_factor * (1 - grad_i)
    dy_gamma = dy_gamma.view(-1)
    # weighting factor
    wf = dy_gamma / gamma_b
    weights = wf * (1 - pt) ** dy_gamma

    efl = weights * ce_loss

    if reduction == "sum":
        efl = efl.sum()
    elif reduction == "mean":
        efl = efl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return efl


# reference https://github.com/tcmyxc/Equalized-Focal-Loss/blob/master/efl.py
def balanced_equalized_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_t=0.25,
    gamma_b=2,
    scale_factor=8,
    reduction="mean",
):
    """balanced EFL loss"""
    return alpha_t * equalized_focal_loss(
        logits, targets, gamma_b, scale_factor, reduction
    )


# reference https://github.com/qiqi-helloworld/ABSGD/blob/main/wilds-competition/main.py
# criterion = ABSGD(args, "CE", abAlpha=1)


# reference https://github.com/qiqi-helloworld/ABSGD/blob/main/mylosses.py
class ABSGD(nn.Module):
    def __init__(self, args, loss_type, abAlpha=0.5):
        super(ABSGD, self).__init__()
        self.loss_type = loss_type
        self.u = 0
        self.gamma = args.drogamma
        self.abAlpha = abAlpha
        self.criterion = CBCELoss(reduction="none")
        if "ldam" in self.loss_type:
            # print(args.cls_num_list)
            self.criterion = LDAMLoss(
                cls_num_list=args.cls_num_list, max_m=0.5, s=30, reduction="none"
            )
        elif "focal" in self.loss_type:
            self.criterion = FocalLoss(gamma=args.gamma, reduction="none")

    def forward(self, output, target, cls_weights=None, myLambda=200):
        loss = self.criterion(output, target, cls_weights)
        if myLambda >= 200:  # reduces to CE
            p = torch.tensor(1 / len(loss))
        else:
            expLoss = torch.exp(loss / myLambda)
            # u  = (1 - gamma) * u + gamma * alpha * g
            self.u = (1 - self.gamma) * self.u + self.gamma * (
                self.abAlpha * torch.mean(expLoss)
            )
            drop = expLoss / (self.u * len(loss))
            drop.detach_()
            p = drop

        return torch.sum(p * loss)
