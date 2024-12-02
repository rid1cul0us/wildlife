import torch
from torch import nn as nn


class AdaptiveWeightLayer(nn.Module):
    def __init__(self, loss_weights):
        super(AdaptiveWeightLayer, self).__init__()
        # self.class_weights = nn.Parameter(torch.Tensor(class_weights))
        # if num_points:
        # self.points_weights = nn.Parameter(torch.Tensor(point_weights))
        self.weights = nn.Parameter(torch.Tensor(loss_weights), requires_grad=True)
