import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce)

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        targets = targets.detach()
        loss = self.nll_loss(log_p, targets)
        return loss



def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    # print(requires_grad)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w).long(), 1)
    return Variable(one_hot, requires_grad=requires_grad)



class WeightedmIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=5,norm=False, upper_bound=1.0):
        super(WeightedmIoULoss, self).__init__()
        self.classes = n_classes
        self.norm = norm
        self.upper_bound = upper_bound
        self.weights = weight

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, target, is_target_variable=False):

        target_cpu = target.data.cpu().numpy()
        self.weights = self.calculateWeights(target_cpu)
        self.weights = torch.Tensor(self.weights).cuda()


        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target .detach(), self.classes).float()


        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (self.weights * union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

