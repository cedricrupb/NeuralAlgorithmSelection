import torch as th
from torch.nn import functional as F
import math


class Rank_BCE(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = th.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y):

        filt = th.abs(2 * y - 1)
        loss = self.loss(x, y)
        loss = filt * loss
        return loss.mean()


class Kendall_Loss(th.nn.Module):

    def __init__(self, eps=1e-08):
        super().__init__()
        self.act = th.nn.Tanh()
        self.eps = eps

    def forward(self, p, y):

        y = 2 * y - 1
        p = self.act(p)

        loss = -y * p
        return loss.mean()


class Relational_Log_Loss(th.nn.Module):

    def __init__(self, eps=1e-08):
        super().__init__()
        self.act = th.nn.Tanh()
        self.eps = eps

    def forward(self, p, y):

        y = 2 * y - 1
        p = self.act(p)

        c = 1 + y*p
        c = c.clamp(self.eps)

        scale = 1 / math.log(2)

        loss = y*y - scale * th.log(c)
        return loss.mean()


def reduce(L, reduction):

    if reduction == 'mean':
        return L.mean()
    if reduction == 'sum':
        return L.sum()
    return L


class MaskedLoss(th.nn.Module):

    def __init__(self, loss, reduction='mean'):
        super().__init__()
        self.loss = loss
        self.reduction = reduction

    def forward(self, p, y):

        _y = 2*y - 1
        _y = _y * _y

        L = _y * self.loss(p, y)

        return reduce(L, self.reduction)


class HingeLoss(th.nn.Module):

    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.margin = margin

    def forward(self, p, y):

        y = 2*y - 1
        L = F.relu(self.margin - y * p)
        return reduce(L, self.reduction)


class HammingLoss(th.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, p, y):

        y = 2*y - 1
        p = th.tanh(p)

        sum = (p * y)

        L = 0.25*((sum - 1)**2)
        return reduce(L, self.reduction)


def select_loss(loss_type):

    if loss_type == 'rank_bce':
        return Rank_BCE(), True

    if loss_type == 'kendall':
        return Kendall_Loss(), True

    if loss_type == 'relational':
        return Relational_Log_Loss(), True

    if loss_type == 'hinge':
        return MaskedLoss(HingeLoss(reduction=None)), True

    raise ValueError("Unknown loss function %s." % loss_type)
