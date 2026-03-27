from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(10)
torch.cuda.manual_seed(10)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
        
def calculate_berhu_loss(pred, gt, mask, weights):
    bs = pred.shape[0]
    diff = gt - pred
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5
    leq = (abs_diff <= c).float()
    l2_losses = (diff**2 + c**2) / (2 * c)
    loss = leq * abs_diff + (1 - leq) * l2_losses
    #_, c, __, ___ = loss.size()
    loss = loss.reshape(bs, -1)
    mask = mask.reshape(bs, -1)
    weights = weights.reshape(bs, -1)
    count = torch.sum(mask, dim=[1], keepdim=True).float()
    masked_loss = loss * mask.float()
    weighted_loss = masked_loss * weights
    return torch.mean(torch.sum(weighted_loss, dim=[1], keepdim=True) / count)

def calculate_l1_loss(pred, gt, mask):
    diff = gt - pred
    loss = torch.abs(diff)
    #_, c, __, ___ = loss.size()
    count = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    masked_loss = loss * mask.float()
    return torch.mean(torch.sum(masked_loss, dim=[1, 2, 3], keepdim=True) / count)
