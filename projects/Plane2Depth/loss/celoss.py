# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdepth.registry import MODELS

# from depth.models.builder import LOSSES

@MODELS.register_module()
class CELoss(nn.Module):
    """CrossEntropyLoss wrapper.
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 loss_weight=1.0):
        super(CELoss, self).__init__()
        self.loss_weight = loss_weight
        
    @torch.no_grad()
    def accuracy(self, output, target, topk=(1, 5, )):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self,
                input,
                target):
        """Forward function."""

        loss_ce = F.cross_entropy(input.squeeze(), target)
        acc = self.accuracy(input.squeeze(), target)
        loss_cls = self.loss_weight * loss_ce
        return loss_cls, acc
