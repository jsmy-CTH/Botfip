import torch
import torch.nn as nn
from torch.func import vmap
import torcheval
import torchmetrics
from torchmetrics import Metric
import numpy as np



def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    if n > m:
        a,b = b,a
        n,m = m,n
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]

def relative_Levenshtein_batch_distance(pred,target,reduction = 'mean'):

    if isinstance(pred,torch.Tensor):
        pred = pred.detach().cpu()
        target = target.detach().cpu()
    target_len = torch.tensor([max(len(p),len(t)) for p,t in zip(pred,target)])
    ld = torch.tensor([levenshtein_distance(p,t) for p,t in zip(pred,target)])
    rld = ld/target_len

    if reduction == 'mean':
        return rld.mean()
    elif reduction == 'sum':
        return rld.sum()
    elif reduction == 'none':
        return rld
    else:
        raise NotImplementedError


class relative_Levenshtein(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("rld", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, pred, target):
        rld = 1 - relative_Levenshtein_batch_distance(pred,target,reduction = 'none')
        self.rld = torch.cat([self.rld,rld])

    def compute(self,reduction = 'mean'):
        if reduction == 'mean':
            return self.rld.mean()
        elif reduction == 'sum':
            return self.rld.sum()
        elif reduction == 'none':
            return self.rld
        else:
            raise NotImplementedError













