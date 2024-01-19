import torch
import torch.nn as nn
from torch.func import vmap
import torcheval
import torchmetrics
from torchmetrics import Metric
import numpy as np
from .levenshtein import levenshtein_distance
from ..operation.operation_tree import OperationRandomTree


class opstr_levenshtein(Metric):
    def __init__(self,hyper_yaml):
        super().__init__()
        self.hyper_yaml = hyper_yaml
        self.add_state("rld", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, pred_ots, target_ots):

        if isinstance(pred_ots, torch.Tensor or np.ndarray):
            bs = pred_ots.shape[0]
        elif isinstance(pred_ots, list):
            bs = len(pred_ots)
        else:
            raise NotImplementedError

        rld_list = []
        for i in range(bs):
            pred = pred_ots[i]
            target = target_ots[i]
            if isinstance(pred,torch.Tensor):
                pred = pred.detach().cpu().tolist()
                target = target.detach().cpu().tolist()
            elif isinstance(pred,np.ndarray):
                pred = pred.tolist()
                target = target.tolist()
            try:
                pred_opt = OperationRandomTree.load_tree_ots(pred,self.hyper_yaml)
                target_opt = OperationRandomTree.load_tree_ots(target,self.hyper_yaml)
                pred_str = pred_opt.to_formula_str(type='simplified')
                target_str = target_opt.to_formula_str(type='simplified')
                rld = 1 - levenshtein_distance(pred_str, target_str) / max(len(pred_str), len(target_str))
            except:
                rld = 0
            rld_list.append(rld)
        self.rld = torch.cat([self.rld,torch.tensor(rld_list)])

    def compute(self,reduction = 'mean'):
        if reduction == 'mean':
            return self.rld.mean()
        elif reduction == 'sum':
            return self.rld.sum()
        elif reduction == 'none':
            return self.rld
        else:
            raise NotImplementedError











    def compute(self,reduction = 'mean'):
        if reduction == 'mean':
            return self.rld.mean()
        elif reduction == 'sum':
            return self.rld.sum()
        elif reduction == 'none':
            return self.rld
        else:
            raise NotImplementedError

