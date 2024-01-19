import torch
import torch.nn as nn
from torch.func import vmap
import torcheval
import torchmetrics
from torchmetrics import Metric
import numpy as np
from ..operation.operation_tree import OperationRandomTree


class ots_rationality_acc(Metric):
    def __init__(self,hyper_yaml):
        super().__init__()
        self.hyper_yaml = hyper_yaml
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_ots):
        if isinstance(pred_ots, torch.Tensor or np.ndarray):
            bs = pred_ots.shape[0]
        elif isinstance(pred_ots, list):
            bs = len(pred_ots)
        for i in range(bs):
            if isinstance(pred_ots[i], torch.Tensor):
                pred_ots[i] = pred_ots[i].detach().cpu().tolist()
            elif isinstance(pred_ots[i], np.ndarray):
                pred_ots[i] = pred_ots[i].tolist()
            try:
                opt = OperationRandomTree.load_tree_ots(pred_ots[i],self.hyper_yaml)
                opt.check_node()
                inputs = opt.variable_symbols
                func = opt.func_iteration(0)
                func(inputs, if_skeleton=True)
                self.correct += 1
            except Exception as e:
                pass
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total.float()


