import torch
import torch.nn as nn
from torch.func import vmap
import torcheval
import torchmetrics
from torchmetrics import Metric
import numpy as np
import math



class ots_tolerance_acc(Metric):
    def __init__(self,
                 tolerance_degree=0.2,
                 max_tolerance_threshold = 3):

        super().__init__()
        self.tolerance_degree = tolerance_degree
        self.max_tolerance_threshold = max_tolerance_threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_ots,target_ots):


        if isinstance(pred_ots, torch.Tensor or np.ndarray):
            bs = pred_ots.shape[0]
        elif isinstance(pred_ots, list):
            bs = len(pred_ots)

        for i in range(bs):
            if isinstance(pred_ots[i], torch.Tensor):
                pred_ots[i] = pred_ots[i].detach().cpu().tolist()
            elif isinstance(pred_ots[i], np.ndarray):
                pred_ots[i] = pred_ots[i].tolist()

            if isinstance(target_ots[i], torch.Tensor):
                target_ots[i] = target_ots[i].detach().cpu().tolist()
            elif isinstance(target_ots[i], np.ndarray):
                target_ots[i] = target_ots[i].tolist()

            nq_index_num = 0
            len_target_ots = len(target_ots[i])
            len_pred_ots = len(pred_ots[i])
            tolerance_threshold = min(math.ceil(self.tolerance_degree * len_target_ots), self.max_tolerance_threshold)
            test_len = min(len_target_ots,len_pred_ots)

            for ind in range(test_len):
                if target_ots[i][ind] != pred_ots[i][ind]:
                    nq_index_num += 1
                if nq_index_num > tolerance_threshold:
                    break
            if nq_index_num <= tolerance_threshold:
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total.float()
