import torch
import torch.nn as nn
from torch.func import vmap
import torcheval
import torchmetrics
from torchmetrics import Metric
import numpy as np

class constant_array_metric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("r2_score", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("mse", default=torch.empty(0), dist_reduce_fx="cat")


    def update(self, pred, target):
        r2_score_metric = torchmetrics.R2Score()
        mse_metric = torchmetrics.MeanSquaredError()

        if isinstance(pred,torch.Tensor):
            bs = pred.shape[0]
            pred = pred.detach().cpu()
        elif isinstance(pred,list):
            bs = len(pred)
        elif isinstance(pred,np.ndarray):
            bs = pred.shape[0]
            pred = torch.tensor(pred)

        for i in range(bs):
            pred_b = torch.tensor(pred[i])
            target_b = torch.tensor(target[i])
            target_len = target_b.shape[0]
            pred_b = pred_b[:target_len]
            r2_score = r2_score_metric(pred_b,target_b)
            mse = mse_metric(pred_b,target_b)
            self.r2_score = torch.cat([self.r2_score,r2_score.unsqueeze(0)])
            self.mse = torch.cat([self.mse,mse.unsqueeze(0)])

    def compute(self,reduction = 'mean'):
        if reduction == 'mean':
            return {
                'r2_score':self.r2_score.mean(),
                'mse':self.mse.mean()
            }
        elif reduction == 'sum':
            return {
                'r2_score':self.r2_score.sum(),
                'mse':self.mse.sum()
            }
        elif reduction == 'none':
            return {
                'r2_score':self.r2_score,
                'mse':self.mse
            }
        else:
            raise NotImplementedError





