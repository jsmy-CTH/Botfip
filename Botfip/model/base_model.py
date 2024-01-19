import logging
import os

import numpy as np
import torch
import torch.nn as nn

import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf

class base_encoder(nn.Module):
    def __init__(self, config,*args,**kwargs):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls,model_config_path,key=None,*args,**kwargs):
        #with open(model_config_path, 'r') as file:
        #    m_dict = yaml.safe_load(file)
        m_dict = OmegaConf.load(model_config_path)

        if key is not None:
            model_config_dict= m_dict[key]
        else:
            model_config_dict= m_dict

        return cls(model_config_dict,*args,**kwargs)

    #def __getattr__(self, item):
    #    if hasattr(self.config, item):
    #        return getattr(self.config, item)



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class BaseEncoder(nn.Module):
    """
    Base class for primitive encoders, such as ViT, TimeSformer, etc.
    """

    def __init__(self):
        super().__init__()

    def forward_features(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


class SharedQueueMixin:
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs=None):#,local_rank = None
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity


        # replace the keys at ptr (dequeue and enqueue)
        try:
            self.funcimg_queue[:, ptr: ptr + batch_size] = image_feats.T
            self.opseq_queue[:, ptr: ptr + batch_size] = text_feats.T

        except:
            print('batch_size:', batch_size)
            print('image_feats.T:', image_feats.T.shape)
            print('text_feats.T:', text_feats.T.shape)
            print('self.funcimg_queue:', self.funcimg_queue.shape)
            print('self.opseq_queue:', self.opseq_queue.shape)
            print('ptr:', ptr)
            raise ValueError('error')

        if idxs is not None:
            idxs = concat_all_gather(idxs)
            self.idx_queue[:, ptr: ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr


class MomentumDistilationMixin:
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                    model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                    model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                        1.0 - self.momentum
                )


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))
