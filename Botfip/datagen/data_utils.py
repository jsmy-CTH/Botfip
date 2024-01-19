import os
import torch
import numpy as np
from torch import nn
from functorch import vmap
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from  torch.utils.data import Dataset, DataLoader
from ..common.utils import input_normalization
from ..operation.operation_tree import OperationRandomTree
from dataclasses import dataclass
from ..common.utils import *

@dataclass
class TrainBatchStruct:
    funcimg: torch.Tensor
    funcimg_max: torch.Tensor
    funcimg_min: torch.Tensor
    opseq: tuple[list[list[int]],list[list[float]]]
    set_node_num: int

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)





def ind2point(mesh, index):
    meshpoints = mesh[tuple(index.T)]
    return meshpoints

def v_ind2point(mesh,indices):
    return vmap(ind2point)(mesh,repeat(indices,'...-> b  ...',b=mesh.shape[0]))



def mesh_generate(sys_range_list,shape_list):
    mesh_list = [torch.linspace(sys_range_list[i,0], sys_range_list[i,1], shape_list[i]) for i in range(len(shape_list))]
    mesh = torch.meshgrid(*mesh_list)
    mesh = torch.stack(mesh, dim=-1)
    return mesh

def multiscale_mesh_generate(multiscale_channels,sys_range,img_shape,max_var_types):
    mesh_gird_list = []
    mesh_ind_list = []
    for channel_num, i in enumerate(range(- (multiscale_channels // 2), multiscale_channels // 2 + 1)):
        range_array = torch.linspace(sys_range[0] * 10 ** i, sys_range[1] * 10 ** i, img_shape)
        ind_array = torch.arange(img_shape, dtype=torch.int)
        mesh_grid = torch.meshgrid(*([range_array, ] *max_var_types))
        mesh_ind = torch.meshgrid(*([ind_array, ] * max_var_types))

        channel_ind_tensor = torch.ones_like(mesh_grid[0], dtype=torch.int) * channel_num
        mesh_ind = (channel_ind_tensor,) + mesh_ind

        mesh_grid = torch.stack(mesh_grid, dim=-1)
        mesh_ind = torch.stack(mesh_ind, dim=-1)
        mesh_gird_list.append(mesh_grid)
        mesh_ind_list.append(mesh_ind)

    mesh = torch.stack(mesh_gird_list, dim=0)
    mesh_index = torch.stack(mesh_ind_list, dim=0)
    return mesh,mesh_index


def custom_collate(batch):
    # 直接将不定长的数据转化为列表
    tree_seq_op = [item['opseq'][0] for item in batch]
    const_array = [item['opseq'][1] for item in batch]

    # 对于其他可以直接转换为tensor的数据
    funcimg = torch.stack([item['funcimg'] for item in batch])
    original_funcimg = torch.stack([item['original_funcimg'] for item in batch])
    funcimg_max = torch.stack([item['funcimg_max'] for item in batch])
    funcimg_min = torch.stack([item['funcimg_min'] for item in batch])
    set_node_num = torch.tensor([item['set_node_num'] for item in batch])
    img_index = torch.tensor([item['img_index'] for item in batch])
    skeleton_index = torch.tensor([item['skeleton_index'] for item in batch])

    return {
        'funcimg': funcimg,
        'original_funcimg': original_funcimg,
        'funcimg_max': funcimg_max,
        'funcimg_min': funcimg_min,
        'opseq': (tree_seq_op,const_array),
        'set_node_num': set_node_num,
        'img_index': img_index,
        'skeleton_index': skeleton_index
    }



def formula_df_to_optree_list(formula_df,operation_registry,config,backend='numpy'):
    #operation_registry = self.model.tokenizer.operation_registry_set
    optree_list = []
    tree_op_seq_list = []
    for i in range(len(formula_df)):
        tree_op_seq = str2list(formula_df.iloc[i]['tree_op_seq'])
        tree_op_seq_list.append(tree_op_seq.copy())
        set_node_num = int(formula_df.iloc[i]['set_node_num'])
        op_tree_config = config.operation_tree_config

        optree = OperationRandomTree(num_nodes=set_node_num,
                                     config=op_tree_config,
                                     operation_registry_set=operation_registry,
                                     backend= backend)
        encoder_vector_dict = {
            'tree_op_seq': tree_op_seq,
            'constants_array': None,
        }

        optree.load_tree(encoder_vector_dict)
        optree_list.append(optree)


    optree_list = list(zip(tree_op_seq_list,optree_list))
    return optree_list





def funcimg_transform(funcimg,
                        img_range = (-1,1),
                        nan_replace = 0.,
                        ):
    masked_img = np.ma.masked_invalid(funcimg)
    funcimg_max = np.max(masked_img, axis=(1, 2))
    funcimg_min = np.min(masked_img, axis=(1, 2))
    funcimg = (2*funcimg - (funcimg_max[:,None,None] + funcimg_min[:,None,None]))/  (funcimg_max[:,None,None] - funcimg_min[:,None,None])

    funcimg[np.isnan(funcimg) | np.isinf(funcimg)] = nan_replace

    funcimg = img_range[0] + (img_range[1] - img_range[0]) * (funcimg - np.min(funcimg)) / (np.max(funcimg) - np.min(funcimg))
    return funcimg,funcimg_max,funcimg_min

#%%
def funcimg_recon(batch_normal_funcimg,funcimg_max,funcimg_min):
    bs = batch_normal_funcimg.shape[0]
    channel = batch_normal_funcimg.shape[1]
    assert channel == funcimg_max.shape[1] == funcimg_min.shape[1], 'channel not match'
    img_shape = batch_normal_funcimg.shape[2:]
    current_funcimg_max = torch.amax(batch_normal_funcimg,dim=(2,3))
    current_funcimg_min = torch.amin(batch_normal_funcimg,dim=(2,3))
    normal_funcimg = (batch_normal_funcimg - current_funcimg_min[...,None,None]) / (current_funcimg_max[...,None,None] - current_funcimg_min[...,None,None]) * 2 - 1
    affine_funcimg =  normal_funcimg * (funcimg_max[...,None,None] - funcimg_min[...,None,None]) / 2 + (funcimg_max[...,None,None] + funcimg_min[...,None,None]) / 2
    return affine_funcimg



class MeshDataset(Dataset):
    def __init__(self,
                 sys_range,
                 space_shape=100,
                 time_channels=10,
                 **kwargs):
        super(MeshDataset, self).__init__()

        self.sys_range = torch.Tensor(sys_range)
        self.sys_dim = self.sys_range.shape[0]
        self.time_channels = time_channels
        self.space_shape = space_shape
        self.mesh,self.mesh_index = self.mesh_generate()
        self.mesh_vec = self.mesh.reshape(-1,self.sys_dim)
        self.mesh_index = self.mesh_index.reshape(-1,self.sys_dim)

    def mesh_generate(self):
        time_list = torch.linspace(self.sys_range[0,0], self.sys_range[0,1], self.time_channels)
        time_index_list = torch.arange(self.time_channels)
        space_list = [torch.linspace(self.sys_range[i,0], self.sys_range[i,1], self.space_shape) for i in range(1,self.sys_dim)]
        space_index_list = [torch.arange(self.space_shape) for i in range(1,self.sys_dim)]
        mesh = torch.meshgrid(time_list,*space_list)
        mesh_index = torch.meshgrid(time_index_list,*space_index_list)
        mesh = torch.stack(mesh, dim=-1)
        mesh_index = torch.stack(mesh_index, dim=-1)
        return mesh[None],mesh_index

    def __len__(self):
        return self.mesh_vec.shape[0]

    def __getitem__(self, idx):
        return self.mesh_vec[idx],self.mesh_index[idx]


class Multiscale_MeshDataset(Dataset):
    def __init__(self,
                 sys_range,
                 multiscale_channels=3,
                 max_var_types = 2,
                 img_shape=128,
                 **kwargs):
        super().__init__()


        self.multiscale_channels = multiscale_channels
        self.img_shape = img_shape
        self.max_var_types = max_var_types
        self.sys_range = torch.Tensor(sys_range)

        self.mesh,self.mesh_index = self.mesh_generate()

        self.mesh_vec = self.mesh.reshape(-1,self.max_var_types)
        self.mesh_index = self.mesh_index.reshape(-1,len(self.mesh.shape)-1)

    def mesh_generate(self):
        mesh_gird_list = []
        mesh_ind_list = []

        for channel_num, i in enumerate(range(- (self.multiscale_channels//2), self.multiscale_channels//2+1)):
            range_array = torch.linspace(self.sys_range[0]*10**i, self.sys_range[1] * 10**i, self.img_shape)
            ind_array = torch.arange(self.img_shape,dtype=torch.int)
            mesh_grid = torch.meshgrid(*([range_array, ] * self.max_var_types))
            mesh_ind = torch.meshgrid(*([ind_array, ] * self.max_var_types))

            channel_ind_tensor = torch.ones_like(mesh_grid[0],dtype=torch.int) * channel_num
            mesh_ind = (channel_ind_tensor,) + mesh_ind

            mesh_grid = torch.stack(mesh_grid, dim=-1)
            mesh_ind = torch.stack(mesh_ind, dim=-1)
            mesh_gird_list.append(mesh_grid)
            mesh_ind_list.append(mesh_ind)

        mesh = torch.stack(mesh_gird_list, dim = 0)
        mesh_index = torch.stack(mesh_ind_list, dim = 0)
        return mesh,mesh_index

    def __len__(self):
        return self.mesh_vec.shape[0]

    def __getitem__(self, idx):
        return self.mesh_vec[idx],self.mesh_index[idx]



def rand_coordinate_points_gen(sample_region,samples_shape):
    samples = torch.rand(*samples_shape)
    samples = samples * (sample_region[1] - sample_region[0]) + sample_region[0]
    return samples












class FuncImgDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = os.listdir(path)

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        根据索引返回一个样本

        参数：
            - index (int): 样本的索引

        返回：
            - sample (torch.Tensor): 一个样本的 Torch 张量
        """
        file_name = self.file_list[index]
        file_path = os.path.join(self.path, file_name)
        data = np.load(file_path)
        sample = torch.from_numpy(data)
        return sample


class FuncImgNormDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = os.listdir(path)

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        根据索引返回一个样本

        参数：
            - index (int): 样本的索引

        返回：
            - sample (torch.Tensor): 一个样本的 Torch 张量
        """
        file_name = self.file_list[index]
        file_path = os.path.join(self.path, file_name)
        data = np.load(file_path)
        sample = torch.from_numpy(data)
        sample_max =  torch.max(sample,dim=1,keepdim= True).values
        sample_min = torch.min(sample, dim=1, keepdim=True).values
        sample_max = torch.max(sample_max,dim=2,keepdim= True).values
        sample_min = torch.min(sample_min, dim=2, keepdim=True).values
        sample = (sample - sample_min)/(sample_max - sample_min + 1e-10)
        return sample

class FuncImgALLNormDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = os.listdir(path)

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        根据索引返回一个样本

        参数：
            - index (int): 样本的索引

        返回：
            - sample (torch.Tensor): 一个样本的 Torch 张量
        """
        file_name = self.file_list[index]
        file_path = os.path.join(self.path, file_name)
        data = np.load(file_path)
        input_tensor = torch.from_numpy(data)
        input_max = torch.func.vmap(torch.max)(input_tensor)
        input_min = torch.func.vmap(torch.min)(input_tensor)
        input_tensor = (input_tensor - input_min.view(-1,*([1]*(len(input_tensor.shape)-1)))) / (input_max.view(-1,*([1]*(len(input_tensor.shape)-1))) - input_min.view(-1,*([1]*(len(input_tensor.shape)-1))))
        return input_tensor