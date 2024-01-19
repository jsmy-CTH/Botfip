from ..operation.operation_tree import OperationRandomTree
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from ..operation.opt_model import opt_model,opt_batch_model
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from ..datagen.data_utils import *




class opt_lbfgs_finetune_trainer():
    def __init__(self,
                 config,
                 opt_batch_model,
                 mesh_dataset,
                 train_config_key='botfip_train_parameters',
                 dataset_key='op_dataset_config',
                 device = 'cuda',):

        self.opt_batch_model = opt_batch_model
        self.config = config
        self.mesh_dataset = mesh_dataset
        self.train_config_key = train_config_key
        self.dataset_key = dataset_key
        self.mesh_sampler = DataLoader(self.mesh_dataset,
                                       batch_size=config[train_config_key].sample_points_num,
                                       shuffle=True,
                                       drop_last=True)
        self.device = device

    @classmethod
    def build(cls,
              model_hyper_yaml,
              opseq,
              op_constant=None,
              train_config_key='botfip_train_parameters',
              dataset_key='op_dataset_config',
              device = 'cuda',
              img_range = (1,10),
              img_shape = 256):

        config = OmegaConf.load(model_hyper_yaml)
        opt_batch = opt_batch_model.build(model_hyper_yaml,opseq_batch=opseq, op_constant_batch= op_constant,).to(device)

        dataset_config = config[dataset_key]
        mesh_dataset = Multiscale_MeshDataset(sys_range = img_range,
                                              multiscale_channels = 1,
                                              max_var_types = config.operation_tree_config.max_var_types,
                                              img_shape = img_shape)

        return cls(config,opt_batch,mesh_dataset,train_config_key = train_config_key,dataset_key = dataset_key,device = device)

    def get_mesh_loss(self,opt_targer_model):
        var_num = self.mesh_dataset.mesh.size(-1)
        mesh_points = self.mesh_dataset.mesh.view(-1,var_num).to(self.device)
        with torch.no_grad():
            y_target = opt_targer_model(mesh_points)
        y_pred = self.opt_batch_model(mesh_points)
        loss = F.mse_loss(y_pred,y_target,reduction='mean')
        return loss


    def train(self,
              opt_targer_model,
              max_epoch=10000,
              reduction='mean',
              lr = 1,
              scheduler_steps = (2000,5000,10000,15000),
              gamma = 0.1,
              print_epochs = 100,
              ):

        self.opt_batch_model.train()
        opt_targer_model.to(self.device)
        optimizer = optim.AdamW(self.opt_batch_model.parameters(), lr=lr)
        if scheduler_steps is not None:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=gamma)
        else:
            scheduler = None

        var_num = self.mesh_dataset.mesh.size(-1)
        mesh_points = self.mesh_dataset.mesh.view(-1,var_num).to(self.device)
        with torch.no_grad():
            y_target = opt_targer_model(mesh_points)

        for i in range(max_epoch):
            self.opt_batch_model.update_num_parameters()
            optimizer.zero_grad()
            def closure():
                y_pred = self.opt_batch_model(mesh_points)
                inf_pred_mask = torch.isinf(y_pred)
                inf_target_mask = torch.isinf(y_target)
                inf_mask = inf_pred_mask | inf_target_mask
                nan_pred_mask = torch.isnan(y_pred)
                nan_target_mask = torch.isnan(y_target)
                nan_mask = nan_pred_mask | nan_target_mask
                mask = inf_mask | nan_mask
                loss = F.mse_loss(y_pred[~mask],y_target[~mask],reduction=reduction)
                loss.backward(retain_graph=True)

            optimizer.step(closure = closure)
            if scheduler_steps is not None:
                scheduler.step()













