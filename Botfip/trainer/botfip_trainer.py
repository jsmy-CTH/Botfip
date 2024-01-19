import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,DistributedSampler
import wandb
from omegaconf import OmegaConf
from ..model.blip.blip_pretrain import BlipPretrain
from ..datagen.opt_datagen import Op_dataset
from ..datagen.data_utils import *
from ..operation.operation_tree import OperationRandomTree
import os
from typing import List,Dict,Tuple
from ..common.schedule import *
from ..common.utils import *
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..metric import *
from einops import rearrange,repeat
from tqdm import tqdm
from .opt_finetune_trainer import opt_lbfgs_finetune_trainer
from ..operation.opt_model import opt_model,opt_batch_model


class Botfip_trainer():
    def __init__(self,
                 config,
                 botfip_model,
                 op_dataset,
                 mesh_dataset,
                 train_config_key = 'botfip_train_parameters',
                 dataset_key = 'op_dataset_config',
                 if_ddp = True,
                 local_rank = -1,
                 config_yaml_path = None,
                 set_node_num = None,
                 if_compute_coded_constant = False,):

        self.train_config = config[train_config_key]
        self.dataset_config = config[dataset_key]
        self.config = config
        self.if_ddp = if_ddp
        self.local_rank = local_rank
        self.config_yaml_path = config_yaml_path
        self.set_node_num = set_node_num
        self.if_compute_coded_constant = if_compute_coded_constant

        if if_ddp:
            device = torch.device("cuda", local_rank)
        else:
            device = self.train_config.device

        self.device = device

        if not os.path.exists(self.train_config.model_save_path):
            os.makedirs(self.train_config.model_save_path)



        self.model = botfip_model.to(device)
        self.op_dataset = op_dataset
        self.mesh_dataset = mesh_dataset

        self.mesh_sampler = DataLoader(self.mesh_dataset,batch_size=self.train_config.sample_points_num,shuffle=True,drop_last=True)

        if if_ddp:
            self.model = DDP(self.model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
            self.op_sampler = DistributedSampler(self.op_dataset)
            self.op_dataloader = DataLoader(self.op_dataset, batch_size=self.train_config.batch_size,
                                            collate_fn=custom_collate, shuffle=False, drop_last=True,
                                            num_workers=self.train_config.num_workers,sampler= self.op_sampler)
        else:
            self.op_dataloader = DataLoader(self.op_dataset, batch_size=self.train_config.batch_size,
                                            collate_fn=custom_collate, shuffle=True, drop_last=True,
                                            num_workers=self.train_config.num_workers)


    @classmethod
    def build_from_yaml(cls,
                        config_yaml,
                        train_config_key='botfip_train_parameters',
                        dataset_key='op_dataset_config',
                        blip_parameters_key = 'blip_pretrain_config',
                        set_node_num = None,
                        load_path = None,
                        if_ddp = True,
                        local_rank = -1,):

        config = OmegaConf.load(config_yaml)
        dataset_config = config[dataset_key]
        op_dataset = Op_dataset(config_yaml,if_load=True,dataset_key=dataset_key,set_node_num=set_node_num)
        print('op_dataset len:',len(op_dataset))

        mesh_dataset = Multiscale_MeshDataset(sys_range = dataset_config.img_range,
                                              multiscale_channels = dataset_config.multiscale,
                                              max_var_types = config.operation_tree_config.max_var_types,
                                              img_shape = dataset_config.img_shape)
        botfip_model =  BlipPretrain.from_config(config,train_parameters_key = blip_parameters_key)


        if load_path is not None:
            pretrain_weight = torch.load(load_path, map_location=torch.device('cpu'))
            botfip_model.load_state_dict(pretrain_weight)
            print('load model from {}'.format(load_path))
            botfip_model.reset_queue_ptr()
            print('botfip_model.queue_ptr:', botfip_model.queue_ptr)



        return cls(config, botfip_model, op_dataset, mesh_dataset,train_config_key,dataset_key,if_ddp=if_ddp,local_rank=local_rank,config_yaml_path = config_yaml,set_node_num=set_node_num)



    def validation(self,
                   val_dataloader,
                   test_num_limit = 1000,
                   if_cal_mse = False,
                   if_bfgs_finetune = False,
                   noise_std = None,
                   train_config_key='botfip_train_parameters',
                   dataset_key='op_dataset_config',
                   device='cuda',
                   img_range=(1, 10),
                   img_shape=256,
                   max_epoch=7000,
                   reduction='mean',
                   lr=1,
                   scheduler_steps=(2000, 5000,),
                   gamma=0.1,
                   print_epochs=100,):

        self.model.eval()
        rl_metricer = relative_Levenshtein()
        ots_rationality_metricer = ots_rationality_acc(self.config_yaml_path)
        opstr_levenshtein_metricer = opstr_levenshtein(self.config_yaml_path)
        constant_metricer = constant_array_metric()
        mse_metricer_original = []
        mse_metricer_bfgs = []


        test_num = 0
        metric_value = {}

        for i, samples in tqdm(enumerate(val_dataloader)):

            bs = samples['funcimg'].shape[0]
            with torch.no_grad():
                funcimg = samples['funcimg'].to(self.device)
                if noise_std is not None:
                    funcimg += torch.randn_like(funcimg) * noise_std
                opseq_target, op_constants_target = samples["opseq"]
                opseq_pred, op_constants_pred = self.model.opseq_generate(funcimg, device=self.device,)
                opseq_pred, op_constants_pred = self.model.tokenizer.detokenize(opseq_pred, op_constants_pred)

                rl_metricer.update(opseq_pred,opseq_target)
                ots_rationality_metricer.update(opseq_pred)
                opstr_levenshtein_metricer.update(opseq_pred,opseq_target)
            if if_cal_mse:
                opt_batch_pred_model = opt_lbfgs_finetune_trainer.build(self.config_yaml_path ,opseq_pred,op_constant= op_constants_pred,device=device,
                                                                        train_config_key=train_config_key,
                                                                        dataset_key=dataset_key,
                                                                        img_range=img_range,
                                                                        img_shape=img_shape)

                opt_target_model = opt_batch_model.build(self.config_yaml_path,opseq_batch=opseq_target,op_constant_batch = op_constants_target)
                original_mse = opt_batch_pred_model.get_mesh_loss(opt_target_model)
                mse_metricer_original.append(original_mse.detach().cpu())
                print('original_mse:',original_mse)
                if if_bfgs_finetune:
                    opt_batch_pred_model.train(opt_target_model,
                                               max_epoch=max_epoch,
                                               reduction=reduction,
                                               lr=lr,
                                               scheduler_steps=scheduler_steps,
                                               gamma=gamma,
                                               print_epochs=print_epochs,)
                    bfgs_mse = opt_batch_pred_model.get_mesh_loss(opt_target_model)
                    mse_metricer_bfgs.append(bfgs_mse.detach().cpu())
                    print('bfgs_mse:',bfgs_mse)

            test_num += funcimg.shape[0]
            if test_num > test_num_limit:
                break

        metric_value['rl_metric'] = rl_metricer.compute()
        metric_value['ots_rationality_metric'] = ots_rationality_metricer.compute()
        metric_value['opstr_levenshtein_metric'] = opstr_levenshtein_metricer.compute()

        if if_cal_mse:
            metric_value['mse_metric_original'] = torch.mean(torch.tensor(mse_metricer_original))
            if if_bfgs_finetune:
                metric_value['mse_metric_bfgs'] = torch.mean(torch.tensor(mse_metricer_bfgs))
        return metric_value



    def train(self, train_type = 'pretrain',noise_std=None):
        if train_type == 'pretrain':
            cal_itc_itm_loss = True
            cal_lm_loss = False
            encoder_freeze = False
        elif train_type == 'lm_task_finetune':
            cal_itc_itm_loss = False
            cal_lm_loss = True
            encoder_freeze = True
        elif train_type == 'all':
            cal_itc_itm_loss = True
            cal_lm_loss = True
            encoder_freeze = False
        else:
            raise ValueError('train_type error')

        if noise_std is not None:
            train_type = train_type + '_noise_std_' + str(noise_std)

        train_config = self.train_config
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.init_lr, weight_decay=train_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config.step_size, gamma=train_config.gamma)

        if (not self.if_ddp or dist.get_rank() == 0):
            nowtime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            project_name = train_config.wandb_project_name+'_'+train_type
            name = train_config.wandb_test_name + f'_{train_type}_{nowtime}'

            if self.set_node_num is not None:
                name += f'_node{self.set_node_num}'
            wandb.init(project=project_name,
                       entity=train_config.wandb_entity_name,
                       name=name)

            total_save_path = os.path.join(train_config.model_save_path, name)
            if not os.path.exists(total_save_path):
                os.makedirs(total_save_path)
            OmegaConf.save(self.config, os.path.join(total_save_path, 'config.yaml'))
        else:
            total_save_path = None

        step = 0
        
        for epoch in range(train_config.epoch_num):

            header = f'{train_type}  Funcimg-Opseq Epoch: [{epoch}]'

            for i,samples in enumerate(self.op_dataloader):

                if step < train_config.warmup_steps:
                    warmup_lr_schedule(optimizer, step, train_config['warmup_steps'], train_config['warmup_lr'],
                                       train_config['init_lr'])
                samples['epoch'] = epoch
                samples['iters'] = i
                samples['num_iters_per_epoch'] = len(self.op_dataloader)
                if noise_std is not None:
                    samples['funcimg'] = samples['funcimg'] + torch.randn_like(samples['funcimg'])*noise_std
                samples['funcimg'] = samples['funcimg'].to(train_config.device)

                output = self.model(samples,
                                    cal_itc_itm_loss=cal_itc_itm_loss,
                                    cal_lm_loss=cal_lm_loss,
                                    encoder_freeze=encoder_freeze,)

                lr = optimizer.param_groups[0]['lr']
                if not self.if_ddp or dist.get_rank() == 0:
                    if self.if_ddp:
                        temp = self.model.module.temp.item()
                    else:
                        temp = self.model.temp.item()
                    wandb.log({'total loss': output.loss.item(),
                               'loss_itc': output.loss_itc.item() if cal_itc_itm_loss else 0,
                               'loss_itm': output.loss_itm.item() if cal_itc_itm_loss else 0,
                               'loss_lm': output.loss_lm.item() if cal_lm_loss else 0,
                               'lr': lr,
                               'temp': temp,
                               })
                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                if (not self.if_ddp or dist.get_rank() == 0) and i % train_config.print_step == 0:
                    print(header + ' Iter: [{}]/[{}], '.format(i, len(self.op_dataloader)) +
                          ' Loss: {:.4f}'.format(output.loss.item())+
                          ' Loss_itc: {:.4f}'.format(output.loss_itc.item() if cal_itc_itm_loss else 0)+
                            ' Loss_itm: {:.4f}'.format(output.loss_itm.item() if cal_itc_itm_loss else 0)+
                            ' Loss_lm: {:.4f}'.format(output.loss_lm.item() if cal_lm_loss else 0))
                step += 1

            if step > train_config.warmup_steps + train_config.init_steps:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'], train_config.min_lr)

            if (not self.if_ddp or dist.get_rank() == 0) and epoch % train_config.save_epoch == 0:
                save_name = f'botfip_{train_type}_epoch_{epoch}'
                save_name = os.path.join(total_save_path, save_name)
                if not self.if_ddp:
                    self.model.save_models(save_name)
                else:
                    self.model.module.save_models(save_name)




































































