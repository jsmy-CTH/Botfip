import sys
import os
from Botfip.trainer.botfip_trainer import Botfip_trainer
from omegaconf import OmegaConf
import os
import argparse
import torch
import torch.distributed as dist


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

if __name__ == '__main__':


    if_ddp = True

    if if_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    else:
        local_rank = None


    config_yaml = 'configs/model_hyper.yaml'
    config = OmegaConf.load(config_yaml)

    trainer = Botfip_trainer.build_from_yaml(config_yaml,
                                             dataset_key='op_dataset_config',
                                             train_config_key='botfip_train_parameters',
                                             if_ddp=if_ddp,
                                             local_rank=local_rank)


    trainer.train(train_type='all', noise_std=0.1)
