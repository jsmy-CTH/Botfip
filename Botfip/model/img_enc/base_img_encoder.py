import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from dataclasses import dataclass


class base_img_encoder(nn.Module):
    def __init__(self, config,*args,**kwargs):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls,model_config_path,*args,**kwargs):
        #with open(model_config_path, 'r') as file:
        #    m_dict = yaml.safe_load(file)
        m_dict = OmegaConf.load(model_config_path)

        model_config_dict= m_dict['img_encoder_config']

        return cls(model_config_dict,*args,**kwargs)

    #def __getattr__(self, item):
    #    if hasattr(self.config, item):
    #        return getattr(self.config, item)



@dataclass
class FuncimgEncOutput:
    last_hidden_state: torch.FloatTensor = None
    mlp_output: torch.FloatTensor = None
