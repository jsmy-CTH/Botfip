from .operation_tree import OperationRandomTree
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from omegaconf import OmegaConf



def opt_generate(model_hyper_yaml,
              set_node_num = None,
              opseq = None,
              op_constant = None,):
        if opseq is None:
            assert set_node_num is not None, 'set_node_num should be given when opseq is None'
            opt = OperationRandomTree.from_config_yaml(set_node_num,model_hyper_yaml,backend='torch')
        else:
            if op_constant is not None:
                if isinstance(op_constant,list):
                    op_constant = torch.tensor(op_constant,dtype=torch.float32)
            opt = OperationRandomTree.load_tree_ots(opseq, model_hyper_yaml, constant_array=op_constant, backend='torch')
        return opt

class opt_model(nn.Module):
    def __init__(self,
                 opt,
                 config):
        super().__init__()

        self.opt = opt
        self.config = config

    @property
    def opseq(self):
        return self.opt.tree_serialized_encode_seq()[0]

    @property
    def constants(self):
        return self.opt.constants_array

    @classmethod
    def build(cls,
              model_hyper_yaml,
              opseq=None,
              op_constant=None,
              set_node_num = None,):

        config = OmegaConf.load(model_hyper_yaml)
        opt = opt_generate(model_hyper_yaml,
                                set_node_num = set_node_num,
                                opseq = opseq,
                                op_constant = op_constant,)
        return cls(opt,config)

    def reset_constants(self,op_constant=None):
        if op_constant is None:
            self.opt.random_assign_num_parameters()
        else:
            self.opt.constants_array = op_constant

    def update(self,opseq,op_constant = None):
        self.opt = opt_generate(self.config,
                                set_node_num = None,
                                opseq = opseq,
                                op_constant = op_constant,)
        return self.opt

    def forward(self, x):
        output = self.opt(x)
        return output

    def loss(self, x, y):
        output = self(x)
        loss = F.mse_loss(output, y)
        return loss







class opt_batch_model(nn.Module):
    def __init__(self,
                 opt_model_list,
                 config):
        super().__init__()

        for opt in opt_model_list:
            opt.check_node()
        self.opt_module_list = nn.ModuleList(opt_model_list)
        self.config = config

    @property
    def opseq(self):
        opseq_list = []
        for opt_model in self.opt_module_list:
            opseq_list.append(opt_model.opseq)
        return opseq_list



    @property
    def constants(self):
        constants_list = []
        for opt_model in self.opt_module_list:
            constants_list.append(opt_model.constants_array)
        return constants_list

    @classmethod
    def build(cls,
              model_hyper_yaml,
              opseq_batch,
              op_constant_batch = None,):

        config = OmegaConf.load(model_hyper_yaml)

        if isinstance(opseq_batch,list):
            bs = len(opseq_batch)
        else:
            bs = opseq_batch.shape[0]

        opt_model_list = []
        for i in range(bs):
            opt = opt_generate(model_hyper_yaml,
                            opseq = opseq_batch[i],
                            op_constant = op_constant_batch[i] if op_constant_batch is not None else None)
            opt_model_list.append(opt)

        return cls(opt_model_list,config)

    def update_num_parameters(self):
        for opt_model in self.opt_module_list:
            opt_model.update_num_parameters()
    def append(self,
               opseq,
               op_constant = None):

        opt = opt_model.build(self.config,
                                    opseq = opseq,
                                    op_constant = op_constant)
        self.opt_module_list.append(opt)

    def delete(self,index=None):
        if index is None:
            del self.opt_module_list[-1]
        else:
            del self.opt_module_list[index]

    def update(self,
               index,
               opseq,
               op_constant = None):
        self.opt_module_list[index].update(opseq,op_constant)

    def reset_constants(self, op_constant_batch=None):
        if op_constant_batch is None:
            for opt_model in self.opt_module_list:
                opt_model.reset_constants()
        else:
            for opt_model,op_constant in zip(self.opt_module_list,op_constant_batch):
                opt_model.reset_constants(op_constant)

    def forward(self, mesh_grids):
        output = []
        for opt_model in self.opt_module_list:
            output.append(opt_model(mesh_grids))
        output = torch.stack(output,dim=0)
        return output

    def loss(self, x, y):
        output = self(x)
        loss = F.mse_loss(output, y)
        return loss





