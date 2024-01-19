import torch
from torch import nn
from Botfip.operation import OperationRegistrySet
from ...operation.operation_tree import *
from omegaconf import OmegaConf
import pandas as pd
from copy import deepcopy
from ..model_utils import *

def is_enclosed_by_brackets(s):
    return s.startswith('[') and s.endswith(']')

def is_token_property(key,value):
    return 'token' in key and isinstance(value,str) and is_enclosed_by_brackets(value)

def is_vec_op(op_dict):
    if 'index_necessary' not in op_dict:
        return False
    if op_dict['index_necessary']:
        return True
    return False

class op_tokenizer():
    def __init__(self,
                 operation_registry_set:OperationRegistrySet,
                 max_seq_length: int,
                 max_var_types: int,
                 max_constants_num: int,
                 cls_token:str="[CLS]",
                 sep_token:str="[SEP]",
                 mask_token:str="[MASK]",
                 def_token:str="[DEF]",
                 bos_token:str="[BOS]",
                 eos_token:str="[EOS]",
                 pad_token:str="[PAD]",
                 unk_token:str="[UNK]",
                 *args,**kwargs):

        self.operation_registry_set=operation_registry_set

        self.max_var_types = max_var_types
        self.max_seq_length = max_seq_length
        self.max_constants_num = max_constants_num

        self.pad_token = pad_token
        self.cls_token=cls_token
        self.def_token = def_token
        self.eos_token = eos_token
        self.sep_token=sep_token
        self.mask_token=mask_token
        self.bos_token=bos_token
        self.unk_token=unk_token


        self.op_start_index = None
        self.vocab = None

        if kwargs is not None:
            for k,v in kwargs.items():
                if isinstance(v,str) and is_enclosed_by_brackets(v):
                    setattr(self,k,v)
        self.init_vocab()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_config(cls,config,*args,**kwargs):
        operation_registry_set=OperationRegistrySet(config.operation_tree_config.operation_config_path)

        for k,v in config.opseq_tokenizer_config.items():
            if k == 'token_name':
                for k1,v1 in v.items():
                    kwargs[k1]=v1
            else:
                kwargs[k]=v

        for k,v in config.operation_tree_config.items():
            kwargs[k]=v

        return cls(operation_registry_set,*args,**kwargs)

    @classmethod
    def from_config_yaml(cls,model_config_path:str,*args,**kwargs):
        with open(model_config_path, 'r') as file:
            m_dict = yaml.safe_load(file)

        model_config_dict= m_dict['opseq_tokenizer_config']
        op_tree_config = m_dict['operation_tree_config']
        operation_registry_set = OperationRegistrySet.from_config_yaml(model_config_path)

        for k,v in model_config_dict.items():
            if isinstance(v,dict):
                for k1,v1 in v.items():
                    kwargs[k1]=v1
            else:
                kwargs[k]=v

        for k,v in op_tree_config.items():
            if isinstance(v,dict):
                for k1,v1 in v.items():
                    kwargs[k1]=v1
            else:
                kwargs[k]=v
        print(kwargs)

        return cls(operation_registry_set,*args,**kwargs)

    def init_vocab(self):
        vocab_df = pd.DataFrame(columns=['token','op_index','var_index','is_special'])
        for k,v in self.__dict__.items():
            if is_token_property(k,v):
                df = pd.DataFrame({
                    'token': v,
                    'op_index': None,
                    'var_index': None,
                    'is_special':True,
                },index=[0])
                vocab_df = pd.concat([vocab_df,df], ignore_index=True)

        self.op_start_index = len(vocab_df)

        for index,row in self.operation_registry_set.operation_info.iterrows():
            token = row['op_name']
            if row['vectorized'] and row['index_necessary']:
                for var in range(self.max_var_types):
                    vec_tocken_name = f"{token}_{var}"
                    op_index = self.operation_registry_set.op_info_get_index(token,var)
                    df = pd.DataFrame({
                        'token': vec_tocken_name,
                        'op_index': op_index,
                        'var_index': var,
                        'is_special': False,
                    },index=[0])
                    vocab_df = pd.concat([vocab_df,df], ignore_index=True)
            else:
                op_index = self.operation_registry_set.op_info_get_index(token)
                df = pd.DataFrame({
                    'token': token,
                    'op_index': op_index,
                    'var_index': None,
                    'is_special': False,
                },index=[0])
                vocab_df = pd.concat([vocab_df,df], ignore_index=True)
        self.vocab = vocab_df
        return vocab_df

    def add_special_tokens(self,token_dict):
        for k,v in token_dict.items():
            if isinstance(v,str) and is_enclosed_by_brackets(v):
                setattr(self,k,v)
                df = pd.DataFrame({
                    'token': v,
                    'op_index': None,
                    'var_index': None,
                    'is_special':True,
                },index=[0])
                self.vocab = pd.concat([self.vocab,df], ignore_index=True)

    def add_operation_tokens(self,operation_form_dict_list):
        for operation_form in operation_form_dict_list:
            self.operation_registry_set.operation_append(operation_form)
            token = operation_form['op_name']
            if is_vec_op(operation_form):
                for var in range(self.max_var_types):
                    vec_tocken_name = f"{token}_{var}"
                    op_index = self.operation_registry_set.op_info_get_index(token,var)
                    df = pd.DataFrame({
                        'token': vec_tocken_name,
                        'op_index': op_index,
                        'var_index': var,
                        'is_special': False,
                    },index=[0])
                    self.vocab = pd.concat([self.vocab ,df], ignore_index=True)
            else:
                op_index = self.operation_registry_set.op_info_get_index(token)
                df = pd.DataFrame({
                    'token': token,
                    'op_index': op_index,
                    'var_index': None,
                    'is_special': False,
                },index=[0])
                self.vocab = pd.concat([self.vocab ,df], ignore_index=True)

    def token2ind(self,token):
        return self.vocab[self.vocab['token']==token].index[0]

    def getid(self,token_name):
        token = eval(f"self.{token_name}_token")
        return self.token2ind(token)

    def random_generate_token(self,size):
        random_seq = torch.randint(0,len(self.vocab),size)
        random_constant = torch.rand(size)
        return random_seq,random_constant

    def generate_empty_token(self,start_token = 'cls'):
        empty_seq = torch.tensor([self.token2ind(self.pad_token),]*(self.max_seq_length+2)).int()
        start_token = eval(f"self.{start_token}_token")
        start_token_id = self.token2ind(start_token)
        empty_seq[0] = start_token_id
        empty_constant = torch.tensor([0,]*(self.max_constants_num),dtype=torch.float)
        seq_mask = torch.tensor([0,]*(self.max_seq_length+self.max_constants_num+2),dtype=torch.int)
        seq_mask[0] = 1
        return empty_seq[None],empty_constant[None],seq_mask[None]


    def tokenize(self,op_seq,const_array=None,device = 'cpu'):

        seq_templete = torch.tensor([self.token2ind(self.pad_token),]*(self.max_seq_length+2)).int()
        seq_templete[0] = self.token2ind(self.cls_token)

        const_templete = torch.tensor([0,]*(self.max_constants_num),dtype=torch.float)

        total_attention_mask_templete = torch.tensor([0,]*(self.max_seq_length+self.max_constants_num+2),dtype=torch.int)
        total_attention_mask_templete[0] = 1

        if isinstance(op_seq,torch.Tensor):
            bs = op_seq.shape[0]
        elif isinstance(op_seq,list):
            bs = len(op_seq)
        else:
            raise ValueError("op_seq must be torch.Tensor or list")

        if const_array is not None:
            if isinstance(const_array,torch.Tensor):
                cs = const_array.shape[0]
            elif isinstance(const_array,list):
                cs = len(const_array)
            else:
                raise ValueError("const_array must be torch.Tensor or list")
            assert cs == bs, "const_array must have the same batch size as op_seq"

        convert_seq_tensor_list = []
        convert_const_tensor_list = []
        total_attention_mask_list = []
        op_start_index = self.vocab.index[self.vocab['is_special']==False][0]

        for i in range(bs):
            seq_tensor = seq_templete.clone()
            total_attention_mask = total_attention_mask_templete.clone()

            current_op_seq_len = len(op_seq[i])
            current_index = 0

            assert current_op_seq_len <= self.max_seq_length, "op_seq is too long"
            if const_array is not None and const_array[i] is not None:
                current_const_num = len(const_array[i])
                assert  current_const_num <= self.max_constants_num, "const_array is too long"

            for j,op_index in enumerate(op_seq[i]):
                if op_index == 0:
                    seq_tensor[j + 1] = torch.tensor(self.token2ind(self.def_token), dtype=torch.int)
                else:
                    seq_tensor[j + 1] = torch.tensor(op_index + op_start_index - 1, dtype=torch.int)
                total_attention_mask[j + 1] = 1
            current_index = current_op_seq_len + 1
            seq_tensor[current_index] =  self.token2ind(self.eos_token)
            total_attention_mask[current_index] = 1
            convert_seq_tensor_list.append(seq_tensor)

            const_tensor = const_templete.clone()
            if const_array:
                if const_array[i] is not None:
                    current_constants_num = len(const_array[i])
                    total_attention_mask[self.max_seq_length+2] = 1
                    for j, const in enumerate(const_array[i]):
                        const_tensor[j] = const
                        total_attention_mask[self.max_seq_length+2+j] = 1


            convert_const_tensor_list.append(const_tensor)
            total_attention_mask_list.append(total_attention_mask)


        convert_seq_tensor = torch.stack(convert_seq_tensor_list,dim = 0)
        convert_const_tensor = torch.stack(convert_const_tensor_list, dim=0)
        total_attention_mask = torch.stack(total_attention_mask_list, dim=0)

        return convert_seq_tensor.to(device), convert_const_tensor.to(device),total_attention_mask.to(device)

    def detokenize(self,
                   seq_tensor,
                   const_tensor):


        if isinstance(seq_tensor,torch.Tensor):
            bs = seq_tensor.shape[0]
            seq_tensor = seq_tensor.detach().cpu().clone().numpy()
        elif isinstance(seq_tensor,list):
            bs = len(seq_tensor)
            seq_tensor = deepcopy(seq_tensor)
        elif isinstance(seq_tensor,np.ndarray):
            bs = seq_tensor.shape[0]
            seq_tensor = np.copy(seq_tensor)
        else:
            raise ValueError("seq_tensor must be torch.Tensor or list")

        if isinstance(const_tensor,torch.Tensor):
            const_tensor = const_tensor.detach().cpu().clone().numpy()
        else:
            const_tensor = deepcopy(const_tensor)

        seq_list = []
        const_list = []

        stop_id = self.getid('eos')
        def_id = self.getid('def')
        op_start_index = self.vocab.index[self.vocab['is_special'] == False][0]

        for i in range(bs):
            seq = np.array(seq_tensor[i])
            const = np.array(const_tensor[i])
            stop_array = np.where(seq == stop_id)[0]
            if len(stop_array) > 0:
                stop_index = stop_array[0]
                seq = seq[1:stop_index]
            else:
                seq = seq[1:]

            for j in range(len(seq)):
                if seq[j] == def_id:
                    seq[j] = 0
                else:
                    seq[j] = seq[j] - op_start_index + 1

            seq_list.append(seq.tolist())
            const_list.append(const)

        return seq_list,const_list






























