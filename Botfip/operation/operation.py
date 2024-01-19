import numpy as np
import sympy as sp
import torch
from torch import nn
import yaml
import pandas as pd
from .operation_func import *
#import operation_func
import types
import os
from omegaconf import OmegaConf

class Operation():
    def __init__(self,
                 op_name:str,
                 op_type:str,
                 op_np_func,
                 op_torch_func,
                 op_sp_func,
                 op_symbol:str=None,
                 is_test_sp:bool = True,
                 is_function_operator:bool = False,
                 constant_chosen=None,
                 constant_num:int = None,
                 repeat_times:int = None,
                 generation_level:str=None,
                 if_test:bool = True,
                 vectorized:bool = False,
                 choisen_index:int = None,
                 height_limit:int = None,
                 if_skeleton_output:bool = False,
                 index_necessary:bool = False,
                 sp_start_index: int = 0,
                 domain:str = None,
                 num_backend = 'numpy', # numpy, torch
                 adjacent_repeatable:bool = None,
                 constant_range:str = None,
                 ):

        #super().__init__()
        self.op_name = op_name
        self.op_type = op_type
        self.num_backend = num_backend
        if  self.num_backend == 'numpy':
            if 'np_'  in op_np_func:
                self.op_num_func = eval(op_np_func)
            else:
                self.op_num_func = eval('np.'+op_np_func)

        elif self.num_backend == 'torch':
            if 'torch_'  in op_torch_func:
                self.op_num_func = eval(op_torch_func)
            else:
                self.op_num_func = eval('torch.'+op_torch_func)

        if 'sp_'  in op_sp_func:
            self.op_sp_func = eval(op_sp_func)
        else:
            self.op_sp_func = eval('sp.'+op_sp_func)

        self.op_symbol = op_symbol
        self.sp_start_index = sp_start_index
        self.height_limit = height_limit
        self.index_necessary = index_necessary
        self.repeat_times = int(repeat_times) if (repeat_times!= None and repeat_times!= 'Never') else repeat_times
        self.if_skeleton_output = if_skeleton_output
        self.domain = domain
        self.is_function_operator = is_function_operator
        self.generation_level = generation_level
        self.is_test_sp = is_test_sp
        self.constant_num = int(constant_num) if constant_num!=None else None
        self.constant_chosen = constant_chosen
        self.adjacent_repeatable = adjacent_repeatable
        self.constant_range = constant_range



        self.vectorized = vectorized

        if self.vectorized and self.index_necessary:
            self.choisen_index = choisen_index if choisen_index!=None else 0
            self.op_name = self.op_name + f'_{self.choisen_index}'
            self.op_symbol = self.op_symbol + f'_{self.choisen_index}'

        if constant_num != None:
            if self.constant_chosen==None:
                self.const_parameters_index = [i for i in range(self.sp_start_index,self.sp_start_index+constant_num)]
                symbols_created = sp.symbols(' '.join([f'C_{self.sp_start_index+i}' for i in range(constant_num)]))# cls=sp.Dummy
                #print('symbols_created:',symbols_created)
                if isinstance(symbols_created, (tuple, list)):
                    self.sp_parameters = symbols_created
                else:
                    self.sp_parameters = [symbols_created]
                if self.num_backend == 'numpy':
                    self.num_parameters = np.random.rand(constant_num)
                elif self.num_backend == 'torch':
                    self.num_parameters = torch.rand(constant_num)
            else:
                if self.num_backend == 'numpy':
                    parameters = np.random.choice(self.constant_chosen,constant_num)
                elif self.num_backend == 'torch':
                    parameters = torch.tensor(np.random.choice(self.constant_chosen,constant_num))
                    
                self.sp_parameters = sp.symbols(' '.join(list(parameters)))
                self.num_parameters = parameters

        if if_test:
            self._test_feasibility()

    def _test_feasibility(self):
        assert self.op_type in ['unary','binary'], 'operation type must be unary or binary!'
        eq_Flag = True
        if self.op_type == 'unary':
            sp_inputs = sp.symbols('x')
            if self.num_backend == 'numpy':
                num_inputs = np.random.rand(1)
            elif self.num_backend == 'torch':
                num_inputs = torch.rand(1)
        else:
            sp_inputs = sp.symbols('x y')
            if self.num_backend == 'numpy':
                num_inputs = [np.random.rand(1),np.random.rand(1)]
            elif self.num_backend == 'torch':
                num_inputs = [torch.rand(1),torch.rand(1)]

        kwargs = {}
        if self.is_function_operator:
            if self.num_backend == 'numpy':
                func = np.sin
            elif self.num_backend == 'torch':
                func = torch.sin
            kwargs['func']=func

        if self.vectorized:
            kwargs['index'] = self.choisen_index

        num_outputs = self.forward(num_inputs,**kwargs)

        if self.is_function_operator:
            func = sp.sin
            kwargs['func']=func

        if self.constant_num != None:
            if self.constant_num ==1:
                parameters = [sp.symbols(' '.join([f'C_{i}' for i in range(self.constant_num)]))]
            else:
                parameters = list(sp.symbols(' '.join([f'C_{i}' for i in range(self.constant_num)])))
            kwargs['parameters']=parameters
            sp_outputs = self.forward(sp_inputs,**kwargs)
        else:
            sp_outputs = self.forward(sp_inputs,**kwargs)

        if not self.is_function_operator and not self.vectorized and self.num_backend == 'numpy':
            if self.is_test_sp:
                if self.constant_num != None:
                    if self.op_type == 'unary':
                        lambda_sp_inputs = [sp_inputs,*parameters]
                    else:
                        lambda_sp_inputs = [*sp_inputs,*parameters]

                    sp_func = sp.lambdify(lambda_sp_inputs, sp_outputs, "numpy")
                    ins = np.concatenate([num_inputs,self.num_parameters],axis=0)
                    sp_num_outputs = sp_func(*ins)
                else:
                    sp_func = sp.lambdify(sp_inputs, sp_outputs, "numpy")
                    sp_num_outputs = sp_func(*num_inputs)
                if num_outputs != sp_num_outputs:
                    eq_Flag = False


    def get_formula(self):
        if self.op_type == 'unary':
            sp_inputs = [sp.symbols('x')]
        else:
            sp_inputs = sp.symbols('x y')
        return self.forward(*sp_inputs)

    def random_assign_parameters(self,*reset_range):
        if self.constant_num != None:
            if len(reset_range)==0:
                reset_range=[0,1]
            elif len(reset_range)==1:
                assert reset_range[0]>=0, f'Reset range must be positive!'
                reset_range=[0,reset_range]
            elif len(reset_range)==2:
                assert reset_range[1]>=reset_range[0], f'Reset range {reset_range[0]} must be smaller than range {reset_range[1]}!'
            else:
                raise ValueError('Reset range must be 1 or 2 numbers!')
            if self.num_backend == 'numpy':
                self.num_parameters = np.random.rand(self.constant_num)*(reset_range[1]-reset_range[0])+reset_range[0]
            elif self.num_backend == 'torch':
                self.num_parameters = torch.rand(self.constant_num)*(reset_range[1]-reset_range[0])+reset_range[0]
            return self.num_parameters

    def forward(self, inputs,if_skeleton=False, **kwargs):
        if self.vectorized and self.index_necessary:
            kwargs['index'] = self.choisen_index

        if self.op_type == 'unary':
            if (isinstance(inputs,sp.Basic)) or (self.vectorized and all(isinstance(item,sp.Basic) for item in inputs)):
                if self.constant_num != None and 'parameters' not in kwargs.keys():
                    kwargs['parameters'] = self.sp_parameters
                sp_func = self.op_sp_func(inputs, **kwargs)
                if self.constant_num != None:
                    subs_dict = {self.sp_parameters[i]: float(self.num_parameters[i]) for i in range(self.constant_num)}
                    if not if_skeleton:
                        sp_func = sp_func.subs(subs_dict).evalf(n=3)
                return sp_func

            #elif isinstance(inputs,(np.ndarray,int,float,torch.Tensor,np.float32)):
            else:
                if self.constant_num != None and 'parameters' not in kwargs.keys():
                    kwargs['parameters'] = self.num_parameters
                return self.op_num_func(inputs, **kwargs)
            #else:
            #    raise ValueError('Unary operation inputs type error!')

        elif self.op_type == 'binary':
            assert isinstance(inputs, (tuple, list)) and len(inputs) == 2, 'Binary operation must have two inputs!'
            if all(isinstance(item,sp.Basic) for item in inputs):
                if self.constant_num != None and 'parameters' not in kwargs.keys():
                    kwargs['parameters'] = self.sp_parameters
                sp_func = self.op_sp_func(*inputs, **kwargs)
                if self.constant_num != None:
                    subs_dict = {self.sp_parameters[i]: float(self.num_parameters[i]) for i in range(self.constant_num)}
                    if not if_skeleton:
                        sp_func = sp_func.subs(subs_dict).evalf(n=3)
                return sp_func
            #elif all(isinstance(item,(np.ndarray,float,int,torch.Tensor,np.float32)) for item in inputs):
            else:
                if self.constant_num != None and 'parameters' not in kwargs.keys():
                    kwargs['parameters'] = self.num_parameters
                return self.op_num_func(*inputs, **kwargs)
            #else:
            #    raise ValueError('Binary operation inputs type error!')

    def func_gen(self,**kwargs):
        #这里是一个生成器，用于将函数生成成新的函数
        def new_func(x,if_skeleton=False,**kwargs):
                return self.forward(x,if_skeleton=if_skeleton,**kwargs)
        return new_func

    def func_iter(self, *func,**kwargs):
        #这里是一个迭代器，用于将函数迭代成新的函数
        if self.op_type == 'unary':
            assert len(func) == 1, 'Unary operation must have one input!'
            def new_func(x,if_skeleton=False,**new_kwargs):
                if self.is_function_operator:
                    new_kwargs['func'] = func[0]
                    return self.forward(x,if_skeleton=if_skeleton,**new_kwargs)
                else:
                    return self.forward(func[0](x,if_skeleton=if_skeleton, **kwargs),if_skeleton=if_skeleton, **new_kwargs)
        else:
            assert len(func) == 2, 'Binary operation must have two inputs!'
            def new_func(x,if_skeleton=False,**new_kwargs):
                if self.is_function_operator:
                    new_kwargs['func'] = func
                    return self.forward(x,if_skeleton=if_skeleton,**new_kwargs)
                else:

                    return self.forward([func[0](x,if_skeleton=if_skeleton,**kwargs),func[1](x,if_skeleton=if_skeleton,**kwargs)],if_skeleton=if_skeleton,**new_kwargs)
        return new_func

    def __call__(self, inputs, **kwargs):
        if all(isinstance(item, (Operation,types.FunctionType)) for item in inputs):
            return self.func_iter(*inputs, **kwargs)
        else:
            return self.forward(inputs, **kwargs)



class OperationRegistrySet:
    def __init__(self,
                 op_config_path:str,
                 max_vars:int = 2,*args,**kwargs):

        with open(op_config_path, 'r') as file:
            self.configs = yaml.safe_load(file)

        # 收集所有可能的列名
        columns = set()
        for config in self.configs['operations']:
            columns.update(config.keys())
        columns = list(columns)

        # 创建一个空的DataFrame
        self.operation_info = pd.DataFrame(columns=columns)
        self.max_var_types = max_vars
        self.init_register_operations()

    @classmethod
    def from_config_yaml(cls,hyper_config_path:str,*args,**kwargs):

        config = OmegaConf.load(hyper_config_path)
        op_config_path = config.operation_tree_config.operation_config_path
        with open(hyper_config_path, 'r') as file:
            hyper_config_path= yaml.safe_load(file)['operation_tree_config']
        #op_tree_config = OmegaConf.load(hyper_config_path)['operation_tree_config']
        for k,v in hyper_config_path.items():
            if isinstance(v,dict):
                for k1,v1 in v.items():
                    kwargs[k1]=v1
            else:
                kwargs[k]=v
        return cls(op_config_path,**kwargs)


    @property
    def op_index_info(self):
        op_info = self.operation_info
        op_index_info_list = []
        for i in range(len(op_info)):
            if not op_info.iloc[i]['index_necessary']:
                op_index_info_list.append((i,None))
            else:
                for j in range(self.max_var_types):
                    op_index_info_list.append((i,j))
        return op_index_info_list

    def op_info_get_index(self,op_name,index=None):
        op_info = self.operation_info
        op_info_df = op_info[(op_info['op_name'] == op_name)]
        op_index = op_info_df.index[0]
        if op_info_df['index_necessary'].iloc[0]:
            assert index is not None, 'index is necessary'
            index_tuple = (op_index,index)
        else:
            index_tuple = (op_index,None)
        return self.op_index_info.index(index_tuple)


    def init_register_operations(self):
        for config in self.configs['operations']:
            new_df = pd.DataFrame(config, index=[0])
            self.operation_info = pd.concat([self.operation_info, new_df], ignore_index=True)
        self.operation_info = self.operation_info.where(pd.notna(self.operation_info), None)

    def operation_append(self, operation_form):
        new_df = pd.DataFrame(operation_form, index=[0])
        self.operation_info = pd.concat([self.operation_info, new_df], ignore_index=True)
        self.operation_info = self.operation_info.where(pd.notna(self.operation_info), None)

    def operation_delete(self, op_name):
        self.operation_info = self.operation_info[self.operation_info['op_name'] != op_name]

    def operation_find(self, key_dict):
        df = self.operation_info
        for key, value in key_dict.items():
            df = df[df[key] == value]
        return df

    def operation_update(self, op_name, key_dict):
        df = self.operation_find({'op_name': op_name})
        if len(df) == 1:
            for key, value in key_dict.items():
                self.operation_info.loc[self.operation_info['op_name'] == op_name, key] = value
        else:
            raise ValueError('op_name not found or key_dict not unique!')

    def generate_operation_instance(self, op_name, sp_start_index=0,**kwargs):
        operation_data = self.operation_info[self.operation_info['op_name'] == op_name].iloc[0]
        # 此处需要调整参数为新的常量
        if operation_data['constant_num'] !=None:
            operation_data['sp_start_index'] = sp_start_index
        if kwargs:
            for key in kwargs.keys():
                operation_data[key] = kwargs[key]
        opr = Operation(**operation_data.to_dict())
        return opr

    def load_yaml(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            self.configs = yaml.safe_load(file)
        self.init_register_operations()


    def __getattr__(self, operation_name):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{operation_name}'")



"""

        if any(isinstance(item,sp.core.symbol.Symbol) for item in inputs):
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.sp_parameters
            if self.op_type == 'unary':
                sp_func =  self.op_sp_func(inputs, **kwargs)
            else:
                assert isinstance(inputs, (tuple, list)) and len(inputs) == 2, 'Binary operation must have two inputs!'
                sp_func =  self.op_sp_func(*inputs,**kwargs)
            #sp_func = self.op_sp_func(inputs, **kwargs)
            if self.constant_num != None:
                subs_dict = {self.sp_parameters[i]: self.num_parameters[i] for i in range(self.constant_num)}
                if not if_skeleton:
                    sp_func = sp_func.subs(subs_dict).evalf(n=3)
            return sp_func
        #elif any(isinstance(item,np.ndarray) for item in inputs):
            #if self.constant_num != None and 'parameters' not in kwargs.keys():
            #    kwargs['parameters']= self.num_parameters
            #return self.op_num_func(inputs, **kwargs)
            #elif 'func' in kwargs.keys():
              #  return np.vectorize(self.op_num_func,excluded=kwargs)(inputs, **kwargs)
            #else:
            #    return np.vectorize(self.op_num_func)(inputs, **kwargs)
        else:
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.num_parameters
            if self.op_type == 'unary':
                return self.op_num_func(inputs, **kwargs)
            else:
                assert isinstance(inputs, (tuple, list))  and len(inputs) == 2, 'Binary operation must have two inputs!'
                return self.op_num_func(*inputs,**kwargs)
class OperationRegistrySet:
    def __init__(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            self.configs = yaml.safe_load(file)

        # 收集所有可能的列名
        columns = set()
        for config in self.configs['operations']:
            columns.update(config.keys())
        columns = list(columns)

        # 创建一个空的DataFrame
        self.operation_info = pd.DataFrame(columns=columns)
        #self.operations = {}
        self.init_register_operations()

    def init_register_operations(self):
        for config in self.configs['operations']:
            #print(config)
            #operation = operation(**config)
            #self.operations[config['op_name']] = operation

            # 将运算符的属性添加到DataFrame中
            new_df = pd.DataFrame(config, index=[0])
            self.operation_info = pd.concat([self.operation_info, new_df], ignore_index=True)
        self.operation_info = self.operation_info.where(pd.notna(self.operation_info), None)

    def operation_append(self,operation_form):
        #if isinstance(operation_form,dict):
        #    operation = operation(**operation_form)
        #elif isinstance(operation_form,operation):
        #    operation = operation_form
        #else:
        #    raise ValueError('operation_form must be dict or operation!')
        #self.operations[operation.op_name] = operation
        new_df = pd.DataFrame(operation_form, index=[0])
        self.operation_info = pd.concat([self.operation_info, new_df], ignore_index=True)
        self.operation_info = self.operation_info.where(pd.notna(self.operation_info), None)

    def operation_delete(self,op_name):
        if op_name in self.operations.keys():
            del self.operations[op_name]
            self.operation_info = self.operation_info[self.operation_info['op_name'] != op_name]
        else:
            raise ValueError('op_name not in operations!')

    def operation_find(self,key_dict):
        df = self.operation_info
        for key,value in key_dict.items():
            df = df[df[key] == value]
        return df

    def operation_update(self,op_name,key_dict):
        if op_name in self.operations.keys():
            df = self.operation_find(key_dict)
            if len(df) == 1:
                for key,value in key_dict.items():
                    self.operations[op_name].__setattr__(key,value)
                    self.operation_info.loc[self.operation_info['op_name'] == op_name,key] = value
            else:
                raise ValueError('key_dict not find unique operation!')
        else:
            raise ValueError('op_name not in operations!')


    def __getattr__(self, operation_name):
        if operation_name in self.operations.keys():
            return self.operations[operation_name]
        else:
            raise ValueError('op_name not in operations!')





class OperationRegistrySet:
    def __init__(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            self.configs = yaml.safe_load(file)

        # 收集所有可能的列名
        columns = set()
        for config in self.configs['operations']:
            columns.update(config.keys())
        columns = list(columns)

        # 创建一个空的DataFrame
        self.operation_info = pd.DataFrame(columns=columns)
        self.init_register_operations()

    def init_register_operations(self):
        for config in self.configs['operations']:
            new_df = pd.DataFrame(config, index=[0])
            self.operation_info = pd.concat([self.operation_info, new_df], ignore_index=True)
        self.operation_info = self.operation_info.where(pd.notna(self.operation_info), None)

    def operation_append(self, operation_form):
        new_df = pd.DataFrame(operation_form, index=[0])
        self.operation_info = pd.concat([self.operation_info, new_df], ignore_index=True)
        self.operation_info = self.operation_info.where(pd.notna(self.operation_info), None)

    def operation_delete(self, op_name):
        self.operation_info = self.operation_info[self.operation_info['op_name'] != op_name]

    def operation_find(self, key_dict):
        df = self.operation_info
        for key, value in key_dict.items():
            df = df[df[key] == value]
        return df

    def operation_update(self, op_name, key_dict):
        df = self.operation_find({'op_name': op_name})
        if len(df) == 1:
            for key, value in key_dict.items():
                self.operation_info.loc[self.operation_info['op_name'] == op_name, key] = value
        else:
            raise ValueError('op_name not found or key_dict not unique!')

    def generate_operation_instance(self, op_name, sp_start_index=0):
        operation_data = self.operation_info[self.operation_info['op_name'] == op_name].iloc[0]
        # 此处需要调整参数为新的常量
        if operation_data['constant_num']:
            operation_data['constants'] = sp.symbols(f'c_{sp_start_index}:{sp_start_index + operation_data["constant_num"]}')
        return operation(**operation_data.to_dict())

    def __getattr__(self, operation_name):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{operation_name}'")

"""













'''


class operation():
    def __init__(self,
                 op_name:str,
                 op_type:str,
                 op_num_func,
                 op_sp_func,
                 op_symbol:str=None,
                 is_test_sp:bool = True,
                 is_function_operator:bool = False,
                 constant_chosen=None,
                 constant_num:int = None,
                 repeat_times:int = None,
                 group_name:str=None):
        self.op_name = op_name
        self.op_type = op_type
        self.op_num_func = op_num_func
        self.op_sp_func = op_sp_func
        self.op_symbol = op_symbol
        self.repeat_times = repeat_times
        self.is_function_operator = is_function_operator
        self.group_name = group_name
        self.is_test_sp = is_test_sp
        self.constant_num = constant_num
        self.constant_chosen = constant_chosen

        if constant_num != None:
            if self.constant_chosen==None:
                self.sp_parameters = sp.symbols(' '.join(['C',]*constant_num), cls=sp.Dummy)
                self.num_parameters = np.random.rand(constant_num)
            else:
                parameters = np.random.choice(self.constant_chosen,constant_num)
                self.sp_parameters = sp.symbols(' '.join(list(parameters)))
                self.num_parameters = parameters

        self.test_feasibility()

    def test_feasibility(self):
        assert self.op_type in ['unary','binary'], 'operation type must be unary or binary!'
        eq_Flag = True
        if self.op_type == 'unary':
            sp_inputs = [sp.symbols('x')]
            num_inputs = [np.random.rand(1)]
        else:
            sp_inputs = sp.symbols('x y')
            num_inputs = [np.random.rand(1), np.random.rand(1)]
        kwargs = {}

        try:
            if self.is_function_operator:
                func = np.sin
                kwargs['func']=func
            num_outputs = self.forward(*num_inputs,**kwargs)

            if self.is_function_operator:
                func = sp.sin
                kwargs['func']=func
            if self.constant_num != None:
                if self.constant_num ==1:
                    parameters = [sp.symbols(' '.join([f'C_{i}' for i in range(self.constant_num)]))]
                else:
                    parameters = list(sp.symbols(' '.join([f'C_{i}' for i in range(self.constant_num)])))
                kwargs['parameters']=parameters
                sp_outputs = self.forward(*sp_inputs,**kwargs)
                if not self.is_function_operator:
                    sp_func = sp.lambdify(sp_inputs + parameters, sp_outputs, "numpy")
            else:
                sp_outputs = self.forward(*sp_inputs,**kwargs)
                if not self.is_function_operator:
                    sp_func = sp.lambdify(sp_inputs, sp_outputs, "numpy")

            if self.is_test_sp:
                if self.constant_num !=None:
                    ins = num_inputs + self.num_parameters
                    sp_num_outputs = sp_func(*ins)
                else:
                    sp_num_outputs = sp_func(*num_inputs)
                if num_outputs != sp_num_outputs:
                    eq_Flag = False
        except:
            raise ValueError('Function operation check error, please check function Settings!')

    def forward(self, *inputs, **kwargs):
        if self.op_type == 'unary':
            assert len(inputs) == 1, 'Unary operation must have one input!'
        elif self.op_type == 'binary':
            assert len(inputs) == 2, 'Binary operation must have two inputs!'

        if any(isinstance(item,sp.core.symbol.Symbol) for item in inputs):
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.sp_parameters
            return self.op_sp_func(*inputs,**kwargs)
        elif any(isinstance(item,np.ndarray) for item in inputs):
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.num_parameters
                return np.vectorize(self.op_num_func,excluded=kwargs)(*inputs, **kwargs)
            elif 'func' in kwargs.keys():
                return np.vectorize(self.op_num_func,excluded=kwargs)(*inputs, **kwargs)
            else:
                return np.vectorize(self.op_num_func)(*inputs, **kwargs)
        else:
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.num_parameters
            return self.op_num_func(*inputs,**kwargs)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
class operation():
    def __init__(self,
                 op_name:str,
                 op_type:str,
                 op_num_func,
                 op_sp_func,
                 op_symbol:str=None,
                 is_test_sp:bool = True,
                 is_function_operator:bool = False,
                 constant_num:int = None,
                 repeat_times:int = None):
        self.op_name = op_name
        self.op_type = op_type
        self.op_num_func = op_num_func
        self.op_sp_func = op_sp_func
        self.op_symbol = op_symbol
        self.repeat_times = repeat_times
        self.is_function_operator = is_function_operator
        self.is_test_sp = is_test_sp
        self.constant_num = constant_num
        if constant_num != None:
            self.sp_parameters = sp.symbols(' '.join(['C',]*constant_num))
            self.num_parameters = np.random.rand(constant_num)

        self.test_feasibility()

    def test_feasibility(self):
        assert self.op_type in ['unary','binary'], 'operation type must be unary or binary!'

        eq_Flag = True
        if self.op_type == 'unary':
            sp_inputs = [sp.symbols('x')]
            num_inputs = [np.random.rand(1)]
        else:
            sp_inputs = sp.symbols('x y')
            num_inputs = [np.random.rand(1), np.random.rand(1)]
        kwargs = {}

        try:
            if self.is_function_operator:
                func = np.sin
                kwargs['func']=func
            num_outputs = self.forward(*num_inputs,**kwargs)

            if self.is_function_operator:
                func = sp.sin
                kwargs['func']=func
            if self.constant_num != None:
                if self.constant_num ==1:
                    parameters = [sp.symbols(' '.join([f'C_{i}' for i in range(self.constant_num)]))]
                else:
                    parameters = list(sp.symbols(' '.join([f'C_{i}' for i in range(self.constant_num)])))
                kwargs['parameters']=parameters
                sp_outputs = self.forward(*sp_inputs,**kwargs)
                if not self.is_function_operator:
                    sp_func = sp.lambdify(sp_inputs + parameters, sp_outputs, "numpy")
            else:
                sp_outputs = self.forward(*sp_inputs,**kwargs)
                if not self.is_function_operator:
                    sp_func = sp.lambdify(sp_inputs, sp_outputs, "numpy")

            if self.is_test_sp:
                if self.constant_num !=None:
                    ins = num_inputs + self.num_parameters
                    sp_num_outputs = sp_func(*ins)
                else:
                    sp_num_outputs = sp_func(*num_inputs)
                if num_outputs != sp_num_outputs:
                    eq_Flag = False
        except:
            raise ValueError('Function operation check error, please check function Settings!')

    def forward(self, *inputs, **kwargs):
        if self.op_type == 'unary':
            assert len(inputs) == 1, 'Unary operation must have one input!'
        elif self.op_type == 'binary':
            assert len(inputs) == 2, 'Binary operation must have two inputs!'

        if any(isinstance(item,sp.core.symbol.Symbol) for item in inputs):
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.sp_parameters
            return self.op_sp_func(*inputs,**kwargs)
        elif any(isinstance(item,np.ndarray) for item in inputs):
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.num_parameters
                return np.vectorize(self.op_num_func,excluded=kwargs)(*inputs, **kwargs)
            elif 'func' in kwargs.keys():
                return np.vectorize(self.op_num_func,excluded=kwargs)(*inputs, **kwargs)
            else:
                return np.vectorize(self.op_num_func)(*inputs, **kwargs)
        else:
            if self.constant_num != None and 'parameters' not in kwargs.keys():
                kwargs['parameters']= self.num_parameters
            return self.op_num_func(*inputs,**kwargs)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


class operation():
    def __init__(self,
                 op_name:str,
                 op_type:str,
                 op_num_func:str,
                 op_sp_func:str,
                 op_symbol=None,
                 test_eq = True,):
        self.op_name = op_name
        self.op_type = op_type
        self.op_num_func = eval(op_num_func)
        self.op_sp_func = eval(op_sp_func)
        self.op_symbol = op_symbol



    def init_test_feasibility(self):
        assert self.op_type in ['unary','binary'], 'operation type must be unary or binary!'
        x,y = sp.symbols('x y')
        test_data_1 = np.random.rand(1)
        test_data_2 = np.random.rand(1)
        eq_Flag = True
        if self.op_type == 'unary':
            try:
                z = self.op_sp_func(x)
                sp_func = sp.lambdify(x, z, "numpy")
                if sp_func(test_data_1) != self.op_num_func(test_data_1):
                    eq_Flag = False
            except:
                raise ValueError('Function operation check error, please check function Settings!')
        elif self.op_type == 'binary':
            try:
                z = self.op_sp_func(x,y)
                sp_func = sp.lambdify([x,y], z, "numpy")
                if sp_func(test_data_1,test_data_2) != self.op_num_func(test_data_1,test_data_2):
                    eq_Flag = False
            except:
                raise ValueError('Function operation check error, please check function Settings!')

        if eq_Flag == False:
            raise ValueError('Sympy operation and numpy operation results are different, please check!')






        if self.op_type == 'unary':
            try:
                self.np_func = getattr(np, name)
                test_data = torch.rand(1, 1)
                self.op_func(test_data)
            except:
                raise ValueError('operation name must be a torch and numpy function!')
        elif self.op_type == 'binary':
            try:
                self.op_func = lambda x,y:eval(f'{x}{self.symbol}{y}')
                self.np_func = lambda x,y:eval(f'{x}{self.symbol}{y}')
                test_data_1 = torch.rand(1, 1)
                test_data_2 = torch.rand(1, 1)
                self.op_func(test_data_1,test_data_2)
            except:
                print('operation symbol is not be a valid python expression,test name instead!')
                self.binary_expression_flag = False

            if self.binary_expression_flag == False:
                try:
                    self.op_func = getattr(torch, name)
                    self.np_func = getattr(np, name)
                    test_data_1 = torch.rand(1, 1)
                    test_data_2 = torch.rand(1, 1)
                    self.op_func(test_data_1, test_data_2)
                except:
                    raise ValueError('operation name must be a torch and numpy function!')

        if test_eq:
            test_data = np.random.rand(1,1)










class BaseOperation(nn.Module):
    def __init__(self,op_type = None):
        super(BaseOperation, self).__init__()
        self.op_type = op_type

        if self.op_type == 'unary':
            try:
                test_data = torch.rand(1, 1)
                self.op_func(test_data)
            except:
                raise ValueError('Unary operation must be a function with one input!')
        elif self.op_type == 'binary':
            try:
                test_data = torch.rand(1, 1)
                self.op_func(test_data, test_data)
            except:
                raise ValueError('Binary operation must be a function with two inputs!')
        else:
            raise ValueError('operation type must be unary or binary!')

    def forward(self, *inputs):
        raise NotImplementedError

    def str_expression(self, input_str):
        raise NotImplementedError




class Operation_Simple(nn.Module):
    def __init__(self, symbol,name=None,op_type=None,latex=None):
        super(Operation_Simple, self).__init__()
        assert  op_type=='unary' or op_type=='binary', 'operation type must be unary or binary!'
        self.op_type = op_type
        self.binary_expression_flag = True
        self.latex_symbol = latex_symbol

        self.symbol = symbol
        if name == None:
            self.op_name = symbol
        else:
            self.op_name = name

        if self.op_type == 'unary':
            try:
                self.op_func = getattr(torch, name)
                self.np_func = getattr(np, name)
                test_data = torch.rand(1, 1)
                self.op_func(test_data)
            except:
                raise ValueError('operation name must be a torch and numpy function!')
        elif self.op_type == 'binary':
            try:
                self.op_func = lambda x,y:eval(f'{x}{self.symbol}{y}')
                self.np_func = lambda x,y:eval(f'{x}{self.symbol}{y}')
                test_data_1 = torch.rand(1, 1)
                test_data_2 = torch.rand(1, 1)
                self.op_func(test_data_1,test_data_2)
            except:
                print('operation symbol is not be a valid python expression,test name instead!')
                self.binary_expression_flag = False

            if self.binary_expression_flag == False:
                try:
                    self.op_func = getattr(torch, name)
                    self.np_func = getattr(np, name)
                    test_data_1 = torch.rand(1, 1)
                    test_data_2 = torch.rand(1, 1)
                    self.op_func(test_data_1, test_data_2)
                except:
                    raise ValueError('operation name must be a torch and numpy function!')

    def forward(self, *inputs):
        if self.op_type == 'unary':
            assert len(inputs) == 1, 'Unary operation must have one input!'
        elif self.op_type == 'binary':
            assert len(inputs) == 2, 'Binary operation must have two inputs!'
        return self.op_func(*inputs)

    def np_forward(self, *inputs):
        if self.op_type == 'unary':
            assert len(inputs) == 1, 'Unary operation must have one input!'
        elif self.op_type == 'binary':
            assert len(inputs) == 2, 'Binary operation must have two inputs!'
        return self.np_func(*inputs)

    def str_expression(self, *input_str):
        if self.op_type == 'unary':
            if '*' not in input_str:
                return self.symbol + '(' + input_str + ')'
            else:
                return self.symbol.replace('*',input_str)
        if self.op_type == 'binary' and self.binary_expression_flag == True:
            return f'{input_str[0]}{self.symbol}{input_str[1]}'

        if self.op_type == 'binary' and self.binary_expression_flag == False:
            return self.name + '(' + input_str[0] + ',' + input_str[1] + ')'

    def latex_str_expression(self, *input_str):
        if self.op_type == 'unary':
            if '*' not in input_str:
                return self.symbol + '(' + input_str + ')'
            else:
                return self.symbol.replace('*',input_str)
        if self.op_type == 'binary' and self.binary_expression_flag == True:
            return f'{input_str[0]}{self.symbol}{input_str[1]}'

        if self.op_type == 'binary' and self.binary_expression_flag == False:
            return self.name + '(' + input_str[0] + ',' + input_str[1] + ')'


class Operation_binary_symbol(BaseOperation):
    def __init__(self, name,symbol,op_type=None):
        super(Operation_binary_symbol, self).__init__(op_type=op_type)
        self.op_name = name
        try:
            self.op_func = getattr(torch, name)
        except:
            raise ValueError('operation name must be a torch function!')
        if symbol==None:
            self.symbol = name


    def forward(self, *inputs):
        if self.op_type == 'unary':
            assert len(inputs) == 1, 'Unary operation must have one input!'
        elif self.op_type == 'binary':
            assert len(inputs) == 2, 'Binary operation must have two inputs!'
        return self.op_func(*inputs)

    def str_expression(self, input_str):
        if self.op_type == 'unary':
            if '*' not in input_str:
                return self.symbol + '(' + input_str + ')'
            else:
                return self.symbol.replace('*',input_str)
        return self.symbol





class Operation_layer(BaseOperation):
    def __init__(self, name, symbol=None, op_type=None):
        super(Operation_layer, self).__init__(op_type=op_type)
        self.op_name = name
        try:
            self.op_func = getattr(torch, name)
        except:
            raise ValueError('operation name must be a torch function!')
        if symbol == None:
            self.symbol = name

    def forward(self, *inputs):
        return self.op_func(*inputs)

    def expression(self):
        return self.symbol
'''