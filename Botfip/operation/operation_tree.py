from .operation import *
from .tree_utils import *
from .tree_encode import *
from .tree_generate import *
import math
from omegaconf import OmegaConf
from copy import deepcopy
import re
#from operation.original.utils import *
import torch
from torch import nn


def str_constant_range_to_list(str_constant_range):
    str_constant_range = str_constant_range[1:-1]
    str_constant_range = str_constant_range.split(',')
    return [float(i) for i in str_constant_range]



class OperationRandomTree(nn.Module):
    def __init__(self,
                 num_nodes,
                 config,
                 operation_registry_set,
                 backend = 'numpy',
):
        super().__init__()
        self.config = config
        self.operation_registry = operation_registry_set
        self.max_var_types = config.max_var_types  # 最大变量种类数，不是真正的最大变量数
        self.unary_probability = config.unary_probability  # 一元操作的概率
        self.max_single_chain_length = config.max_single_chain_length  # 单链最大长度
        self.constants_ratio_range = config.constants_ratio_range  # 常量所占叶节点的比例区间
        self.max_constants_num = config.max_constants_num  # 常量的最大数目
        self.constants_range = config.constants_range  # 常量的取值范围
        self.constants_node_dict = {}
        self.variable_node_dict = {}
        self.min_leaf_nodes = config.min_leaf_nodes
        self.constants_array = None
        self.variable_replace_probability = config.variable_replace_probability
        self.node_info = {}
        self.num_backend = backend #config.num_backend
        self.constants_num = 0
        self.op_generation_probability = {'high': 0.6, 'medium': 0.3, 'low': 0.1}
        self.if_print = config.if_print


        #if hyperparameters_yaml_path is not None:
        #    self.load_hyperparameter(hyperparameters_yaml_path)

        if self.min_leaf_nodes is None:
            self.min_leaf_nodes = max(math.ceil(num_nodes * config.min_leaf_nodes_ratio),  self.max_var_types)

        self.variable_symbols = sp.symbols(f'x_:{self.max_var_types}')

        self.operation_tree_skeleton = random_tree_generate(num_nodes, self.min_leaf_nodes, self.unary_probability, self.max_single_chain_length)

    @classmethod
    def from_config_yaml(cls,node_num,hyperparameters_yaml_path,backend='numpy'):
        opr = OperationRegistrySet.from_config_yaml(hyperparameters_yaml_path)
        config = OmegaConf.load(hyperparameters_yaml_path)
        config = config.operation_tree_config
        return cls(node_num,config,opr,backend=backend)

    @property
    def opseq(self):
        return self.tree_serialized_encode_seq()[0]

    @property
    def param(self):
        return self.constants_array



    @property
    def hyperparameters_dict(self):
        return {'max_var_types': self.max_var_types,
                'min_leaf_nodes': self.min_leaf_nodes,
                'unary_probability': self.unary_probability,
                'max_single_chain_length': self.max_single_chain_length,
                'max_constants_num': self.max_constants_num,
                'constants_ratio_range': self.constants_ratio_range,
                'constants_range': self.constants_range,
                'variable_replace_probability': self.variable_replace_probability,
                'num_backend': self.num_backend,
                'op_generation_probability': self.op_generation_probability}

    @property
    def num_nodes(self):
        return len(self.operation_tree_skeleton)

    @property
    def num_leaves(self):
        return len([n for n, deg in self.operation_tree_skeleton.out_degree() if deg == 0])

    @property
    def num_constants_node(self):
        return len(self.constants_node_dict)

    @property
    def op_num_list(self):
        op_num = []
        op_info = self.operation_registry.operation_info
        for i in range(len(op_info)):
            if op_info.iloc[i]['index_necessary']:
                op_num.append(self.max_var_types)
            else:
                op_num.append(1)
        return op_num

    @property
    def op_index_max(self):
        return sum(self.op_num_list)

    @property
    def op_index_info(self):
        op_info = self.operation_registry.operation_info
        op_index_info_list = []
        for i in range(len(op_info)):
            if not op_info.iloc[i]['index_necessary']:
                op_index_info_list.append((i,None))
            else:
                for j in range(self.max_var_types):
                    op_index_info_list.append((i,j))
        return op_index_info_list


    def op_index_get_info(self,op_index):
        return self.op_index_info[op_index]

    def op_info_get_index(self,op_name,index=None):
        op_info = self.operation_registry.operation_info
        op_info_df = op_info[(op_info['op_name'] == op_name)]
        op_index = op_info_df.index[0]
        if op_info_df['index_necessary'].iloc[0]:
            assert index is not None, 'index is necessary'
            index_tuple = (op_index,index)
        else:
            index_tuple = (op_index,None)

        return self.op_index_info.index(index_tuple)

    def _check_leaf_type(self,node):
        leaf_list = find_leaves(self.operation_tree_skeleton,node)
        var_index_list = []
        for leaf in leaf_list:
            leaf_type = self.node_info[leaf]['type']
            if leaf_type == 'var':
                var_index = int(self.node_info[leaf]['index'])
                var_index_list.append(var_index)
        return var_index_list



    def _random_assign_operations(self,non_leaf_nodes,current_node_with_constant_index):
        operation_info = self.operation_registry.operation_info
        operation_info = operation_info[operation_info['repeat_times'] != 'Never']
        operation_count = {row['op_name']: 0 for _, row in operation_info.iterrows()}
        valid_levels = self.op_generation_probability.keys()
        cannot_ad_op_list = list(self.operation_registry.operation_info[(self.operation_registry.operation_info['adjacent_repeatable'].notnull()) & (
                    self.operation_registry.operation_info['adjacent_repeatable'] == False)]['op_name'].unique())
        #non_leaf_nodes = node_sort(self.operation_tree_skeleton, non_leaf_nodes)

        for node in non_leaf_nodes:
            valid_operations = operation_info.copy()
            if node == 0:
                valid_operations = valid_operations[valid_operations['op_name'] == 'linear']
            else:
                valid_operations = valid_operations[valid_operations['generation_level'].isin(valid_levels)]
                # 根据该节点的子节点数量（unary 或 binary）筛选运算符
                num_children = len(list(self.operation_tree_skeleton.successors(node)))
                node_type = 'unary' if num_children == 1 else 'binary'
                valid_operations = valid_operations[
                    valid_operations['op_type'] == ("unary" if num_children == 1 else "binary")]

                ad_node_op_list = list(self.operation_tree_skeleton.predecessors(node))+list(self.operation_tree_skeleton.successors(node))
                for ad_node in ad_node_op_list:
                    if 'op' in self.operation_tree_skeleton.nodes[ad_node].keys():
                        op_name = self.operation_tree_skeleton.nodes[ad_node]['op'].op_name
                        if op_name in cannot_ad_op_list:
                            valid_operations = valid_operations[valid_operations['op_name'] != op_name]

                node_height_level = node_height(self.operation_tree_skeleton, node)
                valid_operations = valid_operations[(valid_operations['height_limit'].isnull()) | (valid_operations['height_limit'] <= node_height_level)]
                if self.constants_num >= self.max_constants_num:
                    valid_operations = valid_operations[(valid_operations['constant_num'].isnull())]


            if valid_operations.empty:
                raise ValueError(
                    "No valid operations available. Please adjust the tree structure or operation registry.")

            existing_levels = valid_operations['generation_level'].unique()
            existing_probabilities = [self.op_generation_probability[level] for level in existing_levels]
            normalized_probabilities = np.array(existing_probabilities) / sum(existing_probabilities)
            sample_grouped = valid_operations.groupby('generation_level')
            # 使用self.op_generation_probability为每个分组生成采样概率
            # 根据分组概率选择一个generation_level
            chosen_group_name = np.random.choice(list(sample_grouped.groups.keys()), p=normalized_probabilities)
            chosen_group = sample_grouped.get_group(chosen_group_name)
            chosen_operation = chosen_group.sample(1).iloc[0]
            chosen_operation_name = chosen_operation['op_name']
            op_kwargs = {}

            if chosen_operation['vectorized'] and chosen_operation['index_necessary']:
                leaf_index_list = self._check_leaf_type(node)
                if len(leaf_index_list) == 0:
                    raise ValueError("No valid leaf variable index found")
                choisen_index = random.choice(leaf_index_list)
                op_kwargs['choisen_index'] = choisen_index
                
            op_kwargs['num_backend'] = self.num_backend

            if chosen_operation['constant_num'] != None:
                operation_instance = self.operation_registry.generate_operation_instance(chosen_operation_name, current_node_with_constant_index, **op_kwargs)
                self.constants_num += chosen_operation['constant_num']
                self.node_info[node] = {'op_name': operation_instance.op_name,
                                        'constant_num': operation_instance.constant_num,
                                        'const_parameters_index':operation_instance.const_parameters_index,}
                self.constants_node_dict[node] = operation_instance.sp_parameters
                current_node_with_constant_index += operation_instance.constant_num
            else:
                operation_instance = self.operation_registry.generate_operation_instance(chosen_operation_name, **op_kwargs)
                self.node_info[node] = {'op_name': operation_instance.op_name}

            if chosen_operation['vectorized'] and chosen_operation['index_necessary']:
                self.node_info[node]['index'] = operation_instance.choisen_index
            op_index = operation_instance.choisen_index if chosen_operation['index_necessary'] else None
            self.node_info[node]['op_index'] = self.op_info_get_index(chosen_operation_name,op_index)

            self.operation_tree_skeleton.nodes[node]['op']= operation_instance
            if self.if_print:
                print(f"node {node}，node type {node_type},assigned operation {chosen_operation_name},start index:{current_node_with_constant_index}")
            operation_count[chosen_operation_name] += 1
            if chosen_operation['repeat_times']!=None and operation_count[chosen_operation_name] >= chosen_operation['repeat_times']:
                operation_info = operation_info[operation_info['op_name'] != chosen_operation_name]

        return current_node_with_constant_index

    def check_node(self):
        leaf_nodes = [node for node in self.operation_tree_skeleton.nodes() if self.operation_tree_skeleton.out_degree(node) == 0]
        binary_node =  [node for node in self.operation_tree_skeleton.nodes() if self.operation_tree_skeleton.out_degree(node) == 2]
        unary_node =  [node for node in self.operation_tree_skeleton.nodes() if self.operation_tree_skeleton.out_degree(node) == 1]

        for node in binary_node:
            op = self.operation_tree_skeleton.nodes[node]['op']
            if op.op_type != 'binary':
                #print(self.node_info)
                #self.draw_tree()
                #self.draw_tree(if_str=True)
                raise ValueError(f"node {node} is binary node,but op type is {op.op_type}")


        for node in unary_node:
            op = self.operation_tree_skeleton.nodes[node]['op']
            if op.op_type != 'unary':
                #print(self.node_info)
                #self.draw_tree()
                #self.draw_tree(if_str=True)
                raise ValueError(f"node {node} is unary node,but op type is {op.op_type}")

        for node in leaf_nodes:
            op_name = self.operation_tree_skeleton.nodes[node]['op'].op_name
            op_name = op_name.split('_')[0]
            if op_name not in ['slice','const']:
                #print(self.node_info)
                #self.draw_tree()
                #self.draw_tree(if_str=True)
                raise ValueError(f"node {node} is leaf node,but op name is {op_name}")





    def _random_assign_var(self,leaf_nodes,current_node_with_constant_index):
        leaf_count = len(leaf_nodes)
        leaf_symbols = [None] * leaf_count
        leaf_type = [None] * leaf_count

        # parent_to_leaves = {leaf: list(opt.operation_tree_skeleton.predecessors(leaf))[0] for leaf in leaf_nodes}
        parent_to_leaves = {}
        for leaf in leaf_nodes:
            parent = list(self.operation_tree_skeleton.predecessors(leaf))[0]
            if parent not in parent_to_leaves:
                parent_to_leaves[parent] = []
            parent_to_leaves[parent].append(leaf)

        unary_leaves = [leaf_list[0] for parent, leaf_list in parent_to_leaves.items() if len(leaf_list) == 1 and self.operation_tree_skeleton.out_degree(parent) == 1]
        binary_parents = [parent for parent, children in parent_to_leaves.items() if len(children) == 2]

        for leaf in unary_leaves:
            leaf_index = leaf_nodes.index(leaf)
            sample_index = np.random.choice(range(len(self.variable_symbols)))
            leaf_symbols[leaf_index] = self.variable_symbols[sample_index]
            leaf_type[leaf_index] = {'type': 'var', 'index': sample_index}

        for parent in binary_parents:
            children = parent_to_leaves[parent]
            children_index = np.random.choice(2)
            leaf_index = leaf_nodes.index(children[children_index])
            if leaf_symbols[leaf_index] is None:
                sample_index = np.random.choice(range(len(self.variable_symbols)))
                leaf_symbols[leaf_index] = self.variable_symbols[sample_index]
                leaf_type[leaf_index] = {'type': 'var', 'index': sample_index}

        remaining_leaves = [leaf for leaf, symbol in enumerate(leaf_symbols) if symbol is None]
        num_remaining = len(remaining_leaves)
        if num_remaining != 0:

            ratio = np.random.uniform(*self.constants_ratio_range)
            expected_num_vars = math.ceil(leaf_count * (1 - ratio))
            current_num_vars = leaf_count - num_remaining
            if current_num_vars > expected_num_vars:
                num_vars = 0
                num_consts = num_remaining - num_vars
            else:
                num_vars = expected_num_vars - current_num_vars
                num_consts = num_remaining - num_vars

            #num_vars = math.ceil(num_remaining * (1 - ratio))
            if self.if_print:
                print(f"num_remaining:{num_remaining}")
                print(f"res_num_vars:{num_vars}")
                print(f"res_num_consts:{num_consts}")

            if num_vars > 0:
                #remaining_leaves_var_list = np.random.choice(self.variable_symbols, num_vars, replace=True).tolist()
                sample_selected_indices = np.random.choice(range(len(self.variable_symbols)), num_vars, replace=True)
                remaining_leaves_var_list = [self.variable_symbols[i] for i in sample_selected_indices]

                empty_indices = [index for index, symbol in enumerate(leaf_symbols) if symbol is None]
                chosen_index = np.random.choice(empty_indices, num_vars, replace=False)

                for index, symbol in zip(chosen_index, remaining_leaves_var_list):
                    leaf_symbols[index] = symbol
                    leaf_type[index] = {'type': 'var', 'index': self.variable_symbols.index(symbol)}
            else:
                if self.if_print:
                    print("No res variables to assign.")

            unique_vars = set(leaf_symbols)
            if None in unique_vars:
                unique_vars.remove(None)
            for symbol in self.variable_symbols:
                if symbol not in unique_vars:
                    if np.random.rand() < self.variable_replace_probability:
                        # Find a duplicate
                        counts = {s: leaf_symbols.count(s) for s in unique_vars if leaf_symbols.count(s) > 1}
                        if counts:
                            to_replace = max(counts, key=counts.get)
                            replace_idx = leaf_symbols.index(to_replace)
                            leaf_symbols[replace_idx] = symbol
                            leaf_type[replace_idx] = {'type': 'var', 'index': self.variable_symbols.index(symbol)}
                            unique_vars.add(symbol)

            if num_consts > 0:
                remaining_leaves_constant_list = [sp.Symbol(f'C_{i}') for i in range(current_node_with_constant_index,
                                                                                     current_node_with_constant_index + num_consts)]
                empty_indices = [index for index, symbol in enumerate(leaf_symbols) if symbol is None]
                #chosen_index = np.random.choice(empty_indices, num_consts, replace=False)
                #print()
                for index, symbol in zip(empty_indices, remaining_leaves_constant_list):
                    leaf_symbols[index] = symbol
                    leaf_type[index] = {'type': 'const', 'const_parameters_index': [current_node_with_constant_index]}
                    self.constants_num += 1
                    current_node_with_constant_index += 1
            else:
                if self.if_print:
                    print("No res constants to assign.")

        for leaf, symbol in enumerate(leaf_symbols):
            self.operation_tree_skeleton.nodes[leaf_nodes[leaf]]['symbol'] = symbol
            self.node_info[leaf_nodes[leaf]] = leaf_type[leaf]
            if leaf_type[leaf]['type'] == 'var':
                op_name = 'slice'
                slice_operation_instance = self.operation_registry.generate_operation_instance('slice',choisen_index=leaf_type[leaf]['index'])
                self.node_info[leaf_nodes[leaf]]['op_name'] = op_name
                self.variable_node_dict[leaf_nodes[leaf]] = symbol
                self.operation_tree_skeleton.nodes[leaf_nodes[leaf]]['op'] = slice_operation_instance
                self.node_info[leaf_nodes[leaf]]['op_index'] = self.op_info_get_index(op_name, leaf_type[leaf]['index'])
            else:
                self.constants_node_dict[leaf_nodes[leaf]] = symbol
                const_func = self.operation_registry.generate_operation_instance('const')
                self.node_info[leaf_nodes[leaf]]['op_name'] = 'const'
                self.operation_tree_skeleton.nodes[leaf_nodes[leaf]]['op'] = const_func
                self.node_info[leaf_nodes[leaf]]['op_index'] = self.op_info_get_index('const')
            if self.if_print:
              print(f"leaf {leaf_nodes[leaf]} assigned symbol {symbol}")
        return current_node_with_constant_index

    def save_tree_info(self):
        treepos_seq, nodeind_seq, op_info_seq, constants_array = self.tree_serialized_encode()
        encoder_vector_dict = {'treepos_seq': treepos_seq,
                                 'nodeind_seq': nodeind_seq,
                                 'op_info_seq': op_info_seq,
                                 'constants_array': constants_array}
        hyperparameters_dict = self.hyperparameters_dict
        opr_info_df = self.operation_registry.operation_info

        inputs = self.variable_symbols
        func = self.func_iteration(0)
        skeleton_formula =  str(func(inputs, if_skeleton=True))
        return encoder_vector_dict, hyperparameters_dict, opr_info_df,self.node_info,skeleton_formula

    def update_from_dict(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def load_hyperparameter(self,hyperparameters_yaml,):
        with open(hyperparameters_yaml, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
        self.update_from_dict(config_data)

    def load_tree(self,
                  encoder_vector_dict,
                  operation_yaml=None,
                  hyperparameters_yaml=None,
                  root_default_op = 'linear'):

        encoder_vector_dict = deepcopy(encoder_vector_dict)

        if hyperparameters_yaml is not None:
            with open(hyperparameters_yaml, 'r') as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            self.update_from_dict(config_data)
        if operation_yaml is not None:
            self.operation_registry = OperationRegistrySet(operation_yaml)
        tree_op_seq = encoder_vector_dict['tree_op_seq']
        if root_default_op is not None:
            op_index = self.op_info_get_index(root_default_op)
            tree_op_seq.insert(0,op_index + 1)
        tree_seq,op_info_seq = parse_tree_node_op_split(tree_op_seq)
        constants_array = encoder_vector_dict['constants_array']
        self.tree_serialized_recon(tree_seq, op_info_seq, constants_array)

    @classmethod
    def load_tree_ots(cls,
                     ots,
                     hyperparameters_yaml,
                     constant_array=None,
                     #operation_yaml=None,
                     root_default_op='linear',
                     backend='numpy',):

        ots = deepcopy(ots)
        if constant_array is not None:
            constant_array = deepcopy(constant_array)

        encoder_vector_dict = {
            'tree_op_seq': ots,
            'constants_array': constant_array,
        }

        num_nodes = len(ots)//2
        config = OmegaConf.load(hyperparameters_yaml)
        config = config['operation_tree_config']
        operation_yaml = config.operation_config_path
        max_vars = config.max_var_types
        operation_registry_set = OperationRegistrySet(operation_yaml,max_vars=max_vars)

        opt = cls(num_nodes,config,operation_registry_set,backend=backend)
        opt.load_tree(encoder_vector_dict,root_default_op = root_default_op)
        return opt


    def random_assign_operations(self):
        self.reset_tree_state()
        # 为非叶节点分配运算符
        leaf_nodes = [n for n, deg in self.operation_tree_skeleton.out_degree() if deg == 0]
        non_leaf_nodes = [n for n in self.operation_tree_skeleton.nodes() if n not in leaf_nodes]
        leaf_nodes.sort()
        non_leaf_nodes.sort()
        current_node_with_constant_index = 0
        current_node_with_constant_index = self._random_assign_var(leaf_nodes, current_node_with_constant_index)
        current_node_with_constant_index = self._random_assign_operations(non_leaf_nodes,current_node_with_constant_index)
        self.random_assign_num_parameters()
        self.random_assign_sp_parameters()
        if self.if_print:
            print(f'random_assign_operations done, constants_num:{self.constants_num}')

    def reset_tree_state(self):
        # 重置树的状态
        if self.operation_tree_skeleton is not None:
            for node in self.operation_tree_skeleton.nodes():
                self.operation_tree_skeleton.nodes[node].clear()
        self.node_info.clear()
        self.constants_node_dict.clear()
        self.variable_node_dict.clear()
        self.constants_array = None
        self.constants_num = 0

    def reset_tree_structure(self):
        # 重置树的结构
        self.operation_tree_skeleton.clear()
        self.operation_tree_skeleton = random_tree_generate(self.num_nodes, self.min_leaf_nodes, self.unary_probability,self.max_single_chain_length)
        self.reset_tree_state()

    def random_assign_num_parameters(self):
        constants_array = np.zeros(self.constants_num)
        constant_node_list = list(self.constants_node_dict.keys())
        constant_node_list.sort()
        leaf_nodes = [n for n, deg in self.operation_tree_skeleton.out_degree() if deg == 0]

        current_index = 0
        for node in constant_node_list:
            const_list = self.constants_node_dict[node]
            if hasattr(const_list, '__len__'):
                const_len = len(const_list)
            else:
                const_len = 1
            const_range = self.constants_range
            if node not in leaf_nodes:
                op_index = self.node_info[node]['op_index']
                if self.operation_registry.operation_info.loc[op_index,'constant_range'] is not None:
                    const_range = str_constant_range_to_list(self.operation_registry.operation_info.loc[op_index,'constant_range'])

            constants_array[current_index:current_index + const_len] = np.random.uniform(*const_range, const_len)
            current_index += const_len

        if self.num_backend == 'torch':
            self.constants_array = torch.nn.Parameter(torch.tensor(constants_array))
        elif self.num_backend == 'numpy':
            self.constants_array = constants_array
        else:
            raise NotImplementedError


    def set_num_parameters(self,constants_array):
        if isinstance(constants_array, (np.ndarray,torch.Tensor,torch.nn.parameter.Parameter,type(None))):
            if self.num_backend == 'numpy' and isinstance(constants_array, (torch.Tensor,torch.nn.parameter.Parameter)):
                self.constants_array = constants_array.detach().numpy()
            elif self.num_backend == 'torch' and isinstance(constants_array, (np.ndarray,torch.Tensor)):
                self.constants_array = torch.nn.Parameter(torch.tensor(constants_array))
            else:
                self.constants_array = constants_array
        else:
            raise ValueError(f'constants_array type:{type(constants_array)}')

        # assert self.operation_tree_skeleton.nodes[node]['op'].num_parameters.shape == original_parameters_shape
        # print(f"node {node} assigned numpy parameters {self.operation_tree_skeleton.nodes[node]['op'].num_parameters}")

    def update_num_parameters(self):
            self.constants_array = self.constants_array


    def random_assign_sp_parameters(self):
        contain_constants_nodes = self.constants_node_dict.keys()
        for node in contain_constants_nodes:
            sp_parameters = self.constants_node_dict[node]
            if not isinstance(sp_parameters, (tuple, list)):
                sp_parameters = [sp_parameters]
            # parameters_index = self.node_info[node]['const_parameters_index']
            # original_parameters_shape = self.operation_tree_skeleton.nodes[node]['op'].num_parameters.shape
            # self.operation_tree_skeleton.nodes[node]['op'].num_parameters = self.constants_array[parameters_index]
            self.operation_tree_skeleton.nodes[node]['op'].sp_parameters = sp_parameters

    def func_iteration(self,node,if_skeleton=False,**kwargs):
        #核心功能，用于遍历树中节点构成的函数，组合成整体
        leaf_nodes = [n for n, deg in self.operation_tree_skeleton.out_degree() if deg == 0]
        non_leaf_nodes = [n for n in self.operation_tree_skeleton.nodes() if n not in leaf_nodes]

        if node in leaf_nodes:
            return self.operation_tree_skeleton.nodes[node]['op'].func_gen(if_skeleton=if_skeleton,**kwargs)
        else:
            children = list(self.operation_tree_skeleton.successors(node))
            op_list = []
            for child in children:
                op_list.append(self.func_iteration(child,**kwargs))
            return self.operation_tree_skeleton.nodes[node]['op'].func_iter(*op_list,**kwargs)

    def vec_func_iteration(self,node,**kwargs):
        new_func = self.func_iteration(node,**kwargs)
        def return_func(x,*args,**kwargs):
            assert isinstance(x,(np.ndarray,float,int,torch.Tensor)), f'x type:{type(x)}'
            if self.num_backend=='numpy':
                vec_func = np.vectorize(new_func, signature='(n)->()')
            elif self.num_backend=='torch':
                vec_func = torch.func.vmap(new_func)
            output = vec_func(x, *args, **kwargs)
            return output #matrix_nan_transformer(output)
        return return_func

    def tree_serialized_encode_dict(self):
        assert len(self.node_info) != 0, 'node_info is empty, please run random_assign_operations first'
        tree_seq = serialize_tree(self.operation_tree_skeleton,0)
        tree_seq = [i+1 if isinstance(i,int) else 0 for i in tree_seq]
        op_info_seq = []
        for node in self.operation_tree_skeleton.nodes():
            op_name = self.operation_tree_skeleton.nodes[node]['op'].op_name
            if '_' in op_name:
                op_name = op_name.split('_')[0]
            op_index = self.node_info[node]['op_index']
            op_info_seq.append(op_index)
        treeflag_seq, nodeind_seq = tree_seq_split(tree_seq)
        treeflag_seq = nposseq2intseq(treeflag_seq)
        return treeflag_seq,nodeind_seq,op_info_seq,self.constants_array

    def tree_serialized_encode_seq(self,if_ignore_root = True):
        assert len(self.node_info) != 0, 'node_info is empty, please run random_assign_operations first'
        tree_seq = serialize_tree(self.operation_tree_skeleton,0)
        serialized_with_value = []
        for node in tree_seq:
            if node != 'end':
                serialized_with_value.append(self.node_info[node]['op_index']+1)
            else:
                serialized_with_value.append(0)
        if if_ignore_root:
            serialized_with_value = serialized_with_value[1:]
        return serialized_with_value,self.constants_array

    def to_formula_str(self,type = 'simplified'):
        assert type in ['simplified','full','skeleton'], f'type:{type} not supported'
        inputs = self.variable_symbols
        func = self.func_iteration(0)

        if type == 'simplified' or type == 'skeleton':
            f_str = str(func(inputs, if_skeleton=True))
        else:
            f_str = str(func(inputs, if_skeleton=False))

        if type == 'simplified':
            pattern = r'C_\d+'
            f_str = re.sub(pattern, 'C', f_str)

        return f_str




    def tree_serialized_recon(self,tree_seq,op_info_seq,constants_array):
#        assert isinstance(constants_array, (np.ndarray,torch.Tensor)), f'constants_seq type:{type(constants_array)}'

        #treeflag_seq = intseq2nposseq(treeflag_seq)
        #tree_seq = splitseq_to_treeseq(treeflag_seq,nodeind_seq)

        self.reset_tree_state()
        self.constants_num = 0
        tree_seq_clone = tree_seq.copy()
        tree_seq_clone = [i-1 if i!=0 else 'end' for i in tree_seq_clone]
        tree_structure = deserialize_tree(tree_seq_clone)
        self.operation_tree_skeleton = tree_structure

        leaf_nodes = [n for n, deg in self.operation_tree_skeleton.out_degree() if deg == 0]
        non_leaf_nodes = [n for n in self.operation_tree_skeleton.nodes() if n not in leaf_nodes]
        leaf_nodes.sort()
        non_leaf_nodes.sort()
        current_node_with_constant_index = 0

        for node in leaf_nodes:
            op_index = op_info_seq[node]
            op_df_index,ind = self.op_index_get_info(op_index)
            op_name = self.operation_registry.operation_info.iloc[op_df_index]['op_name']
            if op_name == 'slice':
                symbol = self.variable_symbols[ind]
                self.operation_tree_skeleton.nodes[node]['symbol'] = symbol
                self.node_info[node] =  {'type': 'var',
                                         'index': ind,
                                         'op_name':op_name,
                                         'op_index':op_index,}
                slice_operation_instance = self.operation_registry.generate_operation_instance('slice',choisen_index=ind)
                self.variable_node_dict[node] = symbol
                self.operation_tree_skeleton.nodes[node]['op'] = slice_operation_instance
            else:
                symbol = sp.Symbol(f'C_{current_node_with_constant_index}')
                self.constants_node_dict[node] = symbol
                const_func = self.operation_registry.generate_operation_instance('const')
                self.operation_tree_skeleton.nodes[node]['op'] = const_func
                self.operation_tree_skeleton.nodes[node]['symbol'] = symbol
                self.node_info[node] =  {'type': 'const',
                                         'const_parameters_index': [current_node_with_constant_index],
                                            'op_name':op_name,
                                             'op_index':op_index,}
                current_node_with_constant_index += 1

        for node in non_leaf_nodes:
            op_index = op_info_seq[node]
            op_df_index,ind = self.op_index_get_info(op_index)
            chosen_operation = self.operation_registry.operation_info.iloc[op_df_index]
            op_name = chosen_operation['op_name']
            op_kwargs = {}
            if chosen_operation['vectorized'] and chosen_operation['index_necessary']:
                op_kwargs['choisen_index'] = ind

            op_kwargs['num_backend'] = self.num_backend
            if chosen_operation['constant_num'] != None:
                operation_instance = self.operation_registry.generate_operation_instance(op_name,
                                                                                         current_node_with_constant_index,
                                                                                         **op_kwargs)
                self.node_info[node] = {'op_name': operation_instance.op_name,
                                        'constant_num': operation_instance.constant_num,
                                        'const_parameters_index': operation_instance.const_parameters_index, }
                self.constants_node_dict[node] = operation_instance.sp_parameters
                current_node_with_constant_index += operation_instance.constant_num
            else:
                operation_instance = self.operation_registry.generate_operation_instance(op_name,
                                                                                         **op_kwargs)
                self.node_info[node] = {'op_name': operation_instance.op_name}

            if chosen_operation['vectorized'] and chosen_operation['index_necessary']:
                self.node_info[node]['index'] = operation_instance.choisen_index
            self.node_info[node]['op_index'] = op_index
            self.operation_tree_skeleton.nodes[node]['op'] = operation_instance
            if self.if_print:

                print(  f"node {node},assigned operation {op_name},start index:{current_node_with_constant_index}")

        self.constants_num = current_node_with_constant_index
        if isinstance(constants_array, (np.ndarray,torch.Tensor,torch.nn.parameter.Parameter,type(None))):
            if self.num_backend == 'numpy' and isinstance(constants_array, (torch.Tensor,torch.nn.parameter.Parameter)):
                self.constants_array = constants_array.detach().numpy()
            elif self.num_backend == 'torch' and isinstance(constants_array, (np.ndarray,torch.Tensor)):
                self.constants_array = torch.nn.Parameter(torch.tensor(constants_array))
            elif constants_array is None:
                self.random_assign_num_parameters()
                #self.constants_array = constants_array
        else:
            raise ValueError(f'constants_array type:{type(constants_array)}')
        contain_constants_nodes =  self.constants_node_dict.keys()
        for node in contain_constants_nodes:
            sp_parameters = self.constants_node_dict[node]
            if not isinstance(sp_parameters, (tuple, list)):
                sp_parameters = [sp_parameters]
            self.operation_tree_skeleton.nodes[node]['op'].sp_parameters = sp_parameters

    def formula_image(self,mesh_grid):
        var_num = mesh_grid.shape[-1]
        assert var_num == self.max_var_types, f'mesh_grid shape:{mesh_grid.shape},variable_num:{self.max_var_types}'
        original_mesh_grid_shape = mesh_grid.shape
        mesh_grid = mesh_grid.reshape(-1,var_num)
        func = self.vec_func_iteration(0)
        output_image = func(mesh_grid)
        output_image = output_image.reshape(original_mesh_grid_shape[:-1])
        return output_image



    def draw_tree(self,draw_type='symbol',node_size=1000, font_size=14,if_str=False):
        pos = nx.nx_agraph.graphviz_layout(self.operation_tree_skeleton, prog='dot')

        # 创建一个标签字典
        labels = {}
        if draw_type == 'name':
            op_key = 'op_name'
        elif draw_type == 'symbol':
            op_key = 'op_symbol'
        else:
            raise ValueError(f'unknown draw_type:{draw_type}')
        leaf_nodes = [n for n, deg in self.operation_tree_skeleton.out_degree() if deg == 0]
        non_leaf_nodes = [n for n in self.operation_tree_skeleton.nodes() if n not in leaf_nodes]

        for node in self.operation_tree_skeleton.nodes():
            if if_str:
                labels[node] = str(node)
            else:
                if node in non_leaf_nodes:
                    if 'op' in self.operation_tree_skeleton.nodes[node]:
                        labels[node] = getattr(self.operation_tree_skeleton.nodes[node]['op'],op_key)  # 这里假设你的Operation实例有一个op_name属性
                    else:
                        labels[node] = str(node)  # 或其他默认值
                else:
                    if 'symbol' in self.operation_tree_skeleton.nodes[node]:
                        labels[node] = self.operation_tree_skeleton.nodes[node]['symbol']
                    else:
                        labels[node] = str(node)

        nx.draw(self.operation_tree_skeleton, pos, labels=labels, with_labels=True, arrows=True, node_size=node_size, font_size=font_size)
        plt.show()

    def forward(self,x):
        assert isinstance(x,torch.Tensor), f'x type:{type(x)}'
        x_shape = x.shape
        x = x.reshape(-1,x_shape[-1])
        self.update_num_parameters()
        try:
            func = self.vec_func_iteration(0)
            output = func(x)
        except Exception as e:
            self.draw_tree()
            raise e
        output = output.reshape(x_shape[:-1])
        return output

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'constants_array' and value is not None:
            contain_constants_nodes = self.constants_node_dict.keys()
            for node in contain_constants_nodes:
                parameters_index = self.node_info[node]['const_parameters_index']
                original_parameters_shape = self.operation_tree_skeleton.nodes[node]['op'].num_parameters.shape
                self.operation_tree_skeleton.nodes[node]['op'].num_parameters = self.constants_array[parameters_index]
                assert self.operation_tree_skeleton.nodes[node]['op'].num_parameters.shape == original_parameters_shape
                if self.if_print:
                    print(f"node {node} assigned numpy parameters {self.operation_tree_skeleton.nodes[node]['op'].num_parameters}")






