from .tree_utils import *


def op_unary_or_binary_choisen(valid_operations,
                               node,
                               operation_tree,*args,**kwargs):
    num_children = len(list(operation_tree.successors(node)))
    node_type = 'unary' if num_children == 1 else 'binary'
    valid_operations = valid_operations[
        valid_operations['op_type'] == ("unary" if num_children == 1 else "binary")]
    if valid_operations.empty:
        raise ValueError(
            "No valid operations available. Please adjust the tree structure or operation registry.")
    return valid_operations

def op_generation_level_choisen(valid_operations,node,operation_tree,*args,**kwargs):
    if 'valid_levels' not in kwargs:
        raise ValueError("this choisen rule need valid_levels in kwargs")
    valid_levels = kwargs['valid_levels']
    node_height_level = node_height(operation_tree, node)
    valid_operations = valid_operations[valid_operations['generation_level'].isin(valid_levels)]
    valid_operations = valid_operations[
        (valid_operations['height_limit'].isnull()) | (valid_operations['height_limit'] <= node_height_level)]
    if valid_operations.empty:
        raise ValueError(
            "No valid operations available. Please adjust the tree structure or operation registry.")
    return valid_operations

def op_const_limit_choisen(valid_operations,node,operation_tree,*args,**kwargs):
    if 'max_constants_num' not in kwargs:
        raise ValueError("this choisen rule need max_constants_num in kwargs")
    if 'current_constants_num' not in kwargs:
        raise ValueError("this choisen rule need current_constants_num in kwargs")

    max_constants_num = kwargs['max_constants_num']
    current_constants_num = kwargs['current_constants_num']

    if current_constants_num >= max_constants_num:
        valid_operations = valid_operations[(valid_operations['constant_num'].isnull())]

    if valid_operations.empty:
        raise ValueError(
            "No valid operations available. Please adjust the tree structure or operation registry.")
    return valid_operations