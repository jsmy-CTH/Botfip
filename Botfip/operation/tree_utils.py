import numpy as np
import sympy as sp
import scipy as sc
import networkx as nx
import math
import random
import matplotlib.pyplot as plt
import torch
import yaml

def draw_tree(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()

def find_leaves(G, start_node):
    leaves = []
    # 定义DFS函数
    def dfs(node):
        children = list(G.successors(node))
        if not children:  # 如果该节点没有子节点，那它就是一个叶节点
            leaves.append(node)
        for child in children:
            dfs(child)

    dfs(start_node)
    return leaves

def node_height(G, node):
    if not list(G.successors(node)):
        return 0
    else:
        return max(node_height(G, child) for child in G.successors(node)) + 1


def node_sort(G,nodes_list):
    nodes_list = nodes_list.copy()
    nodes_list = list(nodes_list)
    nodes_list.sort(key=lambda x: node_height(G, x))
    return nodes_list

def matrix_nan_transformer(matrix, alpha=0.5,num_backend='numpy'):
    used_backend = np if num_backend == 'numpy' else torch
    nan_mask = used_backend.isnan(matrix)
    if matrix[nan_mask].size >= int(alpha * matrix.size):
        print('All nan matrix, plz change the tree structure.')
        return False
    non_nan_matrix = matrix[~nan_mask]
    matrix_max = used_backend.max(non_nan_matrix)
    matrix_min = used_backend.min(non_nan_matrix)
    if matrix_max > matrix_max:
        matrix_range = matrix_max - matrix_min
        matrix[nan_mask] = matrix_min - matrix_range
        new_matrix_min = used_backend.min(non_nan_matrix)
        new_matrix_range = matrix_max - new_matrix_min
        normalized_matrix = (matrix - new_matrix_min) / new_matrix_range
        eq_flag = False
        return True, normalized_matrix, nan_mask, matrix_min, matrix_max, eq_flag
    else:
        matrix[nan_mask] = matrix_max - 1
        new_matrix_min = used_backend.min(matrix)
        new_matrix_range = matrix_max - new_matrix_min
        normalized_matrix = (matrix - new_matrix_min) / new_matrix_range
        eq_flag = True
        return True, normalized_matrix, nan_mask, matrix_min, matrix_max, eq_flag

'''
def draw_tree_with_value(tree,
                         non_leaf_draw_type='symbol',
                         non_leaf_name = 'op',
                         leaf_name='symbol',
                         node_size=1000, 
                         font_size=14,
                         if_str=False):
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')
    # 创建一个标签字典
    labels = {}
    if non_leaf_name == 'name':
        op_key = 'op_name'
    elif non_leaf_name == 'symbol':
        op_key = 'op_symbol'
    else:
        raise ValueError(f'unknown draw_type:{op_draw_type}')

    leaf_nodes = [n for n, deg in tree.out_degree() if deg == 0]
    non_leaf_nodes = [n for n in tree.nodes() if n not in leaf_nodes]

    for node in tree.nodes():
        if if_str:
            labels[node] = str(node)
        else:
            if node in non_leaf_nodes:
                if 'op' in tree.nodes[node]:
                    labels[node] = getattr(tree.nodes[node]['op'],op_key)  # 这里假设你的Operation实例有一个op_name属性
                else:
                    labels[node] = str(node)  # 或其他默认值
            else:
                if var_draw_type in tree.nodes[node]:
                    labels[node] = tree.nodes[node]['symbol']
                else:
                    labels[node] = str(node)

    nx.draw(tree, pos, labels=labels, with_labels=True, arrows=True, node_size=node_size, font_size=font_size)
    plt.show()
'''