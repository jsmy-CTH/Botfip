import numpy as np
import sympy as sp
import scipy as sc
import networkx as nx
import math
import random
import matplotlib.pyplot as plt
import torch
import yaml
from collections import deque


def renumber_tree_bfs(G, root):
    # Step 1: 使用 BFS 遍历并创建一个映射字典
    new_id = 0  # 设置一个初始值用于新的编号
    mapping = {root: root}  # 保持根节点的编号

    for node in nx.bfs_edges(G, source=root):
        if node[1] not in mapping:  # 如果子节点尚未在映射中
            new_id += 1  # 递增新的编号
            mapping[node[1]] = new_id

    # Step 2: 使用映射创建一个新的图
    H = nx.DiGraph()
    for edge in G.edges():
        new_source = mapping[edge[0]]
        new_target = mapping[edge[1]]
        H.add_edge(new_source, new_target)

    return H

def random_tree_generate(num_nodes, min_leaf_nodes, p=0.5, max_single_chain_length=2):
    G = nx.DiGraph()
    root = 0
    G.add_node(root)
    first_child = 1
    G.add_edge(root, first_child)

    available_nodes = [(first_child, 1)] # (node, chain_length)
    current_node = first_child + 1

    while current_node < num_nodes:
        if not available_nodes:
            break
        parent, chain_length = available_nodes.pop(0)

        if chain_length >= max_single_chain_length:
            num_children = 2
        else:
            num_children = 1 if random.random() < p else 2

        # 确保我们不超过num_nodes
        num_children = min(num_children, num_nodes - current_node)

        # 如果只有一个子节点，调整链长度；如果有两个子节点，重置链长度为0
        new_chain_length = 0 if num_children == 2 else chain_length + 1

        for _ in range(num_children):
            G.add_edge(parent, current_node)
            available_nodes.append((current_node, new_chain_length))
            current_node += 1

    # 如果叶子节点数目小于min_leaf_nodes
    while len([n for n, deg in G.out_degree() if deg == 0]) < min_leaf_nodes:
        choice = random.choice([1, 2])
        if choice == 1:  # 在只有一个子节点的节点上添加一个子节点
            possible_parents = [n for n, deg in G.out_degree() if deg == 1]
            if possible_parents:  # 确保我们有至少一个节点只有一个子节点
                parent = random.choice(possible_parents)
                G.add_edge(parent, current_node)
                current_node += 1
        else:  # 在一个叶节点上增加两个子节点
            parent = random.choice([n for n, deg in G.out_degree() if deg == 0])
            if parent:  # 确保我们不是在根节点上添加子节点
                G.add_edge(parent, current_node)
                current_node += 1
                G.add_edge(parent, current_node)
                current_node += 1

    G = renumber_tree_bfs(G,0)
    return G



def random_tree_generate_original(num_nodes, min_leaf_nodes, p=0.5, max_single_chain_length=2):
    G = nx.DiGraph()
    G.add_node(0)
    available_nodes = [(0, 0)]  # (node, chain_length)
    current_node = 1

    while current_node < num_nodes:
        if not available_nodes:
            break
        parent, chain_length = available_nodes.pop(0)

        if chain_length >= max_single_chain_length:
            num_children = 2
        else:
            num_children = 1 if random.random() < p else 2

        # 确保我们不超过num_nodes
        num_children = min(num_children, num_nodes - current_node)

        # 如果只有一个子节点，调整链长度；如果有两个子节点，重置链长度为0
        new_chain_length = 0 if num_children == 2 else chain_length + 1

        for _ in range(num_children):
            G.add_edge(parent, current_node)
            available_nodes.append((current_node, new_chain_length))
            current_node += 1

    # 如果叶子节点数目小于min_leaf_nodes
    while len([n for n, deg in G.out_degree() if deg == 0]) < min_leaf_nodes:
        choice = random.choice([1, 2])
        if choice == 1:  # 在只有一个子节点的节点上添加一个子节点
            possible_parents = [n for n, deg in G.out_degree() if deg == 1]
            if possible_parents:  # 确保我们有至少一个节点只有一个子节点
                parent = random.choice(possible_parents)
                G.add_edge(parent, current_node)
                current_node += 1
        else:  # 在一个叶节点上增加两个子节点
            parent = random.choice([n for n, deg in G.out_degree() if deg == 0])
            if parent:  # 确保我们不是在根节点上添加子节点
                G.add_edge(parent, current_node)
                current_node += 1
                G.add_edge(parent, current_node)
                current_node += 1

    G = renumber_tree_bfs(G,0)
    return G







def random_tree_generate_another(num_nodes, min_leaf_nodes, p=0.5, max_single_chain_length=2):
    G = nx.DiGraph()
    G.add_node(0)

    available_nodes = deque([(0, 0)])  # (node, chain_length)
    current_node = 1

    while current_node < num_nodes and available_nodes:
        parent, chain_length = available_nodes.popleft()

        if chain_length >= max_single_chain_length:
            num_children = 2
        else:
            num_children = 1 if random.random() < p else 2

        num_children = min(num_children, num_nodes - current_node)
        new_chain_length = 0 if num_children == 2 else chain_length + 1

        for _ in range(num_children):
            G.add_edge(parent, current_node)
            available_nodes.append((current_node, new_chain_length))
            current_node += 1


    # 确保叶子节点不小于min_leaf_nodes
    while len([n for n, deg in G.out_degree() if deg == 0]) < min_leaf_nodes:
        # 选择一个有一个或者没有孩子的节点
        possible_parents = [n for n, deg in G.out_degree() if deg < 2]
        if possible_parents:
            parent = random.choice(possible_parents)
            # 添加一个或两个孩子以确保达到叶子节点数量要求
            children_to_add = 2 - G.out_degree(parent)
            for _ in range(children_to_add):
                if current_node >= num_nodes: # 防止节点数超过限制
                    break
                G.add_edge(parent, current_node)
                current_node += 1
    print('current_node:', current_node)

    return G